import glob
import json
import os
import time
import warnings
import torch
from pathlib import Path
from tqdm import tqdm
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel
from pybo.optimizer.sobol import SobolOptimizer
from pybo.optimizer.bayesian import BayesianOptimizer
from pybo.samplers.sobol import SobolSampler
from pybo.utils.cli import parse_trial_args, default_output_dir, resolve_device, unique_dir
from tutorials.multi_objective.vformac.objective import VFormAC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

# The initial population is read from the records. Anchored on __file__ rather than the
# cwd, because the loop chdirs into each step directory as it goes.
CAMPAIGN_DIR = Path(__file__).resolve().parents[4] / "data" / "vformac"


def load_initial_design(objective, device: torch.device = DEVICE, source: str = "sobol"):
    """The campaign's measured observations of one provenance, as (X, Y_obj, Y_trk).

    `source` is the provenance to take, matched against each record's own field: "sobol"
    is the design this campaign started from, and the default for that reason. Every other
    record - the proposals the run went on to make - is left out.

    Read by label from the ported experiment.json records rather than by column position,
    so a reordered problem cannot silently transpose the data. These are real measurements:
    they carry no variance, and none is invented for them.
    """
    par_labels = [cfg.label for cfg in objective.par_cfg or []]
    obj_labels = [cfg.label for cfg in objective.obj_cfg or []]
    trk_labels = [cfg.label for cfg in objective.trk_cfg or []]

    X, Y_obj, Y_trk = [], [], []
    for path in sorted(glob.glob(str(CAMPAIGN_DIR / "*" / "experiment.json"))):
        record = json.loads(Path(path).read_text(encoding="utf-8"))["data"][0]
        if record["source"] != source:
            continue
        X.append([record["parameters"][label] for label in par_labels])
        Y_obj.append([record["objectives"][label] for label in obj_labels])
        Y_trk.append([record["trackers"][label] for label in trk_labels])
    if not X:
        raise SystemExit(f"No {source!r} observations under {CAMPAIGN_DIR}. "
                         f"Run convert_metadata.py first.")

    def tensor(rows):
        return torch.tensor(rows, device=device, dtype=DTYPE)

    return tensor(X), tensor(Y_obj), tensor(Y_trk)


def main(output_dir: Path, n_evals=64, q: int = 1, n_initial: int = None, seed: int = 2063,
         verbose: bool = True, device: torch.device = DEVICE, strategy: str = "bo",
         repeats: int = 1, noise: bool = False):
    """ Make directory """""
    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting optimization ({n_evals} evals, q={q}, seed={seed})")

    """ Seed the global torch RNG to ensure reproducibility. """
    torch.manual_seed(seed)

    """ Instantiate true objective """
    objective = VFormAC(
        device=device,
        dtype=DTYPE
    )

    """ Instantiate kernel """
    kernel = ScaleKernel(
        base_kernel=RBFKernel(
            ard_num_dims=objective.num_par,
            lengthscale_constraint=Interval(
                lower_bound=1e-4,
                upper_bound=1.0
            ),
        ),
        outputscale_constraint=Interval(
            lower_bound=1e-4,
            upper_bound=1e2
        ),
    )

    """ Draw the initial parameter set """
    X_initial, Y_obj_initial, Y_trk_initial = load_initial_design(objective, device)
    # Whole batches only, so a partial last batch never mixes measured with proposed rows.
    n_initial = (len(X_initial) // q) * q
    X_initial = X_initial[:n_initial]
    sampler = SobolSampler(device=device, dtype=DTYPE, objective=objective)

    """ Instantiate Bayesian optimizer """
    optimizer_class = SobolOptimizer if strategy == "sobol" else BayesianOptimizer
    bo = optimizer_class(
        device=device,
        dtype=DTYPE,
        objective=objective,
        acqf=qLogNoisyExpectedHypervolumeImprovement,
        kernel=kernel,
        X=None,
        Y_obj=None,
        Y_obj_var=None,
        Y_trk=None,
        batch_size=q,
        # The initial design was measured, not drawn, so there is no sequence for the
        # sobol arm to continue - it starts its own.
        **({"sampler": sampler} if strategy == "sobol" else {}),
    )

    """ Choose between deterministic and noisy objective """
    # Asking for noise the objective cannot produce is a mistake, not a preference
    if noise and objective.gt_obj_noise_std is None:
        raise ValueError(
            f"--noise true needs an objective that declares gt_obj_noise_std, and "
            f"{type(objective).__name__} declares none. Declare one in its "
            f"__init__, or pass --noise false to measure it exactly.")
    noisy = noise
    evaluate_obj = (objective.evaluate_true_objective_with_noise if noisy
                    else objective.evaluate_true_objective)
    evaluate_trk = (objective.evaluate_tracker_with_noise if noisy
                    else objective.evaluate_tracker)
    if repeats > 1 and not noisy:
        print(f"! --repeats {repeats} on a deterministic objective: every repetition "
              f"returns the same values, so the extra rows carry no information.")

    """ Main optimization loop """
    n_initial_steps = n_initial // q
    n_steps = n_initial_steps + int(n_evals / q)
    if not verbose:
        # Keep stderr clean so stray GP-fit warnings don't fragment the tqdm bar.
        warnings.filterwarnings("ignore")
    # Counts the number of measurements, not proposals: with repeats > 1 a step costs
    # q * repeats. The initial design is exempt - those cavities were fabricated once,
    # and repeating a recorded measurement would invent data rather than take it.
    n_measurements = n_initial + n_evals * repeats
    pbar = tqdm(total=n_measurements, unit="eval", desc="Optimizing") if not verbose else None

    for i in range(n_steps):
        modelling = i >= n_initial_steps
        source = "proposed" if modelling else "initial"
        description = "Optimizing" if modelling else "Initial design"
        if pbar is not None:
            pbar.set_description(description)
        if verbose:
            phase = "propose" if modelling else "initial design"
            print(f"\n*** Step {i + 1}/{n_steps} ({phase}) | eval {(i + 1) * q}/{n_initial + n_evals} ***")

        if modelling:
            """ Optimize and get new X """
            bo.optimize(verbose=verbose)
            new_X = bo.new_X
            bo.compute_acquisition_function_value_at_X(X=new_X, verbose=verbose)
            bo.compute_posterior_mean_at_X(X=new_X, verbose=verbose)
        else:
            """ Take the next batch of the initial design """
            new_X = X_initial[i * q:(i + 1) * q]

        """ Simulate the experiment at new X, once per repetition """
        for rep in range(repeats if modelling else 1):
            step_dir = run_dir / f"step_{i:03d}_rep{rep:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(step_dir)
            if modelling:
                new_Y_obj = evaluate_obj(new_X)
                new_Y_trk = evaluate_trk(new_X)
            else:
                # The measured values, not the surrogate's opinion of them: the ground
                # truth was fitted to these rows, so replaying them through it would feed
                # the campaign a smoothed copy of its own initial design.
                new_Y_obj = Y_obj_initial[i * q:(i + 1) * q]
                new_Y_trk = Y_trk_initial[i * q:(i + 1) * q]
            if verbose:
                print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
            bo.update_XY(
                new_X=new_X,
                new_Y_obj=new_Y_obj,
                new_Y_trk=new_Y_trk,
                source=source
            )

            """ Save the running summary (run root) and this measurement's record """
            bo.to_file(filepath=run_dir / "summary.bin", verbose=verbose)
            bo.to_json(filepath=step_dir / "experiment.json", verbose=verbose)

            if pbar is not None:
                time.sleep(0.1)
                pbar.update(q)

    if pbar is not None:
        pbar.close()

    print("Optimization Finished.")


if __name__ == "__main__":
    args = parse_trial_args(description="Run a single Avagama spark-accelerator BO trial.")
    device = resolve_device(args.device)
    if args.verbose:
        print(f"Running on {device}.")
    output_dir = unique_dir(args.output_dir or default_output_dir(__file__))

    main(
        n_evals=args.n_evals,
        q=args.q_batch,
        n_initial=args.n_initial,
        seed=args.seed,
        output_dir=output_dir,
        verbose=args.verbose,
        device=device,
        strategy=args.strategy,
        repeats=args.repeats,
        noise=args.noise,
    )
