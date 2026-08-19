import os
import math
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel
from pybo.optimizer.sobol import SobolOptimizer
from pybo.optimizer.random import RandomOptimizer
from pybo.optimizer.bayesian import BayesianOptimizer
from pybo.samplers.sobol import SobolSampler
from pybo.utils.cli import parse_trial_args, default_output_dir, resolve_device, unique_dir
from pybo.utils.init_dataset import load_initial_dataset, slice_initial_batch
from pybo.utils.trial_record import TrialRecord
from tutorials.multi_objective.vformac.objective import VFormAC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(output_dir: Path, n_evals=64, q: int = 1, n_initial: int = None, seed: int = 2063,
         verbose: bool = True, device: torch.device = DEVICE, strategy: str = "bo",
         repeats: int = 1, noise: bool = False, init_data: Path = None,
         shuffle_init: bool = False):
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

    """ Draw or load the initial parameter set """
    sampler = SobolSampler(device=device, dtype=DTYPE, objective=objective)
    initial = None
    if init_data is not None:
        # A recorded warm start, not a fresh draw: --n-initial then means "keep only
        # the first this many", so the sampler below is unused for X but is still built,
        # since the sobol arm reuses it to continue that sequence for its proposals.
        initial = load_initial_dataset(init_data, objective, n_initial=n_initial,
                                       shuffle_seed=seed if shuffle_init else None)
        X_initial = initial["X"].to(device=device, dtype=DTYPE)
        n_initial = X_initial.shape[0]
        print(f"Loaded {n_initial} initial point{'' if n_initial == 1 else 's'} from {init_data}")
    else:
        n_initial = n_initial or 5 * (objective.dim + 1)
        n_initial = math.ceil(n_initial / q) * q
        X_initial = sampler.draw_samples(n=n_initial)
    # Nothing here reads this back - it is the hand-off to whoever (or whatever machine
    # control software) actually sets these points on the rig: a flat table in the
    # parameters' real units, rather than the nested, machine-oriented experiment.json.
    X_np = X_initial.detach().cpu().numpy()
    np.savetxt(run_dir / "output.csv", X_np, delimiter=",",
               header=", ".join(cfg.label for cfg in objective.par_cfg))

    # What this trial asked for (or defaulted to), for every step's experiment.json -
    # the optimizer itself has no way to know any of it. "synthetic" by default: the
    # loop below evaluates the objective in Python either way; a loaded initial row is
    # the one exception, overridden per step below (see loaded_initial).
    trial = TrialRecord(n_initial=n_initial, seed=seed, provenance="synthetic",
                        n_evals=n_evals, q=q, noise=noise, repeats=repeats,
                        device=str(device))

    """ Instantiate Bayesian optimizer """
    optimizer_class = {"sobol": SobolOptimizer,
                       "random": RandomOptimizer}.get(strategy, BayesianOptimizer)
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
        # Reuse the initial design's sampler, so the sobol arm continues that
        # sequence instead of starting a second one.
        **({"sampler": sampler} if strategy == "sobol" else {}),
    )

    """ Choose between deterministic and noisy objective """
    noisy = noise
    if repeats > 1 and not noisy:
        print(f"! --repeats {repeats} on a deterministic objective: every repetition "
              f"returns the same values, so the extra rows carry no information.")

    """ Main optimization loop """
    # ceil, not //: a loaded initial dataset need not divide evenly by q, and the last
    # batch is then just smaller rather than dropping a recorded point.
    n_initial_steps = math.ceil(n_initial / q)
    n_steps = n_initial_steps + int(n_evals / q)
    if not verbose:
        # Keep stderr clean so stray GP-fit warnings don't fragment the tqdm bar.
        warnings.filterwarnings("ignore")
    # Counts the number of measurements, not proposals: with repeats > 1 a step costs q * repeats.
    # A loaded initial design is already-measured data - it is read once, not repeated.
    n_measurements = n_initial + n_evals * repeats if init_data is not None \
        else (n_initial + n_evals) * repeats
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

        # A loaded initial row is already-measured data - read once, not repeated the
        # way a fresh simulated measurement is.
        loaded_initial = not modelling and init_data is not None
        reps = 1 if loaded_initial else repeats

        """ Simulate the experiment at new X, once per repetition - or read it back, for
        a loaded initial row """
        for rep in range(reps):
            step_dir = run_dir / f"step_{i:03d}_rep{rep:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(step_dir)
            if loaded_initial:
                sl = slice(i * q, (i + 1) * q)
                updates = slice_initial_batch(initial, sl, device=device, dtype=DTYPE)
            else:
                new_Y_obj = objective.evaluate_true_objective(new_X, noisy=noisy)
                new_Y_trk = objective.evaluate_tracker(new_X, noisy=noisy)
                updates = {"new_Y_obj": new_Y_obj, "new_Y_trk": new_Y_trk}
            if verbose:
                print(f"New Y_obj: {updates['new_Y_obj'].detach().cpu().numpy()}")
            bo.update_XY(new_X=new_X, source=source, **updates)

            """ Save the running summary (run root) and this measurement's record """
            bo.to_file(filepath=run_dir / "summary.bin", verbose=verbose)
            bo.to_json(filepath=step_dir / "experiment.json", verbose=verbose,
                      extra=trial.step_fields(
                          step_index=i, repetition=rep,
                          provenance="experimental" if loaded_initial else None))

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
        init_data=args.init_data,
        shuffle_init=args.shuffle_init,
    )
