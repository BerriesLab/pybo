import os
import math
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
from tutorials.multi_objective.linear_inequality.objective import LinearInequalityTest

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(output_dir: Path, n_evals=64, q: int = 1, n_initial: int = None, seed: int = 2063,
         verbose: bool = True, device: torch.device = DEVICE, strategy: str = "bo",
         repeats: int = 1, noise: bool = False):
    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting optimization ({n_evals} evals, q={q}, seed={seed})")

    """ Seed the global torch RNG to ensure reproducibility. """
    torch.manual_seed(seed)

    """ Define the objective """
    objective = LinearInequalityTest(device=device, dtype=DTYPE)

    """ Instantiate kernel """
    kernel = ScaleKernel(
        base_kernel=RBFKernel(
            ard_num_dims=objective.dim,
            lengthscale_constraint=Interval(1e-3, 1.0),
        ),
        outputscale_constraint=Interval(1e-3, 1e2),
    )

    """ Draw the initial parameter set """
    n_initial = n_initial or 5 * (objective.dim + 1)
    n_initial = math.ceil(n_initial / q) * q
    sampler = SobolSampler(device=device, dtype=DTYPE, objective=objective)
    X_initial = sampler.draw_samples(n=n_initial)

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
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
        # Reuse the initial design's sampler, so the sobol arm continues that
        # sequence instead of starting a second one.
        **({"sampler": sampler} if strategy == "sobol" else {}),
    )

    """ Choose between deterministic and noisy objective """
    # Asking for noise the objective cannot produce is a mistake, not a preference
    if noise and objective.gt_noise_std is None:
        raise ValueError(
            f"--noise true needs an objective that declares gt_noise_std, and "
            f"{type(objective).__name__} declares none. Declare one in its "
            f"__init__, or pass --noise false to measure it exactly.")
    noisy = noise
    evaluate = (objective.evaluate_true_objective_with_noise if noisy
                else objective.evaluate_true_objective)
    if repeats > 1 and not noisy:
        print(f"! --repeats {repeats} on a deterministic objective: every repetition "
              f"returns the same values, so the extra rows carry no information.")

    """ Main optimization loop """
    n_initial_steps = n_initial // q
    n_steps = n_initial_steps + int(n_evals / q)
    if not verbose:
        # Keep stderr clean so stray GP-fit warnings don't fragment the tqdm bar.
        warnings.filterwarnings("ignore")
    # Counts the number of measurements, not proposals: with repeats > 1 a step costs q * repeats
    n_measurements = (n_initial + n_evals) * repeats
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
        for rep in range(repeats):
            step_dir = run_dir / f"step_{i:03d}_rep{rep:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            os.chdir(step_dir)
            new_Y_obj = evaluate(new_X)
            if verbose:
                print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
            bo.update_XY(
                new_X=new_X,
                new_Y_obj=new_Y_obj,
                source=source
            )

            """ Save the running summary (run root) and this measurement's record """
            bo.to_file(filepath=run_dir / "summary.bin", verbose=verbose)
            bo.to_json(filepath=step_dir / "experiment.json", verbose=verbose)

            if pbar is not None:
                pbar.update(q)

    if pbar is not None:
        pbar.close()

    print("Optimization Finished.")


if __name__ == "__main__":
    args = parse_trial_args(description="Run a single linear-inequality-constrained BO trial.")
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
