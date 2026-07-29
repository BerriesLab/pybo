import os
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
from pybo.plotters.experiment import *
from pybo.plotters.acqf import *
from pybo.plotters.metrics import *
from pybo.plotters.evolution import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(output_dir: Path, n_evals=64, q: int = 1, n_initial: int = None, seed: int = 2063, plot: bool = True,
         verbose: bool = True, device: torch.device = DEVICE, strategy: str = "bo"):
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

    """ Generate initial dataset """
    n_initial = n_initial or 5 * (objective.dim + 1)
    sampler = SobolSampler(device=device, dtype=DTYPE, objective=objective, seed=seed)
    X = sampler.draw_samples(n=n_initial)
    Y_obj = objective.evaluate_true_objective(X)

    """ Instantiate Bayesian optimizer """
    optimizer_class = SobolOptimizer if strategy == "sobol" else BayesianOptimizer
    bo = optimizer_class(
        device=device,
        dtype=DTYPE,
        objective=objective,
        acqf=qLogNoisyExpectedHypervolumeImprovement,
        kernel=kernel,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
        **({"sampler": sampler} if strategy == "sobol" else {}),
    )

    """ Main optimization loop """
    n_steps = int(n_evals / q)
    if not verbose:
        # Keep stderr clean so stray GP-fit warnings don't fragment the tqdm bar.
        warnings.filterwarnings("ignore")
    pbar = tqdm(total=n_evals, unit="eval", desc="Optimizing") if not verbose else None
    for i in range(n_steps):
        """ One folder per evaluation step; figures and per-step files go here """
        step_dir = run_dir / f"step_{i:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(step_dir)

        if verbose:
            print(f"\n*** Step {i + 1}/{n_steps} | eval {(i + 1) * q}/{n_evals} ***")

        """ Optimize and get new X """
        bo.optimize(verbose=verbose)

        """ Plot """
        if plot:
            ParetoFront2DPlotter(
                bo=bo,
                x=("obj", 0),
                y=("obj", 1),
                z=("par", 0),
                seed=254,
            ).plot().save_figure().close_figure()
            plot_and_save_metrics(bo=bo)
            plot_and_save_evolutions(bo=bo)

        """ Evaluate posterior and acquisition function at new X """
        new_X = bo.new_X
        bo.compute_acquisition_function_value_at_X(X=new_X, verbose=verbose)
        bo.compute_posterior_mean_at_X(X=new_X, verbose=verbose)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        if verbose:
            print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

        """ Save the running summary (run root) and this step's experiment record """
        bo.to_file(filepath=run_dir / "summary.bin", verbose=verbose)
        bo.to_json(filepath=run_dir / "summary.json", latest=False, verbose=verbose)
        bo.to_json(filepath=step_dir / "experiment.json", latest=True, verbose=verbose)
        bo.to_csv(filepath=step_dir / "experiment.csv", latest=True, verbose=verbose)

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
        plot=args.plot,
        verbose=args.verbose,
        device=device,
        strategy=args.strategy,
    )
