import os
import math
import warnings
import torch
from pathlib import Path
from tqdm import tqdm
from botorch.acquisition import *
from gpytorch.kernels import *
from gpytorch.constraints import Interval, LessThan
from pybo.optimizer.sobol import SobolOptimizer
from pybo.optimizer.bayesian import BayesianOptimizer
from pybo.utils.cli import parse_trial_args, default_output_dir, resolve_device, unique_dir
from tutorials.single_objective.wave_packet.objective import WavePacket
from pybo.plotters.acqf import Acqf1DPlotter
from pybo.plotters.experiment import Experiment1DPlotter
from pybo.samplers.sobol import *
from pybo.plotters.evolution import *
from pybo.plotters.metrics import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(output_dir: Path, n_evals=64, q: int = 1, n_initial: int = None, seed: int = 2063, plot: bool = True,
         verbose: bool = True, device: torch.device = DEVICE, strategy: str = "bo"):
    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting optimization ({n_evals} evals, q={q}, seed={seed})")

    """ Seed the global torch RNG to ensure reproducibility. """
    torch.manual_seed(seed)

    """ Instantiate true objective """
    objective = WavePacket(device=device, dtype=DTYPE, )

    """ Instantiate kernel """
    rbf = RBFKernel(ard_num_dims=objective.num_obj, lengthscale_constraint=LessThan(1 / 8))
    periodic = PeriodicKernel(period_length_constraint=Interval(1 / 4 * 0.8, 1 / 4 * 1.2))
    kernel = ScaleKernel(base_kernel=rbf * periodic)

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
        acqf=qLogNoisyExpectedImprovement,
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

    """ Main optimization loop """
    n_initial_steps = n_initial // q
    n_steps = n_initial_steps + int(n_evals / q)
    if not verbose:
        # Keep stderr clean so stray GP-fit warnings don't fragment the tqdm bar.
        warnings.filterwarnings("ignore")
    pbar = tqdm(total=n_initial + n_evals, unit="eval", desc="Optimizing") if not verbose else None

    for i in range(n_steps):
        # One folder per evaluation step
        step_dir = run_dir / f"step_{i:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(step_dir)

        modelling = i >= n_initial_steps
        if pbar is not None:
            pbar.set_description("Optimizing" if modelling else "Initial design")
        if verbose:
            phase = "propose" if modelling else "initial design"
            print(f"\n*** Step {i + 1}/{n_steps} ({phase}) | eval {(i + 1) * q}/{n_initial + n_evals} ***")

        if modelling:
            """ Optimize and get new X """
            bo.optimize(verbose=verbose)

            """ Plot """
            if plot:
                Experiment1DPlotter(bo=bo).plot().save_figure().close_figure()
                Acqf1DPlotter(bo=bo, z=("obj", 0)).plot().save_figure().close_figure()
                ElapsedTimePlotter(bo=bo).plot().save_figure().close_figure()
                BestValuePlotter(bo=bo).plot().save_figure().close_figure()
                EvolutionPlotter(bo=bo, y=("obj", 0)).plot().save_figure().close_figure()
                EvolutionPlotter(bo=bo, y=("par", 0)).plot().save_figure().close_figure()

            """ Evaluate posterior and acquisition function at new X """
            new_X = bo.new_X
            bo.compute_acquisition_function_value_at_X(X=new_X, verbose=verbose)
            bo.compute_posterior_mean_at_X(X=new_X, verbose=verbose)

        else:
            """ Take the next batch of the initial design """
            new_X = X_initial[i * q:(i + 1) * q]

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        if verbose:
            print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj, source="proposed" if modelling else "initial")

        """ Save the running summary (run root) and this step's experiment record """
        bo.to_file(filepath=run_dir / "summary.bin", verbose=verbose)
        bo.to_json(filepath=step_dir / "experiment.json", verbose=verbose)

        if pbar is not None:
            pbar.update(q)

    if pbar is not None:
        pbar.close()

    print("Optimization Finished.")


if __name__ == "__main__":
    args = parse_trial_args(description="Run a single WavePacket BO trial.")
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
