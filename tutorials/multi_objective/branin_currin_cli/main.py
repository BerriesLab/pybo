import os
import torch
from pathlib import Path
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel
from pybo.optimizer.optimizer import BayesianOptimizer
from pybo.plotters.experiment import ParetoFront2DPlotter
from pybo.samplers.samplers import SobolSampler
from pybo.plotters.metrics import plot_and_save_metrics
from pybo.plotters.evolution import plot_and_save_evolutions
from pybo.utils.cli import build_trial_args_parser, default_output_dir
from tutorials.multi_objective.branin_currin.objective import BraninCurrin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_evals=64, q: int = 1, n_initial: int = None, seed: int = 2063,
         output_dir: Path = None, plot: bool = True):
    run_dir = output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    """ Instantiate true objective """
    objective = BraninCurrin(device=DEVICE, dtype=DTYPE)

    """ Instantiate kernel """
    kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=objective.num_par))

    """ Generate initial dataset """
    n_initial = n_initial or 5 * (objective.dim + 1)
    sampler = SobolSampler(device=DEVICE, dtype=DTYPE, objective=objective, seed=seed)
    X = sampler.draw_samples(n=n_initial)
    Y_obj = objective.evaluate_true_objective(X)

    """ Instantiate Bayesian optimizer """
    bo = BayesianOptimizer(
        device=DEVICE,
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
    )

    """ Main optimization loop """
    for i in range(int(n_evals / q)):
        if i > 0 and bo.is_converged(patience=32):
            break

        """ One folder per evaluation step; figures and per-step files go here """
        step_dir = run_dir / f"step_{i:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(step_dir)

        print()
        print(f"*** Iteration {i + 1}/{int(n_evals / q)} ***")

        """ Optimize and get new X """
        bo.optimize()

        """ Plot """
        if plot:
            ParetoFront2DPlotter(
                bo=bo,
                x=("obj", "Branin"),
                y=("obj", "Currin"),
                z=("par", 1),
                seed=254,
            ).plot().save_figure().close_figure()
            plot_and_save_metrics(bo=bo)
            plot_and_save_evolutions(bo=bo)

        """ Evaluate posterior and acquisition function at new X """
        new_X = bo.new_X
        bo.compute_acquisition_function_value_at_X(new_X)
        bo.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

        """ Save the running summary (run root) and this step's experiment record """
        bo.to_json(filepath=run_dir / "summary.json", latest=False, verbose=True)
        bo.to_json(filepath=step_dir / "experiment.json", latest=True, verbose=True)
        bo.to_csv(filepath=step_dir / "experiment.csv", latest=True, verbose=True)

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")

    args = build_trial_args_parser(description="Run a single Branin-Currin BO trial.").parse_args()
    output_dir = args.output_dir or default_output_dir(__file__)
    output_dir.mkdir(parents=True, exist_ok=True)

    main(
        n_evals=args.n_evals,
        q=args.q_batch,
        n_initial=args.n_initial,
        seed=args.seed,
        output_dir=output_dir,
        plot=args.plot,
    )
