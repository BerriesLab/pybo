import os
from datetime import datetime
import torch
from pathlib import Path
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from pybo.optimizer.optimizer import BayesianOptimizer
from pybo.plotters.experiment import ParetoFront2DPlotter
from pybo.samplers.samplers import SobolSampler
from pybo.plotters.metrics import plot_and_save_metrics
from pybo.plotters.evolution import plot_and_save_evolutions
from tutorials.multi_objective.c2dtlz2.objective import C2DTLZ2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_evals=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Instantiate true objective """
    objective = C2DTLZ2(device=DEVICE, dtype=DTYPE)

    """ Instantiate kernel """
    kernel = ScaleKernel(
        base_kernel=RBFKernel(
            ard_num_dims=objective.dim,
            lengthscale_constraint=Interval(1e-4, 1),
        ),
        outputscale_constraint=Interval(1e-4, 1e2),
    )

    """ Generate initial dataset """
    sampler = SobolSampler(device=DEVICE, dtype=DTYPE, objective=objective, seed=2063)
    X = sampler.draw_samples(n=5 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)
    Y_con = objective.evaluate_true_constraint(X)

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
        Y_con=Y_con,
        Y_con_var=None,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_evals / q)):
        if i > 0 and bo.is_converged(patience=32):
            break

        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_evals / q)} ***")

        """ Optimize and get new X """
        bo.optimize()

        """ Plot """
        ParetoFront2DPlotter(
            bo=bo,
            x=("obj", 0),
            y=("obj", 1),
            # z=("par", "P2"),
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
        new_Y_con = objective.evaluate_true_constraint(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj, new_Y_con=new_Y_con)

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_path = Path.cwd() / "data" / date_time
    main_path.mkdir(parents=True, exist_ok=True)

    batch_sizes = [1]
    for batch_size in batch_sizes:
        main(n_evals=128, q=batch_size, output_dir=main_path)
