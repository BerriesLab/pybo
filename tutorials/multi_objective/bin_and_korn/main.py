import os
import torch
from pathlib import Path

from botorch.acquisition import qLogExpectedImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel

from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.experiment import ParetoFront2DPlotter
from samplers.samplers import SobolSampler
from utils.helpers import create_experiment_directory
from plotters.evolution import HypervolumePlotter, HypervolumeImprovementPlotter, ElapsedTimePlotter, ObjectivePlotter, \
    ConstraintPlotter, TrackerPlotter, ParameterPlotter
from tutorials.multi_objective.bin_and_korn.objective import BinhAndKornMCMultiOutputObjective

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, ):
    data_path = main_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    experiment_name = f"binh_and_korn"
    directory = create_experiment_directory(data_path, experiment_name)
    os.chdir(directory)

    """ Instantiate true objective """
    objective = BinhAndKornMCMultiOutputObjective(
        device=DEVICE,
        dtype=DTYPE,
    )

    """ Instantiate kernel """
    kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=objective.num_objectives))

    """ Generate initial dataset """
    sampler = SobolSampler(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        seed=2063
    )
    X = sampler.draw_samples(n=5 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Instantiate Bayesian optimizer """
    bo = BayesianOptimizer(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acqf=qLogExpectedImprovement,
        kernel=kernel,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        if i > 0 and bo.is_converged(patience=32):
            break

        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        """ Optimize and get new X """
        bo.optimize()

        """ Plot """
        ParetoFront2DPlotter.plot_ground_truth()
        ElapsedTimePlotter(bo=bo).plot().save_figure().close_figure()
        HypervolumePlotter(bo=bo).plot().save_figure().close_figure()
        HypervolumeImprovementPlotter(bo=bo).plot().save_figure().close_figure()
        for idx in range(bo.objective.dim):
            ParameterPlotter(bo=bo, idx=idx).plot().save_figure().close_figure()
        for idx in range(bo.objective.num_objectives):
            ObjectivePlotter(bo=bo, idx=idx).plot().save_figure().close_figure()
        for idx in range(bo.objective.num_constraints):
            ConstraintPlotter(bo=bo, idx=idx).plot().save_figure().close_figure()
        for idx in range(bo.objective.num_trackers):
            TrackerPlotter(bo=bo, idx=idx).plot().save_figure().close_figure()

        # Experiment2DPlotter(bo=bo).plot().save_figure().close_figure()
        # Acqf2DPlotter(bo=bo).plot().save_figure().close_figure()
        # ElapsedTimePlotter(bo=bo).plot().save_figure().close_figure()
        # BestValuePlotter(bo=bo).plot().save_figure().close_figure()

        """ Evaluate posterior and acquisition function at new X """
        new_X = bo.new_X
        bo.compute_acquisition_function_value_at_X(new_X)
        bo.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")
    main_path = Path.cwd().parent
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
