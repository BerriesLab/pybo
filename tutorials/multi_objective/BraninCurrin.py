import os
import torch
from pathlib import Path
from bayesian_optimizer.bayesian_optimizer import BayesianOptimizer
from samplers.samplers import Sampler
from utils.helpers import create_experiment_directory
from utils.types import AcquisitionFunctionType, SamplerType
from plotters.multi_objective import MultiObjectivePlotter
from plotters.evolution import ElapsedTimePlotter, HypervolumePlotter, HypervolumeImprovementPlotter, ParameterPlotter, \
    ObjectivePlotter, TrackerPlotter, ConstraintPlotter
from plotters.utils import make_grid
from objectives.multi_objective.branin_currin import BraninCurrinMCMultiOutputObjective

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples: int = 64, q: int = 1, ):
    data_path = main_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    experiment_name = f"branincurrin"
    directory = create_experiment_directory(data_path, experiment_name)
    os.chdir(directory)

    """ Define the objective """
    objective = BraninCurrinMCMultiOutputObjective(
        device=DEVICE,
        dtype=DTYPE,
    )

    """ Instantiate a random generator """
    sampler = Sampler(
        device=DEVICE,
        dtype=DTYPE,
        sampler_type=SamplerType.Sobol,
        bounds=objective.bounds,
        n_dimensions=objective.dim,
        normalize=False
    )

    """ Generate initial dataset """
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Generate samples for ground truth evaluation - random sampler or grid """
    # This is done before the optimization loop to show the same ground truth
    # in each iteration step's figure.
    X_gt = make_grid(
        device=DEVICE,
        dtype=DTYPE,
        size=100,
        bounds=objective.bounds,
    )

    """ Instantiate a Mobo object """
    mobo = BayesianOptimizer(
        experiment_name=experiment_name,
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_type=AcquisitionFunctionType.qLogEHVI,
        X=X,
        Y_obj=Y_obj,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        print("\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        """ Optimize and get new X """
        mobo.optimize()
        new_X = mobo.new_X
        print(f"New X: {new_X.detach().cpu().numpy()}")

        """ Evaluate posterior and acquisition function at new X """
        mobo.compute_acquisition_function_value_at_X(new_X)
        mobo.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Yobj = objective.evaluate_true_objective(new_X)
        print(f"New Yobj: {new_Yobj.detach().cpu().numpy()}")
        mobo.update_XY(new_X=new_X, new_Y_obj=new_Yobj)

        """ Compute pareto front and hypervolume """
        mobo.compute_pareto_front()
        mobo.compute_hypervolume()

        """ Save"""
        mobo.to_file(output_path=Path.cwd() / f"bayesian_optimizer.dat")

        """ Plots """
        multi_objective_plotter = MultiObjectivePlotter(
            title="Pareto Front",
            bayesian_optimizer=mobo,
            X_gt=X_gt,
            idx_x=0,
            idx_y=1,
            idx_color=None,
            use_tracker=True,
            pareto_idxs=[0, 1],
        )
        multi_objective_plotter.plot_ground_truth()
        multi_objective_plotter.plot_objectives()
        multi_objective_plotter.save_figure().close_figure()
        ElapsedTimePlotter(bayesian_optimizer=mobo).plot().save_figure().close_figure()
        HypervolumePlotter(bayesian_optimizer=mobo).plot().save_figure().close_figure()
        HypervolumeImprovementPlotter(mobo=mobo).plot().save_figure().close_figure()
        for idx in range(mobo.objective.dim):
            ParameterPlotter(bayesian_optimizer=mobo, idx=idx).plot().save_figure().close_figure()
        for idx in range(mobo.objective.num_objectives):
            ObjectivePlotter(bayesian_optimizer=mobo, idx=idx).plot().save_figure().close_figure()
        for idx in range(mobo.objective.num_constraints):
            ConstraintPlotter(bayesian_optimizer=mobo, idx=idx).plot().save_figure().close_figure()
        for idx in range(mobo.objective.num_trackers):
            TrackerPlotter(bayesian_optimizer=mobo, idx=idx).plot().save_figure().close_figure()

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")
    main_path = Path.cwd().parent
    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
