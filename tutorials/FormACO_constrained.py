import os
import torch
from pathlib import Path
from mobo.mobo import Mobo
from objectives.formaco import FormACOMCMultiOutputConstrainedObjective
from plotters.evolution import *
from samplers.samplers import Sampler
from utils.helpers import create_experiment_directory
from utils.types import AcquisitionFunctionType, SamplerType
from plotters.multi_objective import plot_log_hypervolume_improvement, plot_elapsed_time, \
    plot_parameters_evolution, plot_objectives_evolution, \
    plot_constraints_evolution, MultiObjectivePlotter
from plotters.utils import make_grid

DEVICE = torch.device("cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, ):
    data_path = main_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    experiment_name = f"formaco_constrained"
    directory = create_experiment_directory(data_path, experiment_name)
    os.chdir(directory)

    """ Define the objective """
    objective = FormACOMCMultiOutputConstrainedObjective(
        device=DEVICE,
        dtype=DTYPE,
    )

    """ Generate initial dataset """
    sampler = Sampler(
        sampler_type=SamplerType.Sobol,
        bounds=objective.bounds,
        n_dimensions=objective.dim,
        normalize=False,
    )
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X=X)
    Y_track = objective.evaluate_trackers(X=X)
    Y_con = objective.evaluate_true_constraint(X=X)

    """ Generate samples for ground truth evaluation - random sampler or grid """
    # This is done before the optimization loop to show the same ground truth
    # in each iteration step's figure.
    gnd_truth_X = make_grid(
        size=20,
        bounds=objective.bounds,
        device=DEVICE,
        dtype=DTYPE
    )

    """ Instantiate a Mobo object """
    mobo = Mobo(
        experiment_name=experiment_name,
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        sampler_type=SamplerType.Sobol,
        X=X,
        Y_obj=Y_obj,
        Y_con=Y_con,
        Y_track=Y_track,
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
        mobo.compute_acquisition_function_value_at_X(X=new_X)
        mobo.compute_posterior_mean_at_X(X=new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(X=new_X)
        new_Y_track = objective.evaluate_trackers(X=new_X)
        new_Y_con = objective.evaluate_true_constraint(X=new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        print(f"New Y_con: {new_Y_con.detach().cpu().numpy()}")
        print(f"New Y_track: {new_Y_track.detach().cpu().numpy()}")
        mobo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj, new_Y_con=new_Y_con, new_Y_track=new_Y_track)

        """ Compute pareto front and hypervolume """
        mobo.compute_pareto_front()
        mobo.compute_hypervolume()

        """ Save"""
        mobo.to_file(output_path=Path.cwd() / f"mobo.dat")

        """ Plots """
        multi_objective_plotter = MultiObjectivePlotter(
            mobo=mobo,
            X_gt=gnd_truth_X,
            idx_x=0,
            idx_y=1,
            pareto_idxs=[0, 1]
        )
        multi_objective_plotter.plot_ground_truth()
        multi_objective_plotter.plot_objectives()
        multi_objective_plotter.save_figure()
        ElapsedTimePlotter(mobo=mobo).plot().save_figure().close_figure()
        HypervolumePlotter(mobo=mobo).plot().save_figure().close_figure()
        HypervolumeImprovementPlotter(mobo=mobo).plot().save_figure().close_figure()
        for idx in range(mobo.objective.dim):
            ParameterPlotter(mobo=mobo, idx=idx).plot().save_figure().close_figure()
        for idx in range(mobo.objective.num_objectives):
            ObjectivePlotter(mobo=mobo, idx=idx).plot().save_figure().close_figure()
        for idx in range(mobo.objective.num_constraints):
            ConstraintPlotter(mobo=mobo, idx=idx).plot().save_figure().close_figure()
        for idx in range(mobo.objective.num_trackers):
            TrackerPlotter(mobo=mobo, idx=idx).plot().save_figure().close_figure()

    print("Optimization Finished.")


if __name__ == "__main__":
    main_path = Path.cwd().parent
    batch_sizes = [1]  # [1,, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
