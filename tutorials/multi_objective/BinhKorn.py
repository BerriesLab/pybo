import os
import torch
from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from samplers.samplers import SamplerBase
from utils.helpers import create_experiment_directory
from utils.bo_types import AcquisitionFunctionType, SamplerType
from plotters.multi_objective import MultiObjectivePlotter
from plotters.evolution import HypervolumePlotter, HypervolumeImprovementPlotter, ElapsedTimePlotter, ObjectivePlotter, \
    ConstraintPlotter, TrackerPlotter, ParameterPlotter
from objectives.multi_objective.binh_korn import BinhAndKornMCMultiOutputObjective

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, ):
    data_path = main_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    experiment_name = f"binh_and_korn"
    directory = create_experiment_directory(data_path, experiment_name)
    os.chdir(directory)

    """ Define the true_objective """
    objective = BinhAndKornMCMultiOutputObjective(
        device=DEVICE,
        dtype=DTYPE,
    )

    """ Instantiate a random generator """
    sampler = SamplerBase(
        device=DEVICE,
        dtype=DTYPE,
        sampler_type=SamplerType.Sobol,
        bounds=objective.bounds,
        n_dimensions=objective.num_objectives,
        normalize=False,
        nonlinear_inequality_constraints=objective.nonlinear_inequality_input_constraints,
    )

    """ Generate initial dataset """
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Generate samples for ground truth evaluation - random sampler or grid """
    # When constraints apply to the input X, build the ground truth by using
    # a random generator subject to constraints
    X_gt = sampler.draw_samples(n=1000)

    """ Main optimization loop """
    mobo = BayesianOptimizer(
        experiment_name=experiment_name,
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_builder=AcquisitionFunctionType.qNEHVI,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,

    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        """ Optimize and get new X """
        mobo.optimize()
        new_X = mobo.new_X
        print(f"New X: {new_X.detach().cpu().numpy()}")

        """ Evaluate posterior and acquisition function at new X """
        mobo.compute_acquisition_function_value_at_X(new_X)
        mobo.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        mobo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

        """ Compute pareto front and hypervolume """
        mobo._compute_feasibility_mask()
        mobo.compute_pareto_front()
        mobo.compute_hypervolume()

        """ Save"""
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
        multi_objective_plotter.save_figure()
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
