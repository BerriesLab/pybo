import os
import torch
from pathlib import Path
from mobo.mobo import Mobo
from objectives.osyczka_kundu import OsyczkaKundu
from samplers.samplers import Sampler
from utils.helpers import create_experiment_directory
from utils.types import AcquisitionFunctionType, SamplerType
from plotters.multi_objective import MultiObjectivePlotter
from plotters.evolution import ElapsedTimePlotter, HypervolumePlotter, HypervolumeImprovementPlotter, ParameterPlotter, \
    ObjectivePlotter, TrackerPlotter, ConstraintPlotter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, ):
    data_path = main_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    experiment_name = f"osyczka_kundu"
    directory = create_experiment_directory(data_path, experiment_name)
    os.chdir(directory)

    """ Define the objective """
    objective = OsyczkaKundu(
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
        normalize=False,
        linear_inequality_constraints=objective.linear_inequality_input_constraints,
        nonlinear_inequality_constraints=objective.nonlinear_inequality_input_constraints,
    )

    """ Generate initial dataset """
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Generate samples for ground truth evaluation - random sampler or grid """
    # When constraints apply to the input X, build the ground truth by using
    # a random generator subject to constraints
    X_gt = sampler.draw_samples(n=10_000)

    """ Instantiate a Mobo object """
    mobo = Mobo(
        experiment_name=experiment_name,
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        X=X,
        Y_obj=Y_obj,
        n_acqf_opt_restarts=50,
        raw_samples=1024
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
        mobo.to_file(output_path=Path.cwd() / f"mobo.dat")

        """ Plots """
        multi_objective_plotter = MultiObjectivePlotter(
            title="Pareto Front",
            mobo=mobo,
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
    batch_sizes = [1]  # [1, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
