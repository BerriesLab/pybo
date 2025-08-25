import os
from objectives.formaco import FormACOMCMultiOutputObjective
from samplers.samplers import Sampler
from utils.helpers import create_experiment_directory
from utils.types import AcquisitionFunctionType, SamplerType
from plotters.multi_objective import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, ):
    data_path = main_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    experiment_name = f"formaco"
    directory = create_experiment_directory(data_path, experiment_name)
    os.chdir(directory)

    """ Define the objective """
    objective = FormACOMCMultiOutputObjective(
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
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        print(f"New Y_track: {new_Y_track.detach().cpu().numpy()}")
        mobo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj, new_Y_track=new_Y_track)

        """ Compute pareto front and hypervolume """
        mobo.compute_pareto_front()
        mobo.compute_hypervolume()

        """ Save"""
        mobo.to_file(output_path=Path.cwd() / f"mobo.dat")

        """ Plots """
        plotter = MultiObjectivePlotter(
            title="Pareto Front",
            mobo=mobo,
            X_gt=gnd_truth_X,
            idx_x=0,
            idx_y=1,
            idx_color=None,
            use_tracker=True,
            pareto_idxs=[0, 1],
        )
        plotter.plot_ground_truth()
        plotter.plot_objectives()
        plotter.save_figure()

        plot_log_hypervolume_improvement(mobo=mobo)
        plot_elapsed_time(mobo=mobo)
        plot_parameters_evolution(mobo=mobo)
        plot_objectives_evolution(mobo=mobo)
        plot_constraints_evolution(mobo=mobo)
        plot_trackers_evolution(mobo=mobo)

    print("Optimization Finished.")


if __name__ == "__main__":
    main_path = Path.cwd().parent
    batch_sizes = [1]  # [1,, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
