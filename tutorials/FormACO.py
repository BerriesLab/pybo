import os
import torch
from pathlib import Path
from mobo.mobo import Mobo
from objectives.formaco import FormACOMCMultiOutputObjective
from samplers.samplers import Sampler
from objectives.c2dtlz2 import C2DTLZ2MCMultiOutputObjective
from utils.io import create_experiment_directory
from utils.make_video import create_video_from_images
from utils.types import AcquisitionFunctionType, SamplerType
from utils.plotters import plot_multi_objective_from_RN_to_R3_with_color_coded_R3, plot_log_hypervolume_improvement, plot_elapsed_time, \
    make_grid


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
    Y_obj = objective.evaluate_true(X)

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
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
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
        new_Y_obj = objective.evaluate_true(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        mobo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

        """ Compute pareto front and hypervolume """
        mobo.compute_pareto_front()
        mobo.compute_hypervolume()

        """ Save"""
        mobo.to_file(output_path=Path.cwd() / f"mobo.dat")

        """ Plots """
        plot_multi_objective_from_RN_to_R3_with_color_coded_R3(
            mobo=mobo,
            title="FormACO Test Problem",
            f1_label="Machining Down Time (min)",
            f2_label=r"Electrode Wear $(\mu m)$",
            f3_label="Orbiting Time (min)",
            f3_lims=(20, 80),
            f1_idx=0,
            f2_idx=1,
            f3_idx=2,
            show_ref_point=True,
            show_ground_truth=True,
            show_observations=True,
            display_figure=False,
            X=gnd_truth_X,
            output_path=Path.cwd() / f"pareto_front.png"
        )
        plot_log_hypervolume_improvement(
            mobo=mobo,
            output_path=Path.cwd() / f"hvi.png"
        )
        plot_elapsed_time(
            mobo=mobo,
            output_path=Path.cwd() / f"elapsed_time.png"
        )

    print("Optimization Finished.")


if __name__ == "__main__":
    main_path = Path.cwd().parent
    batch_sizes = [1]  # [1,, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
