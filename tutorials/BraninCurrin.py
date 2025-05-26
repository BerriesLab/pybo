import os
from mobo.mobo import Mobo
from mobo.samplers import Sampler
from utils.io import *
from utils.make_video import create_video_from_images
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_improvement, plot_elapsed_time, \
    plot_allocated_memory
from botorch.test_functions.multi_objective import BraninCurrin


def main(n_samples=64, q: int = 1, ):
    data_dir = main_dir / "data"
    experiment_name = f"test_branincurrin_64iter_{q}q_256mc_128rs_qnehvi"
    directory = create_experiment_directory(data_dir, experiment_name)
    os.chdir(directory)

    """ Define the true_objective """
    true_objective = BraninCurrin(negate=True)

    """ Instantiate a random generator """
    sampler = Sampler(
        sampler_type=SamplerType.Sobol,
        bounds=true_objective.bounds,
        n_dimensions=true_objective.dim,
        normalize=False
    )

    """ Generate initial dataset and random samples for posterior and ground truth evaluation """
    X = sampler.draw_samples(n=2*(true_objective.dim+1))
    rnd_X = sampler.draw_samples(n=1000)

    """ Instantiate a Mobo object """
    mobo = Mobo(
        experiment_name=experiment_name,
        X=X,
        Yobj=true_objective(X),
        Yobj_var=None,
        Ycon=None,
        Ycon_var=None,
        bounds=true_objective.bounds,
        optimization_problem_type=OptimizationProblemType.Maximization,
        true_objective=true_objective,
        objective=None,
        output_constraints=None,
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        sampler_type=SamplerType.Sobol,
        raw_samples=128,
        mc_samples=256,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        mobo.optimize()
        mobo.to_file()
        plot_multi_objective_from_RN_to_R2(
            mobo=mobo,
            show_ref_point=True,
            show_ground_truth=True,
            show_observations=True,
            f1_lims=(-250, 10),
            f2_lims=(-15, 0),
            display_figures=False,
            x=rnd_X,
        )

        """ Simulate experiment at new X """
        new_X = mobo.get_new_X()
        new_Yobj = BraninCurrin(negate=True)(new_X)
        print(f"New Yobj: {new_Yobj}")

        """ Save to csv """
        mobo.update_XY(new_X=new_X, new_Yobj=new_Yobj)
        mobo.save_dataset_to_csv()

    plot_log_hypervolume_improvement(mobo, show=False)
    plot_elapsed_time(mobo, show=False)
    create_video_from_images()
    print("Optimization Finished.")


if __name__ == "__main__":
    main_dir = Path.cwd().parent
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
