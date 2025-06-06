import os
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective
from botorch.test_functions import C2DTLZ2
from pybo.output_constraints import UpperBound
from pybo.mobo import Mobo
from pybo.samplers import Sampler
from pybo.utils.make_video import create_video_from_images
from pybo.utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from pybo.utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_improvement, plot_elapsed_time


def main(n_samples=64, q: int = 1, ):
    data_dir = main_dir / "data"
    experiment_name = f"test_c2dtlz2_64samples_{q}q_256mc_128rs_qnehvi"
    directory = create_experiment_directory(data_dir, experiment_name)
    os.chdir(directory)

    """ Define the true_objective """
    true_objective = C2DTLZ2(num_objectives=2, dim=4, negate=True)  # Negate for maximization problems

    """ Instantiate a random generator """
    sampler = Sampler(
        sampler_type=SamplerType.Sobol,
        bounds=true_objective.bounds,
        n_dimensions=true_objective.dim,
        normalize=False,
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
        Ycon=-true_objective.evaluate_slack(X),
        Ycon_var=None,
        bounds=true_objective.bounds,
        optimization_problem_type=OptimizationProblemType.Maximization,
        true_objective=true_objective,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        output_constraints=[UpperBound(0)],
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        sampler_type=SamplerType.Sobol,
        raw_samples=128,
        mc_samples=256,
        batch_size=q,
        device=torch.device("cpu"),
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
            show_posterior=False,
            show_observations=True,
            f1_lims=(-1.6, 0.1),
            f2_lims=(-1.6, 0.1),
            display_figures=False,
            x=rnd_X,
        )

        """ Simulate experiment at new X """
        new_X = mobo.get_new_X()
        new_Yobj = true_objective(new_X)
        new_Ycon = -true_objective.evaluate_slack(new_X)
        print(f"New Yobj: {new_Yobj}")
        print(f"New Ycon: {new_Ycon}")

        """ Save to csv """
        mobo.update_XY(new_X=new_X, new_Yobj=new_Yobj, new_Ycon=new_Ycon)
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
