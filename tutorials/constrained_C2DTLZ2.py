import os
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective
from botorch.test_functions import C2DTLZ2
from mobo.constraints import UpperBound
from mobo.mobo import Mobo
from mobo.samplers import draw_samples
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_difference, plot_elapsed_time, \
    plot_allocated_memory


def main(batch_size: int = 1, ):
    experiment_name = f"test_c2dtlz2_64samples_{batch_size}q_1024mc_512rs_qlognehvi"
    main_directory = f"../data"
    initial_sampling_type = SamplerType.Sobol
    directory = create_experiment_directory(main_directory, experiment_name)
    os.chdir(directory)

    """ Define the true_objective, bounds and constraints"""
    n_objectives = 2
    n_dimensions = 4
    true_objective = C2DTLZ2(num_objectives=n_objectives, dim=n_dimensions,
                             negate=True)  # Negate for maximization problems
    bounds = true_objective.bounds
    constraints = [UpperBound(0)]  # None or a list of Callable
    objective = IdentityMCMultiOutputObjective(outcomes=[0, 1])

    """ Define the optimization parameters """
    n_samples = 64
    n_init_samples = 2 * (n_dimensions + 1)
    batch_size = batch_size
    mc_samples = 1024
    raw_samples = 512
    n_iterations = int(n_samples / batch_size)
    optimization_problem_type = OptimizationProblemType.Maximization
    acquisition_function_type = AcquisitionFunctionType.qLogNEHVI
    sampler_type = SamplerType.Sobol

    """ Generate initial dataset """
    X = draw_samples(
        sampler_type=initial_sampling_type,
        bounds=bounds,
        n_samples=n_init_samples,
        n_dimensions=n_dimensions
    )

    """ Simulate experiments """
    Yobj = true_objective(X)
    Yobj_var = None
    Ycon = -true_objective.evaluate_slack(X)
    Ycon_var = None

    """ Main optimization loop """
    mobo = Mobo(
        experiment_name=experiment_name,
        X=X,
        Yobj=Yobj,
        Yobj_var=Yobj_var,
        Ycon=Ycon,
        Ycon_var=Ycon_var,
        bounds=bounds,
        optimization_problem_type=optimization_problem_type,
        true_objective=true_objective,
        objective=objective,
        constraints=constraints,
        acquisition_function_type=acquisition_function_type,
        sampler_type=sampler_type,
        raw_samples=raw_samples,
        mc_samples=mc_samples,
        batch_size=batch_size,
    )

    for i in range(n_iterations):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{n_iterations} ***")

        mobo.optimize()
        mobo.to_file()
        plot_multi_objective_from_RN_to_R2(
            mobo=mobo,
            show_ref_point=True,
            show_ground_truth=True,
            show_posterior=True,
            show_rejected_observations=True,
            show_accepted_pareto_observations=True,
            show_accepted_non_pareto_observations=True,
            f1_lims=(-1.6, 0.1),
            f2_lims=(-1.6, 0.1),
            display_figures=False
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
        print(f"GPU Memory Allocated: {mobo.get_allocated_memory()[-1]:.2f} MB")

    plot_log_hypervolume_difference(mobo, show=False)
    plot_elapsed_time(mobo, show=False)
    plot_allocated_memory(mobo, show=False)
    print("Optimization Finished.")


if __name__ == "__main__":
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        main(batch_size)
