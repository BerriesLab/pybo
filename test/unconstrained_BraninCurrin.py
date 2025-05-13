import os
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from bayesian_optimization_for_gpu.samplers import draw_samples
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_difference, plot_elapsed_time, \
    plot_allocated_memory
from botorch.test_functions.multi_objective import BraninCurrin


experiment_name = "test_branincurrin_50iter_4q_1024mc_512rs"
main_directory = f"../data"
initial_sampling_type = SamplerType.Sobol
directory = create_experiment_directory(main_directory, experiment_name)
os.chdir(directory)

""" Define the true_objective, bounds and constraints"""
true_objective = BraninCurrin(negate=True)  # Negate for maximization problems
n_objectives = true_objective.num_objectives
n_dimensions = true_objective.dim
bounds = true_objective.bounds
constraints = None
objective = None  # IdentityMCMultiOutputObjective(outcomes=(0, 1))

""" Define the optimization parameters """
n_init_samples = 2 * (n_dimensions + 1)
n_iterations = 2
batch_size = 4
mc_samples = 1024
raw_samples = 512
optimization_problem_type = OptimizationProblemType.Maximization
acquisition_function_type = AcquisitionFunctionType.qNEHVI
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
Ycon = None
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

# mobo = Mobo(experiment_name=experiment_name)
# mobo.set_X(X=X)
# mobo.set_Yobj(Yobj=Yobj)
# mobo.set_Yobj_var(Yobj_var=Yobj_var)
# mobo.set_Ycon(Ycon=Ycon)
# mobo.set_Ycon_var(Ycon_var=Ycon_var)
# mobo.set_optimization_problem(optimization_problem_type=optimization_problem_type)
# mobo.set_bounds(bounds=bounds)
# mobo.set_true_objective(true_objective=true_objective)
# mobo.set_objective(objective=objective)
# mobo.set_constraints(constraints=constraints)
# mobo.set_acquisition_function(acquisition_function_type=acquisition_function_type)
# mobo.set_sampler_type(sampler_type=sampler_type)
# mobo.set_raw_samples(raw_samples=raw_samples)
# mobo.set_batch_size(batch_size=batch_size)
# mobo.set_MC_samples(MC_samples=monte_carlo_samples)


for i in range(n_iterations):

    print("\n\n")
    print(f"*** Iteration {i + 1} ***")

    # if i > 0:
        # mobo = Mobo.from_file()
        # X = torch.cat([X.to(mobo.get_device(), mobo.get_dtype()), new_X], dim=0)
        #Yobj = torch.cat([mobo.get_Yobj().to(mobo.get_device(), mobo.get_dtype()), new_Yobj], dim=0)
        # mobo.set_X(X=X)
        # mobo.set_Yobj(Yobj=Yobj)

    mobo.optimize()
    mobo.to_file()
    plot_multi_objective_from_RN_to_R2(
        mobo=mobo,
        ref_point=True,
        ground_truth=True,
        posterior=True,
        f1_lims=(-250, 10),
        f2_lims=(-15, 0),
        show=False
    )

    """ Simulate experiment at new X """
    new_X = mobo.get_new_X()
    new_Yobj = true_objective(new_X)
    # new_Ycon = -true_objective.evaluate_slack(new_X)
    print(f"New Yobj: {new_Yobj}")
    # print(f"New Ycon: {new_Ycon}")

    """ Save to csv """
    mobo.update_XY(new_X = new_X, new_Yobj=new_Yobj)
    mobo.save_dataset_to_csv()
    print(f"GPU Memory Allocated: {mobo.get_allocated_memory()[-1]:.2f} MB")
    # del mobo

plot_log_hypervolume_difference(mobo, show=False)
plot_elapsed_time(mobo, show=False)
plot_allocated_memory(mobo, show=False)
print("Optimization Finished.")
