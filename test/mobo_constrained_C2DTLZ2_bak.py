import torch

import utils.cuda
from bayesian_optimization_for_gpu.constraints import UpperBound
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from bayesian_optimization_for_gpu.objectives import IdentityMCMultiOutputObjectiveWrapper
from bayesian_optimization_for_gpu.samplers import draw_samples
from utils.cuda import *
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_difference
from botorch.test_functions.multi_objective import C2DTLZ2
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective


experiment_name = "test0_c2dtlz2"
main_directory = f"../data"
initial_sampling_type = SamplerType.LatinHypercube
directory = create_experiment_directory(main_directory, experiment_name)
os.chdir(directory)

""" Define the true_objective, bounds and constraints"""
n_objectives = 2
n_dimensions = 4
true_objective = C2DTLZ2(dim=n_dimensions, num_objectives=n_objectives, negate=True)
bounds = true_objective.bounds
constraints = [UpperBound(0)]
objective = IdentityMCMultiOutputObjective(outcomes=(0, 1))

""" Define the optimization parameters """
n_init_samples = 10
n_iterations = 20
batch_size = 1
monte_carlo_samples = 128
raw_samples = 512

""" Generate initial dataset """
X = draw_samples(
    sampler_type=SamplerType.LatinHypercube,
    bounds=bounds,
    n_samples=n_init_samples,
    n_dimensions=n_dimensions
)
#save_dataset_to_csv(X=X, Yobj=None, Yobj_var=None, Ycon=None, Ycon_var=None)

""" Simulate experiments """
Yobj = true_objective(X)
Yobj_var = None
Ycon = -true_objective.evaluate_slack(X)  # Negative values imply feasibility in BoTorch
Ycon_var = None
#save_dataset_to_csv(X=X, Yobj=Yobj, Yobj_var=Yobj_var, Ycon=Ycon, Ycon_var=Ycon_var)

""" Main optimization loop """
mobo = Mobo(experiment_name=experiment_name)
mobo.set_X(X=X)
mobo.set_Yobj(Yobj=Yobj)
mobo.set_Yobj_var(Yobj_var=Yobj_var)
mobo.set_Ycon(Ycon=Ycon)
mobo.set_Ycon_var(Ycon_var=Ycon_var)
mobo.set_bounds(bounds=bounds)
mobo.set_true_objective(true_objective=true_objective)
mobo.set_objective(objective=objective)
mobo.set_constraints(constraints=constraints)
mobo.set_optimization_problem(optimization_problem_type=OptimizationProblemType.Maximization)
mobo.set_acquisition_function(acquisition_function_type=AcquisitionFunctionType.qNEHVI)
mobo.set_sampler_type(sampler=SamplerType.Sobol)
mobo.set_batch_size(batch_size=batch_size)
mobo.set_MC_samples(MC_samples=monte_carlo_samples)
mobo.set_raw_samples(raw_samples=raw_samples)

for i in range(n_iterations):

    print(f"*** Iteration {mobo.get_iteration_number() + 1} ***")

    if i > 0:
        mobo.set_X(X)
        mobo.set_Yobj(Yobj)
        mobo.set_Ycon(Ycon)
        #mobo = Mobo.from_file()
        #mobo.load_dataset_from_csv()

    mobo.optimize()
    mobo.to_file()
    # plot_multi_objective_from_RN_to_R2(mobo, ground_truth=True, posterior=True, show=False)

    """ Simulate experiment at new X """
    new_X = mobo.get_new_X()
    new_Yobj = true_objective(new_X)
    new_Ycon = -true_objective.evaluate_slack(new_X)

    """ Save to csv """
    X = torch.cat([X.to(mobo.get_device(), mobo.get_dtype()), new_X], dim=0)
    Yobj = torch.cat([mobo.get_Yobj(), new_Yobj], dim=0)
    Ycon = torch.cat([mobo.get_Ycon(), new_Ycon], dim=0)
    #save_dataset_to_csv(X=X, Yobj=Yobj, Yobj_var=Yobj_var, Ycon=Ycon, Ycon_var=Ycon_var)

plot_log_hypervolume_difference(mobo, show=True)
