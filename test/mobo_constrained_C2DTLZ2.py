from bayesian_optimization_for_gpu.constraints import UpperBound
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from bayesian_optimization_for_gpu.objectives import IdentityMCMultiOutputObjectiveWrapper
from bayesian_optimization_for_gpu.samplers import draw_samples
from utils.cuda import *
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_difference
from botorch.test_functions.multi_objective import C2DTLZ2


experiment_name = "test0_c2dtlz2"
main_directory = f"../data"
initial_sampling_type = SamplerType.LatinHypercube
directory = create_experiment_directory(main_directory, experiment_name)
os.chdir(directory)

""" Define the problem, bounds and constraints"""
n_objectives = 2
n_dimensions = 4
problem = C2DTLZ2(dim=n_dimensions, num_objectives=n_objectives, negate=True)
bounds = problem.bounds
constraints = [UpperBound(0)]

""" Define the optimization parameters """
n_init_samples = 100
n_iterations = 50
batch_size = 1
monte_carlo_samples = 128
raw_samples = 256

""" Generate initial dataset """
X = draw_samples(sampler_type=SamplerType.LatinHypercube, bounds=bounds, n_samples=n_init_samples, n_dimensions=n_dimensions)
save_dataset_to_csv(X)

""" Simulate experiments """
Yobj = problem(X)
Yobj_var = None
Ycon = -problem.evaluate_slack(X)  # Negative values imply feasibility in BoTorch
Ycon_var = None
# TODO: decide how to save the variances
Y = torch.cat([Yobj, Ycon], dim=-1)
XY = np.concatenate([X.numpy(), Y.numpy()], axis=-1)
save_dataset_to_csv(XY)


""" Main optimization loop """
for i in range(n_iterations):

    print(f"*** Iteration {i + 1} ***")

    if i == 0:
        # Instantiate a new MOBO
        mobo = Mobo(experiment_name=experiment_name)
        mobo.set_X(X=X)
        mobo.set_Yobj(Yobj=Yobj)
        mobo.set_Yobj_var(Yobj_var=Yobj_var)
        mobo.set_Ycon(Ycon=Ycon)
        mobo.set_Ycon_var(Ycon_var=Ycon_var)
        mobo.set_bounds(bounds=bounds)
        mobo.set_true_objective(f0=problem)
        mobo.set_constraints(constraints=constraints)
        mobo.set_optimization_problem(optimization_problem_type=OptimizationProblemType.Minimization)
        mobo.set_acquisition_function(acquisition_function_type=AcquisitionFunctionType.qNEHVI)
        mobo.set_objective(IdentityMCMultiOutputObjectiveWrapper(outcomes=range(n_objectives)))
        mobo.set_sampler_type(sampler=SamplerType.Sobol)
        mobo.set_batch_size(batch_size=batch_size)
        mobo.set_MC_samples(MC_samples=monte_carlo_samples)
        mobo.set_raw_samples(raw_samples=raw_samples)

    else:
        # Load an existing MOBO
        mobo = Mobo.from_file()
        XY = load_dataset_from_csv()
        X = XY[:, 0:n_dimensions]
        Yobj = XY[:, n_dimensions:n_dimensions+n_objectives]
        Ycon = XY[:, n_dimensions+n_objectives:n_dimensions+n_objectives+len(constraints):]
        mobo.set_X(X=torch.tensor(X))
        mobo.set_Yobj(Yobj=torch.tensor(Yobj))
        mobo.set_Yobj_var(Yobj_var=torch.tensor(Yobj_var) if Ycon_var is not None else None)
        mobo.set_Ycon(Ycon=torch.tensor(Ycon))
        mobo.set_Ycon_var(Ycon_var=torch.tensor(Ycon_var) if Ycon_var is not None else None)


    mobo.optimize()
    mobo.to_file()
    # plot_multi_objective_from_RN_to_R2(mobo, ground_truth=True, posterior=True, show=False)
    plot_log_hypervolume_difference(mobo, show=True)

    # Evaluate the objective function at the new point(s).
    new_X = mobo.get_new_X().cpu().numpy()
    new_Yobj = problem(mobo.get_new_X()).cpu().numpy()
    new_Ycon = -problem.evaluate_slack(mobo.get_new_X()).cpu().numpy()

    # Update the models with the new data.
    X = np.concatenate([X, new_X], axis=0)
    Y = np.concatenate([Y, np.concatenate([new_Yobj, new_Ycon], axis=-1)], axis=0)

    """ Save to csv """
    XY = np.concatenate((X, Y), axis=-1)
    save_dataset_to_csv(XY)
