from torch.quasirandom import SobolEngine

from bayesian_optimization_for_gpu.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from utils.cuda import *
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R3
from botorch.test_functions.multi_objective import DTLZ2


experiment_name = "test_mobo_from_RN_to_R3"
main_directory = f"../data"
directory = create_experiment_directory(main_directory, experiment_name)
os.chdir(directory)

""" Define the problem and bounds"""
n_objectives = 3
problem = DTLZ2(dim=6, num_objectives=n_objectives)
bounds = problem.bounds

""" Define the optimization parameters """
n_init_samples = 20
n_iterations = 50
batch_size = 1
monte_carlo_samples = 128
raw_samples = 256

""" Generate initial dataset """
# lhs = LatinHypercubeSampling()
# lhs.set_n_samples(n_samples)
# lhs.set_bounds(bounds)
# X = lhs.sample_domain()
sobol = SobolEngine(dimension=6, scramble=True)
X = sobol.draw(n=n_init_samples)
save_dataset_to_csv(X)

""" Simulate experiments """
Y = problem(X)
Yvar = None
XY = np.concatenate((X.numpy(), Y.numpy()), axis=1)
save_dataset_to_csv(XY)

""" Main optimization loop """
for i in range(n_iterations):

    print(f"*** Iteration {i + 1} ***")

    if i == 0:
        # Instantiate a new MOBO
        mobo = Mobo(experiment_name=experiment_name)
        mobo.set_X(X)
        mobo.set_Y(Y)
        mobo.set_Yvar(Yvar)
        mobo.set_bounds(bounds)
        mobo.set_f0(problem)
        mobo.set_acquisition_function(AcquisitionFunctionType.qLogEHVI)
        mobo.set_optimization_problem_type(OptimizationProblemType.Maximization)
        mobo.set_sampler(SamplerType.Sobol)
        mobo.set_batch_size(batch_size)
        mobo.set_MC_samples(monte_carlo_samples)
        mobo.set_raw_samples(raw_samples)

    else:
        # Load an existing MOBO
        mobo = Mobo.from_file()
        XY = load_dataset_from_csv()
        mobo.set_X(torch.tensor(XY[:, 0:-n_objectives]))
        mobo.set_Y(torch.tensor(XY[:, -n_objectives:]))
        mobo.set_Yvar(Yvar)


    mobo.optimize()
    mobo.to_file()
    plot_multi_objective_from_RN_to_R3(mobo, ground_truth=True, show=False)

    # Evaluate the objective function at the new point(s).
    new_X = mobo.get_new_X().cpu().numpy()
    new_Y = problem(mobo.get_new_X()).cpu().numpy()

    # Update the models with the new data.
    X = np.concatenate([X, new_X], axis=0)
    Y = np.concatenate([Y, new_Y], axis=0)

    """ Save to csv """
    XY = np.concatenate((X, Y), axis=1)
    save_dataset_to_csv(XY)

