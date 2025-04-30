from bayesian_optimization_for_gpu.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from utils.cuda import *
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType
from utils.plotters import plot_multi_objective_from_RN_to_R2
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.sampling import draw_sobol_samples


experiment_name = "test_mobo_from_R3_to_R2"
main_directory = f"../data"
directory = create_experiment_directory(main_directory, experiment_name)
os.chdir(directory)

n_samples = 10
problem = BraninCurrin(negate=True)
bounds = problem.bounds
n_opt_iter = 50

# generate training data
NOISE_SE = torch.tensor([15.19, 0.63])
X = draw_sobol_samples(bounds=problem.bounds, n=n_samples, q=1).squeeze(1)

""" Sample domain """
# lhs = LatinHypercubeSampling()
# lhs.set_n_samples(n_samples)
# lhs.set_bounds(bounds)
# X = lhs.sample_domain()
save_dataset_to_csv(X)

""" Simulate experiments """
X = torch.tensor(X)
Y = problem(X) + torch.randn_like(problem(X)) * NOISE_SE
Yvar = None
# Y = problem(X)
bounds = torch.tensor(bounds)

""" Save to csv """
XY = np.concatenate((X.cpu().numpy(), Y.cpu().numpy()), axis=1)
save_dataset_to_csv(XY)

""" Main optimization loop """
for i in range(n_opt_iter):

    print(f"*** Iteration {i + 1} ***")

    if i == 0:
        # Instantiate a new MOBO
        mobo = Mobo(experiment_name="test_mobo_from_R3_to_R2")
        mobo.set_X(X)
        mobo.set_Y(Y)
        mobo.set_Yvar(Yvar)
        mobo.set_bounds(bounds)
        mobo.set_acquisition_function(AcquisitionFunctionType.qLogEHVI)
        mobo.set_sampler(SamplerType.Sobol)
        mobo.set_batch_size(1)

    else:
        # Load an existing MOBO
        mobo = Mobo.from_file()
        XY = load_dataset_from_csv()
        mobo.set_X(torch.tensor(XY[:, 0:-2]))
        mobo.set_Y(torch.tensor(XY[:, -2:]))
        mobo.set_Yvar(Yvar)


    mobo.optimize()
    mobo.to_file()
    plot_multi_objective_from_RN_to_R2(mobo, directory="../data", show=False)

    # Evaluate the objective function at the new point(s).
    new_X = mobo.get_new_X().cpu().numpy()
    new_Y = problem(mobo.get_new_X()).cpu().numpy()

    # Update the models with the new data.
    X = np.concatenate([X, new_X], axis=0)
    Y = np.concatenate([Y, new_Y], axis=0)

    """ Save to csv """
    XY = np.concatenate((X, Y), axis=1)
    save_dataset_to_csv(XY)

