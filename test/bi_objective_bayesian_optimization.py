import numpy as np

from bayesian_optimization_for_cpu.enums import Kernel
from bayesian_optimization_for_cpu.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization_for_cpu.multi_objective_bayesian_optimization import \
    MultiObjectiveBayesianOptimization as Mobo
from objective_functions import test_f0_wear, test_f0_speed

""" Inputs """
f0: list[callable] = [test_f0_wear, test_f0_speed]
n_samples: int = 20  # Number of initial samples
bounds: list[tuple[float, float]] = [(-5, 5), (-5, 5), (-8, 8)]  # Domain bounds
n: list[int] = [100, 100, 200]
header: str = "x1,x2,x3,y1,y2"  # Dynamic header for objectives
n_experiments: int = 20  # Number of optimization steps to take = 1 in practical settings

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(n_samples)
lhs.set_bounds(bounds)
samples = lhs.sample_domain()

""" Save samples to a CSV file """
np.savetxt("../data/mobo_dataset.csv", samples, delimiter=",", header=header, comments="")

""" Run experiments and collect data - Simulation"""
X = np.loadtxt("../data/mobo_dataset.csv", delimiter=",", skiprows=1)
Ys = [f(samples).reshape(-1, 1) for f in f0]  # Compute all objective outputs
XY = np.hstack((X, *Ys))  # Combine inputs and all objectives
np.savetxt("../data/mobo_dataset.csv", XY, delimiter=",", header=header, comments="")

""" Make a number of experiments """
for i in range(n_samples, n_samples + n_experiments):

    if i == n_samples:

        mobo = Mobo()
        mobo.set_experiment_name("test_mobo_optimization")
        mobo.set_n_objectives = len(f0)
        mobo.set_bounds(bounds=bounds, n=n)
        mobo.set_acquisition_function("ehvi")
        mobo.set_kernel(Kernel.RBF.value(length_scale=1.0))
        mobo.set_observation_noise(1e-6)
        mobo.import_data("../data/mobo_dataset.csv")
        mobo.set_number_of_optimizer_restarts(50)

    else:
        """ Instantiate optimizer, then import data and model """
        mobo = Mobo()
        mobo.import_model(filepath, format="pickle")
        mobo.import_data("../data/1d_dataset.csv")

    """ Solve to get new X """
    mobo.optimize(live_plot=True)

    """ Save mode, data and figure to disc """
    mobo.save_figure_to_disc(directory="../data")
    mobo.save_model_to_disc(directory="../data", format="pickle")
    mobo.save_model_to_disc(directory="../data", format="json")
    print(f"Iteration {i + 1}, new_x = {new_X}")

    """ Update the dataset with new x """
    new_X = mobo.get_new_X()
    XY = np.loadtxt("../data/2f0_mobo_dataset.csv", delimiter=",", skiprows=1)
    new_XY = np.nan * np.ones((1, XY.shape[1]))
    new_XY[0, 0:3] = new_X
    XY = np.vstack((XY, new_XY))
    np.savetxt("../data/2f0_mobo_dataset.csv", XY, delimiter=",", header="x1, y", comments="")

    """ Run experiment at new_X and get new_Ys """
    new_Ys = [f(new_X) for f in f0]  # Evaluate all objectives for new_X

    """ Update the dataset with new_Ys """
    XY = np.loadtxt("../data/2f0_mobo_dataset.csv", delimiter=",", skiprows=1)
    XY[-1, -len(f0):] = [new_Y.item() for new_Y in new_Ys]  # Update all objectives
    np.savetxt("../data/2f0_mobo_dataset.csv", XY, delimiter=",", header=header, comments="")
