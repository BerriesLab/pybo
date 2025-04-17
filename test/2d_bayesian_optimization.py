import numpy
import numpy as np

from bayesian_optimization_for_cpu.bayesian_optimization import BayesianOptimization
from bayesian_optimization_for_cpu.enums import Kernel
from bayesian_optimization_for_cpu.latin_hypercube_sampling import LatinHypercubeSampling
from objective_functions import test_f0_2d

""" Sample domain """
n_samples = 10
lhs = LatinHypercubeSampling()
lhs.set_n_samples(n_samples)
lhs.set_bounds([(-5, 5), (-8, 8)])
samples = lhs.sample_domain()

""" Save samples to a CSV file """
numpy.savetxt("../data/2d_dataset.csv", samples, delimiter=",", header="x1,x2", comments="")

""" Run experiments and collect data - Simulation"""
X = numpy.loadtxt("../data/2d_dataset.csv", delimiter=",", skiprows=1).reshape(-1, 1)
Y = test_f0_2d(samples)
XY = numpy.hstack((X.reshape(-1, 2), Y.reshape(-1, 1)))
numpy.savetxt("../data/2d_dataset.csv", XY, delimiter=",", header="x1,x2,y", comments="")

""" Make a number of experiments """
n_experiments = 20
for i in range(n_samples, n_samples + n_experiments):

    if i == n_samples:
        """ Initialize the Bayesian Optimizer - Its parameters must stay the same across 
        all futures optimization steps """
        opt = BayesianOptimization()
        opt.set_objective_function(test_f0_2d)
        opt.set_experiment_name("test_2d_optimization")
        opt.set_bounds([(-5, 5), (-8, 8)])
        opt.set_observation_noise(1e-6)
        opt.set_acquisition_function("EI")
        opt.set_kernel(Kernel.RBF.value(length_scale=1.0))
        opt.set_number_of_optimizer_restarts(100)
        XY = numpy.loadtxt("../data/2d_dataset.csv", delimiter=",", skiprows=1)
        opt.set_dataset_X(XY[:, 0:2])
        opt.set_dataset_Y(XY[:, 2:])

    else:
        """ Instantiate optimizer, then import data and model """
        opt = BayesianOptimization()
        opt.import_model(filepath, format="pickle")
        opt.import_data("../data/2d_dataset.csv")

    """ Solve to get new x """
    opt.optimize()
    opt.save_figure_to_disc(directory="../data")
    filepath = opt.save_model_to_disc(directory="../data", format="pickle")
    opt.save_model_to_disc(directory="../data", format="json")
    new_X = opt.get_new_X()
    print(f"Iteration {i + 1}, new_x = {new_X}")

    """ Update the dataset with new x """
    XY = numpy.loadtxt("../data/2d_dataset.csv", delimiter=",", skiprows=1)
    XY = numpy.vstack((XY, numpy.hstack((new_X, np.nan * np.ones((1, 1))))))
    numpy.savetxt("../data/2d_dataset.csv", XY, delimiter=",", header="x1,x2,y", comments="")

    """ Run experiment at new_x and get new y """
    new_y = test_f0_2d(new_X)

    """ Update the dataset with new y """
    XY = numpy.loadtxt("../data/2d_dataset.csv", delimiter=",", skiprows=1)
    XY[-1, -1] = new_y.item()
    numpy.savetxt("../data/2d_dataset.csv", XY, delimiter=",", header="x1,x2,y", comments="")
