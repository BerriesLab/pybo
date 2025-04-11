import numpy
import numpy as np

from bayesian_optimization.bayesian_optimization import BayesianOptimization
from bayesian_optimization.enums import Kernel
from bayesian_optimization.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization.objective_functions import test_f0_1d

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(10)
lhs.set_bounds([(-5, 5)])
samples = lhs.sample_domain()

""" Save samples to a CSV file """
numpy.savetxt("../data/1d_dataset.csv", samples, delimiter=",", header="x1", comments="")

""" Run experiments and collect data - Simulation"""
X = numpy.loadtxt("../data/1d_dataset.csv", delimiter=",", skiprows=1).reshape(-1, 1)
Y = test_f0_1d(samples)
XY = numpy.hstack((X, Y.reshape(-1, 1)))
numpy.savetxt("../data/1d_dataset.csv", XY, delimiter=",", header="x1, y", comments="")

""" Load samples from the CSV file """
XY = numpy.loadtxt("../data/1d_dataset.csv", delimiter=",", skiprows=1)

""" Initialize the optimizer """
opt = BayesianOptimization()
opt.set_objective_function(test_f0_1d)
opt.set_experiment_name("test_1d_optimization")
opt.set_bounds([(-5, 5)])
opt.import_data("../data/1d_dataset.csv")
opt.set_observation_noise(0.001)
opt.set_acquisition_function("EI")
opt.set_kernel(Kernel.RBF.value(length_scale=1.0))
opt.set_number_of_optimizer_restarts(50)

for _ in range(10):
    """ Solve to get new x """
    opt.optimize()
    new_x = opt.get_new_X()
    opt.export_attributes("../data/1d_experiment.dat")

    """ Update the dataset with new x """
    XY = numpy.loadtxt("../data/1d_dataset.csv", delimiter=",", skiprows=1)
    XY = numpy.vstack((XY, numpy.hstack((new_x, np.nan * np.ones_like(new_x)))))
    numpy.savetxt("../data/1d_dataset.csv", XY, delimiter=",", header="x1, y", comments="")

    """ Run experiment at x_new and get new y """
    new_y = test_f0_1d(new_x)

    """ Update the dataset with new y """
    XY = numpy.loadtxt("../data/1d_dataset.csv", delimiter=",", skiprows=1)
    XY[-1, -1] = new_y
    numpy.savetxt("../data/1d_dataset.csv", XY, delimiter=",", header="x1, y", comments="")

    """ Re-instantiate an optimizer with attributes and update dataset """
    opt.import_attributes("../data/experiment.dat")
    opt.import_data("../data/1d_dataset.csv")