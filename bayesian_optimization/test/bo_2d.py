import numpy
import numpy as np

from bayesian_optimization.bayesian_optimization import BayesianOptimization
from bayesian_optimization.enums import Kernel
from bayesian_optimization.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization.test.objective_functions import test_f0_2d

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(10)
lhs.set_bounds([(-5, 5), (-10, 10)])
samples = lhs.sample_domain()

""" Save samples to a CSV file """
numpy.savetxt("../../data/2d_dataset.csv", samples, delimiter=",", header="x1,x2,y", comments="")

""" Run experiments and collect data - Simulation"""
X = numpy.loadtxt("../../data/2d_dataset.csv", delimiter=",", skiprows=1)
Y = test_f0_2d(samples)
XY = numpy.hstack((X, Y.reshape(-1, 1)))
numpy.savetxt("../../data/2d_dataset.csv", XY, delimiter=",", header="x1,x2,y", comments="")

""" Load samples from the CSV file """
XY = numpy.loadtxt("../../data/2d_dataset.csv", delimiter=",", skiprows=1)

""" Initialize the optimizer """
opt = BayesianOptimization()
opt.set_objective_function(test_f0_2d)
opt.set_experiment_name("test_2d_optimization")
opt.set_bounds([(-5, 5), (-10, 10)])
opt.set_observation_noise(1e-6)
opt.set_acquisition_function("EI")
opt.set_kernel(Kernel.RBF.value(length_scale=1.0))
opt.set_number_of_optimizer_restarts(100)

for i in range(2):
    """ Import data and model """
    opt.import_data("../../data/2d_dataset.csv")
    if i > 0:
        filepath = opt._compose_filepath(directory="../../data", previous=True)
        opt.import_attributes(filepath, format="pickle")

    """ Solve to get new x """
    opt.optimize(live_plot=True)
    opt.save_figure_to_disc(directory="../../data")
    opt.save_attributes_to_disc(directory="../../data", format="pickle")
    opt.save_attributes_to_disc(directory="../../data", format="json")
    new_x = opt.get_new_X()
    print(f"Iteration {i + 1}, new_x = {new_x}")

    """ Update the dataset with new x """
    XY = numpy.loadtxt("../../data/2d_dataset.csv", delimiter=",", skiprows=1)
    XY = numpy.vstack((XY, np.hstack((new_x.reshape(1, -1), np.nan * np.ones((1, 1))))))
    numpy.savetxt("../../data/2d_dataset.csv", XY, delimiter=",", header="x1,x2,y", comments="")

    """ Run experiment at new_x and get new y """
    new_y = test_f0_2d(new_x.reshape(1, -1))

    """ Update the dataset with new y """
    XY = numpy.loadtxt("../../data/2d_dataset.csv", delimiter=",", skiprows=1)
    XY[-1, -1] = new_y.item()
    numpy.savetxt("../../data/2d_dataset.csv", XY, delimiter=",", header="x1,x2,y", comments="")