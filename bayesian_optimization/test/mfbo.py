import numpy as np

from bayesian_optimization.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization.multi_objective_bayesian_optimizaton import MultiObjectiveBayesianOptimization as Mobo
from bayesian_optimization.test.objective_functions import test_f0_wear, test_f0_speed

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(20)
lhs.set_bounds([(-5, 5), (-10, 10), (-8, 8)])
samples = lhs.sample_domain()

""" Save samples to a CSV file """
np.savetxt("../../data/2f0_mobo_dataset.csv", samples, delimiter=",", header="x1,x2,x3,y1,y2", comments="")

""" Run experiments and collect data - Simulation"""
X = np.loadtxt("../../data/2f0_mobo_dataset.csv", delimiter=",", skiprows=1)
Y1 = test_f0_wear(samples)
Y2 = test_f0_speed(samples)
XY = np.hstack((X, Y1.reshape(-1, 1), Y2.reshape(-1, 1)))
np.savetxt("../../data/2f0_mobo_dataset.csv", XY, delimiter=",", header="x1,x2,x3,y1,y2", comments="")

""" Load samples from the CSV file """
XY = np.loadtxt("../../data/2f0_mobo_dataset.csv", delimiter=",", skiprows=1)
print(XY)

mobo = Mobo()
mobo.set_bounds([(-5, 5), (-10, 10), (-8, 8)])
mobo.set_acquisition_function("ehvi")
mobo.set_observation_noise(1e-6)
mobo.set_dataset_X(XY[:, 0:3])
mobo.set_dataset_Y(XY[:, 3:])
mobo.optimize(live_plot=False)