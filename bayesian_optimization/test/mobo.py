import numpy as np

from bayesian_optimization.enums import Kernel
from bayesian_optimization.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization.multi_objective_bayesian_optimization.multi_objective_bayesian_optimizaton import \
    MultiObjectiveBayesianOptimization as Mobo
from bayesian_optimization.test.objective_functions import test_f0_wear, test_f0_speed

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(20)
lhs.set_bounds([(-5, 5), (-5, 5), (-8, 8)])
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

mobo = Mobo()
mobo.set_experiment_name("test_mobo_optimization")
mobo.set_bounds([(-5, 5), (-10, 10), (-8, 8)], n=10)
mobo.set_acquisition_function("ehvi")
mobo.set_kernel(Kernel.RBF.value(length_scale=1.0))
mobo.set_observation_noise(1e-6)
mobo.set_dataset_X(XY[:, 0:3])
mobo.set_dataset_Y(XY[:, 3:])
mobo.set_number_of_optimizer_restarts(50)

for i in range(5):
    """ Import data and model """
    mobo.import_data("../../data/2f0_mobo_dataset.csv", n_features=3)
    if i > 0:
        filepath = mobo._compose_filepath(directory="../../data", previous=True)
        mobo.import_attributes(filepath, format="pickle")

    """ Solve to get new x """
    mobo.optimize(live_plot=True)
    mobo.save_figure_to_disc(directory="../../data")
    mobo.save_attributes_to_disc(directory="../../data", format="pickle")
    mobo.save_attributes_to_disc(directory="../../data", format="json")
    new_X = mobo.get_new_X()
    print(f"Iteration {i + 1}, new_x = {new_X}")

    """ Update the dataset with new x """
    XY = np.loadtxt("../../data/2f0_mobo_dataset.csv", delimiter=",", skiprows=1)
    new_XY = np.nan * np.ones((1, XY.shape[1]))
    new_XY[0, 0:3] = new_X
    XY = np.vstack((XY, new_XY))
    np.savetxt("../../data/2f0_mobo_dataset.csv", XY, delimiter=",", header="x1, y", comments="")

    """ Run experiment at new_X and get new_Ys """
    new_Y1 = test_f0_wear(new_X)
    new_Y2 = test_f0_speed(new_X)
    # XY = np.hstack((X, Y1.reshape(-1, 1), Y2.reshape(-1, 1)))

    """ Update the dataset with new_Ys """
    XY = np.loadtxt("../../data/2f0_mobo_dataset.csv", delimiter=",", skiprows=1)
    XY[-1, -2] = new_Y1.item()
    XY[-1, -1] = new_Y2.item()
    np.savetxt("../../data/2f0_mobo_dataset.csv", XY, delimiter=",", header="x1, y", comments="")
