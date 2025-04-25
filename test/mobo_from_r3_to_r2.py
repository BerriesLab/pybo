import numpy as np

from bayesian_optimization_for_cpu.enums import Kernel
from bayesian_optimization_for_cpu.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization_for_cpu.mobo import \
    MultiObjectiveBayesianOptimization as Mobo
from objective_functions import test_f0_wear, test_f0_machining_time

""" Inputs """
n_objectives = 2
f0: list[callable] = [test_f0_wear, test_f0_machining_time]
n_samples: int = 20  # Number of initial samples
bounds: list[tuple[float, float]] = [(-5, 5), (-5, 5), (-5, 5)]  # Domain bounds
header: str = "x1,x2,x3,y1"
n_experiments: int = 20  # Number of optimization steps to take = 1 in practical settings
filepath: str = ""

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(n_samples)
lhs.set_bounds(bounds)
samples = lhs.sample_domain()
np.savetxt("../data/mobo_dataset.csv", samples, delimiter=",", header=header, comments="")

""" Simulate User: Run experiments and collect data"""
X = np.loadtxt("../data/mobo_dataset.csv", delimiter=",", skiprows=1)
Y = np.array([f(samples) for f in f0]).T  # Compute all objective outputs
XY = np.hstack((X, Y))  # Combine inputs and all objectives
np.savetxt("../data/mobo_dataset.csv", XY, delimiter=",", header=header, comments="")

""" Make a number of experiments """
for i in range(n_samples, n_samples + n_experiments):

    if i == n_samples:

        # Instantiate new Mobo
        mobo = Mobo(
            n_objectives=n_objectives,
            experiment_name="test_mobo_from_R3_to_R2_gpu",
            bounds=bounds,
            acquisition_function="ehvi",
            f0=f0,
            kernel=Kernel.RBF.value(length_scale=1.0),
            observation_noise=1e-6,
            n_optimizer_restarts=100,
            X=X,
            Y=Y,
        )

    else:
        # Load Mobo from disk
        mobo = Mobo.load_from_disk(filepath)
        mobo.import_XY_from_csv("../data/mobo_dataset.csv")

    """ Solve to get new X """
    mobo.optimize(plot=True)

    """ Save mode, data and figure to disc """
    mobo.save_figure_to_disc(directory="../data")
    filepath = mobo.save_to_disk(directory="../data")
    new_X = mobo.get_new_X()
    print(f"Iteration {i + 1}, new_x = {new_X}")

    """ Simulate User: Update the dataset with new x """
    XY = np.loadtxt("../data/mobo_dataset.csv", delimiter=",", skiprows=1)
    new_XY = np.nan * np.ones((1, XY.shape[1]))
    new_XY[0, 0:-n_objectives] = new_X
    XY = np.vstack((XY, new_XY))
    np.savetxt("../data/mobo_dataset.csv", XY, delimiter=",", header=header, comments="")

    """ Simulate User: Run experiment at new_X and get new_Ys """
    new_Ys = [f(new_X) for f in f0]  # Evaluate all objectives for new_X

    """ Simulate User: Update the dataset with new_Ys """
    XY = np.loadtxt("../data/mobo_dataset.csv", delimiter=",", skiprows=1)
    XY[-1, -n_objectives:] = [new_Y.item() for new_Y in new_Ys]  # Update all objectives
    np.savetxt("../data/mobo_dataset.csv", XY, delimiter=",", header=header, comments="")
