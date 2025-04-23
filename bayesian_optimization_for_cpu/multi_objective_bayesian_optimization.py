import json
import os
import pickle
from collections.abc import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor


class MultiObjectiveBayesianOptimization:

    def __init__(
            self,
            n_objectives: int,
            experiment_name: str,
            bounds: list[tuple[float, float]],
            acquisition_function: str,
            f0: list[Callable] | None,
            kernel: str,
            X: np.ndarray,  # The X dataset (n samples x N dimensions)
            Y: np.ndarray,  # The Y dataset (n samples x M objectives)
            observation_noise: float = 1e-12,
            n_optimizer_restarts: int = 100):

        # Validate passed arguments
        self._validate_XY(X, Y)
        self._validate_f0(n_objectives, f0)

        # Attributes required at initialization
        self._n_objectives = n_objectives
        self._experiment_name = experiment_name
        self._datetime = datetime.now()
        self._observation_noise = observation_noise
        self._n_optimizer_restarts = n_optimizer_restarts
        self._bounds = bounds
        self._acquisition_function = acquisition_function
        self._f0 = f0
        self._kernel = kernel
        self._X = X
        self._Y = Y

        # Optimization State Attributes
        self._domain: np.ndarray | None = None
        self._model: list[GaussianProcessRegressor] = []
        self._pareto_front = None
        self._ref_point = None
        self._new_X = None  # The new X location
        
        # Plotting Attributes
        self._fig = None
        self._ax = None
        self._mu: list[np.ndarray] | None = None
        self._sigma: list[np.ndarray] | None = None

    """ Setters and getters """

    def set_experiment_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        self._experiment_name = name

    def get_experiment_name(self):
        return self._experiment_name

    def set_n_objectives(self, n_objectives: int):
        if not isinstance(n_objectives, int) or n_objectives <= 0:
            raise ValueError("The number of objectives must be a positive integer.")
        self._n_objectives = n_objectives

    def get_n_objectives(self):
        return self._n_objectives

    def set_dataset_X(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise ValueError("X_train must be a numpy array.")
        self._X = X

    def get_dataset_X(self):
        return self._X

    def set_dataset_Y(self, Y: np.ndarray):
        if not isinstance(Y, np.ndarray):
            raise ValueError("Y_train must be a numpy array.")
        self._Y = Y

    def get_dataset_Y(self):
        return self._Y

    def set_observation_noise(self, noise):
        if noise is None or noise == 0:
            self._observation_noise = 1e-10
            return
        self._observation_noise = noise

    def get_observation_noise(self):
        return self._observation_noise

    def get_new_X(self):
        return self._new_X

    # TODO: n is not used anymore to set the bounds
    def set_bounds(self, bounds, n: None or list[int] = None):
        """ Set bounds. If n is None, then n is set to 100 for each bound."""
        if n is None:
            n = [100] * len(bounds)
        self._validate_bounds(bounds, n)
        self._bounds = bounds
        self._build_domain_from_bounds(n=n)

    def get_bounds(self):
        return self._bounds

    def set_acquisition_function(self, acquisition_function: str):
        supported_functions = ['ei', 'ehvi']
        if acquisition_function not in supported_functions:
            raise ValueError(f"Invalid acquisition function. Supported values are {supported_functions}.")
        self._acquisition_function = acquisition_function

    def get_acquisition_function(self):
        return self._acquisition_function

    # TODO: the f0 now is a list of callable
    def set_objective_function(self, objective_function):
        if not callable(objective_function):
            raise ValueError("Objective function must be callable.")
        self._f0 = objective_function

    def get_objective_function(self):
        return self._f0

    def get_number_of_objective_function_calls(self):
        return self._n_iter

    def set_number_of_optimizer_restarts(self, n_restarts: int):
        if not isinstance(n_restarts, int) or n_restarts <= 0:
            raise ValueError("The number of optimizer restarts must be a positive integer.")
        self._n_optimizer_restarts = n_restarts

    def get_number_of_optimizer_restarts(self):
        return self._n_optimizer_restarts

    def set_number_of_iterations(self, n_iter: int):
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")
        self._n_iter = n_iter

    def get_number_of_iterations(self):
        return self._n_iter

    def set_kernel(self, kernel: str):
        self._kernel = kernel

    def get_kernel(self):
        return self._kernel

    """ I/O """

    # TODO: add json
    @classmethod
    def load_from_disk(cls, file_path: str):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected object of type {cls.__name__}, got {type(obj).__name__}")
        return obj

    # TODO: add json
    def save_to_disk(self, directory: str):
        filepath = self._compose_filepath(directory) + ".dat"
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        return filepath

    def import_XY_from_csv(self, filename):
        """ Assume that the first row is a header """
        XY = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.set_dataset_X(XY[:, 0:-self._n_objectives])
        self.set_dataset_Y(XY[:, -self._n_objectives:])

    def export_XY_to_csv(self, filename):
        if self._X is None or self._Y is None:
            raise ValueError("No data available to export. Ensure that _X and _Y are set.")
        XY = np.hstack((self._X, self._Y))
        np.savetxt(filename, XY, delimiter=",", comments="")

    def save_figure_to_disc(self, directory):
        # Handle base case
        if not self._fig or not self._ax:
            raise ValueError("No plot is available. Cannot export empty plot.")

        filepath = self._compose_filepath(directory)
        try:
            self._fig.savefig(f"{filepath}.png")
            plt.close(self._fig)
        except IOError:
            print(f"Warning: File '{filepath}' could not be saved. Continuing operation.")
            return

    """ Optimizer """

    def optimize(self, live_plot=True):
        self._instantiate_gaussian_process()
        self._fit_gaussian_process()
        self._calculate_pareto_front()
        self._calculate_reference_point()
        self._minimize_acquisition_function()
        if live_plot:
            self._live_plot()

    def _fit_gaussian_process(self):
        """ Fit a Gaussian Process Regressor for each objective. """
        for i in range(self._n_objectives):
            self._model[i].fit(self._X, self._Y[:, i])

    def _instantiate_gaussian_process(self):
        """ Instantiate a Gaussian Process Regressor for each objective. """
        for i in range(self._n_objectives):
            self._instantiate_a_gaussian_process()

    def _instantiate_a_gaussian_process(self):
        """ Instantiate a Gaussian Process Regressor. """
        gpr = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self._observation_noise,  # Noise added to the diagonal of the input matrix
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=self._n_optimizer_restarts, )
        self._model.append(gpr)

    def _calculate_pareto_front(self):
        """
        Args: Y (np.ndarray): Points to calculate the Pareto front from. Shape: (n_points, n_objectives)
        Returns: np.ndarray: Pareto front of Y. Shape: (n_pareto, n_objectives)
        """
        is_efficient = np.ones(self._Y.shape[0], dtype=bool)
        for idx, value in enumerate(self._Y):
            if is_efficient[idx]:
                # A point is set as efficient if at least one of its objectives is less than the current
                # point's objectives OR if all objectives are equal.  This ensures that points that are
                # equal to the current point are also kept.
                is_efficient[is_efficient] = (np.any(self._Y[is_efficient] < value, axis=1) |
                                              np.all(self._Y[is_efficient] == value, axis=1))
                # This line explicitly sets the current value to efficient. This is crucial because the
                # previous step might have incorrectly marked the current point as "False" if it was equal
                # to other points. This line ensures that the current point is always retained
                # for comparison against subsequent points.
                is_efficient[idx] = True
        self._pareto_front = self._Y[is_efficient]

    def _calculate_reference_point(self):
        self._ref_point = np.max(self._pareto_front, axis=0) + 0.1

    def _minimize_acquisition_function(self):
        """ Minimizes the acquisition function (EHVI) to find the next point to evaluate. """
        best_result = None
        best_value = float("inf")
        lower_bounds = np.array([self._bounds[i][0] for i in range(len(self._bounds))])
        upper_bounds = np.array([self._bounds[i][1] for i in range(len(self._bounds))])
        for _ in range(self._n_optimizer_restarts):
            x0 = np.random.uniform(lower_bounds, upper_bounds, size=(self._X.shape[1],))
            res = minimize(fun=self._ehvi,
                           x0=x0,
                           args=(self._n_objectives, self._model, self._pareto_front, self._ref_point),
                           bounds=self._bounds,
                           method="L-BFGS-B")
            if res.fun < best_value:
                best_result = res
                best_value = res.fun
            print(res)
        self._new_X = best_result.x.reshape(1, -1)

    @staticmethod
    def _ehvi(x, n_objectives, model: list[GaussianProcessRegressor], pareto_front, ref_point=None):

        x = x.reshape(1, -1)
        mean = []
        sigma = []

        for i in range(n_objectives):
            mu, std = model[i].predict(x, return_std=True)
            mean.append(mu)
            sigma.append(std)

        # 1. Define the reference point
        if ref_point is None:
            raise ValueError("A reference point mus tbe defined before optimizing the EHVI.")

        # 2. Calculate the EHVI
        ehvi = 0
        if n_objectives == 1:
            # Expected Improvement (EI) for single objective
            improvement = mean - pareto_front[0]  # pareto_front is just one value
            z = improvement / sigma
            ehvi = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
            ehvi = max(ehvi, 0)

        # TODO: check analytical formula and its implementation
        elif n_objectives == 2:
            # Analytic EHVI for two objectives
            front = pareto_front
            mu1 = mean[0]
            mu2 = mean[1]
            sigma1 = sigma[0]
            sigma2 = sigma[1]
            r1 = ref_point[0]
            r2 = ref_point[1]

            # Standardize Pareto front and reference values
            f1_s = (front[:, 0] - mu1) / sigma1
            f2_s = (front[:, 1] - mu2) / sigma2
            r1_s = (r1 - mu1) / sigma1
            r2_s = (r2 - mu2) / sigma2

            ehvi = 0
            for i in range(len(front)):
                t1 = (r1_s - f1_s) * (r2_s - f2_s) * norm.cdf(f1_s) * norm.cdf(f2_s)
                t2 = sigma1 * (r2 - mu2) * norm.cdf(f2_s) * norm.pdf(f1_s)
                t3 = sigma2 * (r1 - mu1) * norm.pdf(f2_s) * norm.cdf(f1_s)
                t4 = sigma1 * sigma2 * norm.pdf(f1_s) * norm.pdf(f2_s)
                ehvi += t1 + t2 + t3 + t4
            ehvi = np.mean(ehvi)

        else:
            # For more than 2 objectives, use Monte Carlo approximation.
            # This is computationally more intensive.
            n_samples = 10000  # Number of samples
            samples = np.random.multivariate_normal(mean, cov, n_samples)  # (n_samples, n_objectives)
            improvement = np.zeros(n_samples)

            for i in range(n_samples):
                sample = samples[i, :]
                # Calculate the hypervolume improvement for this sample
                dominated = False
                for j in range(pareto_front.shape[0]):
                    if all(sample <= pareto_front[j]):
                        dominated = True
                        break
                if not dominated:  # the sample is not dominated by any point in Pareto front
                    # Calculate the hypervolume improvement contributed by this sample
                    hv_with_sample = 1
                    hv_without_sample = 1

                    # Hypervolume with the sample
                    for k in range(n_objectives):
                        hv_with_sample *= max(ref_point[k], sample[k]) - ref_point[k]

                    # Hypervolume without the sample (Pareto front)
                    for k in range(n_objectives):
                        max_val_k = ref_point[k]
                        for j in range(pareto_front.shape[0]):
                            max_val_k = max(max_val_k, pareto_front[j, k])
                        hv_without_sample *= max_val_k - ref_point[k]
                    improvement[i] = hv_with_sample - hv_without_sample

            ehvi = np.mean(improvement)

        return ehvi

    def _predict_gaussian_process_on_domain(self):
        # Predict mean and standard deviation for each Gaussian process model
        for i in range(self._n_objectives):
            if self._domain.shape[1] > 1:
                # If the domain is N-D, predict on a grid
                grids = [self._domain[:, i] for i in range(self._domain.shape[1])]
                mesh = np.meshgrid(*grids)
                grid = np.stack([m.flatten() for m in mesh], axis=-1)
                mu, sigma = self._model[i].predict(grid, return_std=True)
                self._mu.append(mu.reshape(mesh[0].shape))
                self._sigma.append(sigma.reshape(mesh[0].shape))
            else:
                # If the domain is 1-D
                mu, sigma = self._model[i].predict(self._domain, return_std=True)
                self._mu.append(mu)
                self._sigma.append(sigma)

    """ Plotters """

    def _live_plot(self):
        # Handle single-objective cases
        if self._n_objectives == 1:

            if len(self._bounds) == 1:
                self._f01_1d_plot()
            elif len(self._bounds) == 2:
                self._f01_2d_plot()
            else:
                raise ValueError("Only 1D and 2D plots are supported.")

        # Handle bi-objective cases
        elif self._n_objectives == 2:
            self._mobo_2f0_plot()

        # Handle tri-objective cases
        elif self._n_objectives == 3:
            self._mobo_3f0_plot()

        else:
            raise ValueError("Cannot display pareto front for more than 3-objectives")

    def _f01_1d_plot(self):
        self._initialize_1d_plot()
        self._plot_1d_objective_function()
        self._plot_1d_new_location()
        self._plot_1d_observations()
        self._plot_1d_posterior()
        self._ax.legend()
        plt.pause(0.01)  # Update the plot without blocking

    def _initialize_1d_plot(self):
        if not self._fig or not self._ax:
            self._fig, self._ax = plt.subplots()
            self._ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R} \rightarrow \mathbb{R}$')
            self._ax.set_xlabel(r"$\mathcal{X}$")
            self._ax.set_ylabel(r"$\mathcal{Y}$")
        self._ax.clear()
        self._ax.set_xlim(self._bounds[0][0] * 1.1, self._bounds[0][1] * 1.1)

    def _plot_1d_objective_function(self):
        self._ax.plot(self._domain, self._f0(self._domain), color="black", linestyle="--",
                      label="True objective function", zorder=1)

    def _plot_1d_new_location(self):
        self._ax.vlines(self._new_X, ymin=self._ax.get_ylim()[0], ymax=self._ax.get_ylim()[1],
                        color='red', alpha=0.3, linestyle="--", zorder=3)
        self._ax.scatter(self._new_X, np.zeros_like(self._new_X), marker="x", s=50, color='red',
                         label='New Location', zorder=3)

    def _plot_1d_observations(self):
        self._ax.scatter(self._X, self._Y, marker="o", s=50, color='red', label='Observations', zorder=3)

    def _plot_1d_posterior(self):
        self._ax.plot(self._domain, self._mu, label="Mean", zorder=2)
        for i in range(1, 4):
            self._ax.fill_between(x=self._domain.flatten(),
                                  y1=self._mu - i * self._sigma,
                                  y2=self._mu + i * self._sigma,
                                  alpha=0.2 / i,
                                  color="blue",
                                  label=rf"{i}$\sigma$")

    def _f01_2d_plot(self):
        self._initialize_2d_plot()
        self._plot_2d_objective_function()
        self._plot_2d_new_location()
        self._plot_2d_observations()
        self._plot_2d_posterior()
        self._fig.canvas.draw()
        plt.pause(0.01)  # Update the plot without blocking

    def _initialize_2d_plot(self):
        if not self._fig or not self._ax:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R}^2 \rightarrow \mathbb{R}$')

        self._ax.clear()  # Clear previous plot
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('f(x, y)')
        self._ax.set_xlim(self._bounds[0][0], self._bounds[0][1])
        self._ax.set_ylim(self._bounds[1][0], self._bounds[1][1])

    def _plot_2d_objective_function(self):
        x_grid = np.linspace(self._bounds[0][0], self._bounds[0][1], 100)
        y_grid = np.linspace(self._bounds[1][0], self._bounds[1][1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.c_[X.ravel(), Y.ravel()]
        Z = self._f0(points).reshape(X.shape)
        self._ax.plot_surface(X, Y, Z, lw=0.5, rstride=8, cstride=8, cmap='coolwarm', alpha=0.2)

    def _plot_2d_new_location(self):
        self._ax.scatter(self._new_X[:, 0], self._new_X[:, 1], np.zeros_like(self._new_X[:, 0]),
                         marker="x", s=50, color='red', label='New Location')

    def _plot_2d_observations(self):
        self._ax.scatter(self._X[:, 0], self._X[:, 1], self._Y, c='red', marker='o', s=50, label='Observations',
                         zorder=4)

    def _plot_2d_posterior(self):
        x_grid = self._domain[:, 0]
        y_grid = self._domain[:, 1]
        X, Y = np.meshgrid(x_grid, y_grid)
        for i in range(1, 4):
            self._ax.plot_surface(X,
                                  Y,
                                  (self._mu - i * self._sigma).reshape(X.shape),
                                  lw=0.5, rstride=8, cstride=8,
                                  alpha=0.2, cmap='coolwarm',
                                  zorder=2)
            self._ax.plot_surface(X,
                                  Y,
                                  (self._mu + i * self._sigma).reshape(X.shape),
                                  lw=0.5, rstride=8, cstride=8,
                                  alpha=0.2, cmap='coolwarm',
                                  zorder=3)

    def _mobo_2f0_plot(self):
        # Initialize figure
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot()
        self._ax.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow '
                           r'\mathbb{R}^2$')
        self._ax.set_xlabel('$f_{01}$')
        self._ax.set_ylabel('$f_{02}$')
        # Plot observations
        self._ax.scatter(self._Y[:, 0], self._Y[:, 1], marker="o", s=50, color='red', label='Observations')
        # Plot Pareto Front
        self._ax.scatter(self._pareto_front[:, 0], self._pareto_front[:, 1], marker="x", s=50, color='olive',
                         label='Pareto Front')
        plt.legend()

    def _mobo_3f0_plot(self):
        raise ValueError("Multi-objective plot for 3 objectives is not supported yet")

    """ Validators """

    @staticmethod
    def _validate_XY(X, Y):
        if X is None or Y is None:
            raise ValueError("X and Y cannot be none")
        if not X.shape[0] == Y.shape[0]:
            raise ValueError("The number of samples in X_init and Y_init must match.")

    @staticmethod
    def _validate_f0(n_objectives, f0):
        if f0 is not None and not n_objectives == len(f0):
            raise ValueError("The number of objectives must equal the number of objective functions.")

    @staticmethod
    def _validate_bounds(bounds, n):
        if not isinstance(bounds, list):
            raise ValueError("Bounds must be a list.")
        else:
            for bound in bounds:
                if not len(bound) == 2:
                    raise ValueError("Bounds must be a list of tuples of two numeric values.")

                if not bound[0] < bound[1]:
                    raise ValueError("Lower bound must be smaller than upper bound.")

                if not isinstance(bound[0], (int, float)) and not isinstance(bound[1], (int, float)):
                    raise ValueError("Bounds must be a list of tuples of two numeric values.")

        if not len(n) == len(bounds):
            raise ValueError("The number of samples must be the same length as the number of bounds.")
        else:
            for n_ in n:
                if not isinstance(n_, int):
                    raise ValueError("The number of samples must be a list of integers.")
                if n_ <= 0:
                    raise ValueError("The number of samples must be a positive integer.")

    """ Helpers """

    def _compose_filepath(self, directory, previous=False):
        filename = self._compose_filename(previous)
        filepath = f"{directory}/{filename}"
        return filepath

    def _compose_filename(self, previous=False):
        if previous:
            filename = f'{self._experiment_name} - {self._datetime.strftime("%Y-%m-%d_%H-%M-%S")} - {self._X.shape[0] - 1} samples'
        else:
            filename = f'{self._experiment_name} - {self._datetime.strftime("%Y-%m-%d_%H-%M-%S")} - {self._X.shape[0]} samples'
        return filename

    def _build_domain_from_bounds(self, n=(100,)):
        if len(n) != len(self._bounds):
            raise ValueError("The length of n must match the number of dimensions in bounds.")
        domain = []
        for i in range(len(self._bounds)):
            domain.append(np.linspace(self._bounds[i][0], self._bounds[i][1], n[i]).reshape(-1, 1))
        # self._domain = np.concatenate(domain, axis=1)
        self._domain = domain
