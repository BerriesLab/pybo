import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor


class BayesianOptimization:

    def __init__(self):

        # Domain attributes
        self._experiment_name = None
        self._datetime = datetime.now()
        self._observation_noise = None
        self._n_restarts = 100
        self._bounds: list[tuple[float, float]] | None = None
        self._domain: np.ndarray or None = None
        self._acquisition_function: str or None = None
        self._f0 = None  # The objective function f0: R^N -> R
        self._kernel = None
        self._X = None  # The X dataset (n samples x N dimensions)
        self._Y = None  # The Y dataset (n samples x 1 dimensions)
        self._new_X = None  # The new X location
        self._fig = None
        self._mu: np.ndarray or None = None
        self._sigma: np.ndarray or None = None

    """ Setters and getters """

    def set_experiment_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        self._experiment_name = name

    def get_experiment_name(self):
        return self._experiment_name

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

    def set_bounds(self, bounds):
        if not isinstance(bounds, list) or not all(
                isinstance(b, tuple) and len(b) == 2 and all(isinstance(v, (int, float)) for v in b) for b in
                bounds):
            raise ValueError("Bounds must be a list of tuples of two numeric values.")
        self._bounds = bounds

    def get_bounds(self):
        return self._bounds

    def set_acquisition_function(self, acquisition_function: str):
        supported_functions = ['EI', 'PI', 'LCB']
        if acquisition_function not in supported_functions:
            raise ValueError(f"Invalid acquisition function. Supported values are {supported_functions}.")
        self._acquisition_function = acquisition_function

    def get_acquisition_function(self):
        return self._acquisition_function

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
        self._n_restarts = n_restarts

    def get_number_of_optimizer_restarts(self):
        return self._n_restarts

    def set_number_of_iterations(self, n_iter: int):
        if not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")
        self._n_iter = n_iter

    def get_number_of_iterations(self):
        return self._n_iter

    def get_attributes(self):
        attributes = {}
        for attr, value in self.__dict__.items():
            if callable(value):
                attributes[attr] = value.__name__ if hasattr(value, '__name__') else "Unnamed Callable"
            else:
                attributes[attr] = value
        return attributes

    def set_callback(self, callback):
        if callback is None:
            self._callback = None
            return

        if isinstance(callback, list):
            if not all(callable(callback[i]) for i in range(len(callback))):
                raise ValueError("All elements in the callback list must be callable.")
        self._callback = callback

    def get_callback(self):
        return self._callback

    def set_kernel(self, kernel: str):
        self._kernel = kernel

    def get_kernel(self):
        return self._kernel

    def set_observation_noise(self, noise):
        if noise is None or noise == 0:
            self._observation_noise = 1e-10
            return
        self._observation_noise = noise

    def get_observation_noise(self):
        return self._observation_noise

    def get_new_X(self):
        return self._new_X

    """ I/O """

    def import_data(self, filename):
        """ Assumes the data is in the following format: n x N+1, where the last column is Y."""
        data = pd.read_csv(filename)
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values
        self.set_dataset_X(X)
        self.set_dataset_Y(Y)

    def export_data(self, filename):
        if self._X is None or self._Y is None:
            raise ValueError("No data available to export. Ensure that _X and _Y are set.")

        columns = ["i_0", "i_max", "t_on", "wear"]
        data = np.hstack((self._X, self._Y.reshape(-1, 1)))
        data = pd.DataFrame(data, columns=columns)
        data.to_csv(filename, index=False)

    def export_attributes(self, filename):
        if not self.__dict__:
            raise ValueError("No attributes are set. Cannot export empty object attributes.")
        with open(filename, 'wb') as file:
            pickle.dump(self.__dict__, file)

    def import_attributes(self, filename):
        with open(filename, 'rb') as file:
            attributes = pickle.load(file)
        for key, value in attributes.items():
            self.__dict__[key] = value

    """ Optimizer """

    def _instantiate_a_gaussian_process(self):
        # Define the regressor, i.e. the kernel and
        gpr = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self._observation_noise,  # Noise added to the diagonal of the input matrix
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=self._n_restarts)
        return gpr

    def _build_domain_from_bounds(self):
        domain = []
        for i in range(len(self._bounds)):
            domain.append(np.linspace(self._bounds[i][0], self._bounds[i][1], 100).reshape(-1, 1))
        self._domain = np.concatenate(domain, axis=1)

    @staticmethod
    def _expected_improvement(x, gpr, y_best):
        # x is an N x 1 vector, where N is the number of dimensions, or features

        # Reshape
        x = x.reshape(1, -1)

        mu, sigma = gpr.predict(x, return_std=True)
        sigma = np.maximum(sigma, 1e-8)  # avoid division by zero
        z = (mu - y_best) / sigma
        ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei

    def optimize(self):
        self._validate()

        # Define domain and fit the Gaussian process
        self._build_domain_from_bounds()
        gpr = self._instantiate_a_gaussian_process()
        gpr.fit(self._X, self._Y)

        # Predict mean and standard deviation
        # If the domain is N-dimensional, we need to predict on a grid. So, let's build the grid:
        if self._domain.shape[1] > 1:
            x, y = np.meshgrid(self._domain[:, 0], self._domain[:, 1])
            grid = np.stack([x.flatten(), y.flatten()], axis=1)
            mu, sigma = gpr.predict(grid, return_std=True)
            self._mu = mu.reshape(100, 100)
            self._sigma = sigma.reshape(100, 100)
        else:
            self._mu, self._sigma = gpr.predict(self._domain, return_std=True)

        # Maximize acquisition function (Expected Improvement)
        y_best = max(self._Y)
        best_result = None
        best_value = float("inf")
        for _ in range(self._n_restarts):
            x0 = np.random.uniform(self._bounds[0][0], self._bounds[0][1], size=(self._X.shape[1],))
            res = minimize(fun=self._expected_improvement,
                           x0=x0,
                           args=(gpr, y_best),
                           bounds=self._bounds,
                           method="L-BFGS-B")
            if res.fun < best_value:
                best_result = res
                best_value = res.fun
        # Store best result
        self._new_X = best_result.x.reshape(1, -1)

        self._init_1d_plot()
        # self._plot_ground_truth()
        # self._plot_observations()
        # self._plot_posterior()
        # self._plot_new_location()
        # plt.legend()
        plt.show()

    """ Plotters """

    def _init_plot(self):
        if len(self._bounds) == 1:
            self._init_1d_plot()
        if len(self._bounds) == 2:
            self._init_2d_plot()

    def _plot_ground_truth(self):
        if self._f0 is None:
            return
        if len(self._bounds) == 1:
            self._plot_1d_objective_function()
        if len(self._bounds) == 2:
            self._plot_2d_objective_function()

    def _plot_new_location(self):
        if len(self._bounds) == 1:
            self._plot_1d_new_location()
        if len(self._bounds) == 2:
            self._plot_2d_new_location()

    def _plot_observations(self):
        if len(self._bounds) == 1:
            self._plot_1d_observations()
        if len(self._bounds) == 2:
            self._plot_2d_observations()

    def _plot_posterior(self):
        if len(self._bounds) == 1:
            self._plot_1d_posterior()
        if len(self._bounds) == 2:
            self._plot_2d_posterior()

    def _init_1d_plot(self):
        self._fig = plt.figure()
        plt.xlim(self._bounds[0][0] * 1.1, self._bounds[0][1] * 1.1)
        plt.title(r'Bayesian Optimization for $f_0:\mathbb{R} \rightarrow \mathbb{R}$')
        plt.xlabel(r"$\mathcal{X}$")
        plt.ylabel(r"$\mathcal{Y}$")

    def _init_2d_plot(self):
        self._fig = plt.figure()
        plt.show(block=False)
        plt.clf()
        ax = self._fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R}^2 \rightarrow \mathbb{R}$')
        ax.set_xlim(self._bounds[0][0], self._bounds[0][1])
        ax.set_ylim(self._bounds[1][0], self._bounds[1][1])

    def _plot_1d_objective_function(self):
        plt.plot(self._domain, self._f0(self._domain), color="black", linestyle="--", label="True objective function",
                 zorder=1)

    def _plot_2d_objective_function(self):
        x_grid = self._domain[:, 0]
        y_grid = self._domain[:, 1]
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.c_[X.ravel(), Y.ravel()]
        Z = self._f0(points).reshape(X.shape)
        self._fig.axes[0].plot_surface(X, Y, Z, lw=0.5, rstride=8, cstride=8, alpha=0.5, cmap='Greys',
                                       label="True objective function", zorder=1)

    def _plot_1d_new_location(self):
        plt.vlines(self._new_X, ymin=plt.ylim()[0], ymax=plt.ylim()[1], color='red', alpha=0.3, linestyle="--",
                   zorder=3)
        plt.scatter(self._new_X, np.zeros_like(self._new_X), marker="x", s=50, color='red', label='New Location',
                    zorder=3)

    def _plot_2d_new_location(self):
        plt.scatter(self._new_X[:, 0], self._new_X[:, 1], marker="x", s=50, color='red', label='New Location', )

    def _plot_1d_observations(self):
        plt.scatter(self._X, self._Y, marker="o", s=50, color='red', label='Observations', zorder=3)

    def _plot_2d_observations(self):
        self._fig.axes[0].scatter(self._X[:, 0], self._X[:, 1], self._Y, c='red', marker='o', s=100,
                                  label='Observations', zorder=4)

    def _plot_1d_posterior(self):
        plt.plot(self._domain, self._mu, label="Mean", zorder=2)
        for i in range(1, 4):
            plt.fill_between(x=self._domain.flatten(),
                             y1=self._mu - i * self._sigma,
                             y2=self._mu + i * self._sigma,
                             alpha=0.2 / i,
                             color="blue",
                             label=rf"{i}$\sigma$")

    def _plot_2d_posterior(self):
        for i in range(1, 4):
            self._fig.axes[0].plot_surface(self._X, self._Y,
                                           self._mu.reshape(self._X.shape) - i * self._sigma.reshape(self._X.shape),
                                           lw=0.5, rstride=8, cstride=8,
                                           alpha=0.2, cmap='coolwarm',
                                           label="True objective function", zorder=2)
            self._fig.axes[0].plot_surface(self._X, self._Y,
                                           self._mu.reshape(self._X.shape) + i * self._sigma.reshape(self._X.shape),
                                           lw=0.5, rstride=8, cstride=8,
                                           alpha=0.2, cmap='coolwarm',
                                           label="True objective function", zorder=3)

    """ Validators """

    def _validate(self):
        self._validate_bounds()
        self._validate_acquisition_function()
        self._validate_initial_dataset()
        self._validate_objective_function()

    def _validate_objective_function(self):
        if self._f0 is None:
            raise ValueError("Please specify the objective function.")

    def _validate_initial_dataset(self):
        if self._X is None and self._Y is None:
            print("No initial dataset provided. No initial points will be used.")
        if (self._X is not None and self._Y is not None and
                self._X.shape[0] != self._Y.shape[0]):
            raise ValueError("The number of samples in X_init and Y_init must match.")

    def _validate_acquisition_function(self):
        if self._acquisition_function is None:
            raise ValueError("Please specify the acquisition function.")

    def _validate_bounds(self):
        # Handle base case
        if self._bounds is None:
            raise ValueError("Bounds for the input domain must be specified.")
        if isinstance(self._X, np.ndarray) and len(self._bounds) != self._X.shape[1]:
            raise ValueError("The number of dimensions in bounds and X_init must match.")