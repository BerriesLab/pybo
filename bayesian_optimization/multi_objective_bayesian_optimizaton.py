import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor


class MultiObjectiveBayesianOptimization:

    def __init__(self, n_objectives: int = 2):

        # Domain attributes
        self._n_objectives = n_objectives
        self._experiment_name = None
        self._datetime = datetime.now()
        self._observation_noise = 1e-12
        self._n_restarts = 100
        self._bounds: list[tuple[float, float]] | None = None
        self._domain: np.ndarray or None = None
        self._acquisition_function: str or None = None
        self._f0: list[callable] = []  # The objective function f0: R^N -> R
        self._kernel = None
        self._X = None  # The X dataset (n samples x N dimensions)
        self._Y = np.zeros((0, n_objectives))  # The Y dataset (n samples x 1 dimensions)
        self._model: list[GaussianProcessRegressor] = []
        self._pareto_front = None
        self._ref_point = None
        self._new_X = None  # The new X location
        self._fig = None
        self._ax = None
        self._mu: list[np.ndarray] = []
        self._sigma: list[np.ndarray] = []
        self._callback = []

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
        self._build_domain_from_bounds()

    def get_bounds(self):
        return self._bounds

    def set_acquisition_function(self, acquisition_function: str):
        supported_functions = ['ei', 'pi', 'lcb', 'ehvi']
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

    def save_attributes_to_disc(self, directory, format="pickle"):
        # Handle base case
        if not self.__dict__:
            raise ValueError("No attributes are set. Cannot export empty object attributes.")

        filepath = self._compose_filepath(directory)
        try:
            if format == "pickle":
                with open(f"{filepath}.dat", 'wb') as file:
                    pickle.dump(self.__dict__, file)
            elif format == "json":
                with open(f"{filepath}.json", 'w') as file:
                    json.dump(self.__dict__, file, default=str)
            else:
                raise ValueError("Unsupported format. Use 'pickle' or 'json'.")
        except IOError:
            print(f"Warning: File '{filepath}' could not be saved. Continuing operation.")
            return

    def import_attributes(self, filename, format="pickle"):

        filename = filename + ".dat"
        if not os.path.exists(filename):
            print(f"Warning: File '{filename}' does not exist. Continuing operation.")
            return

        if format == "pickle":
            with open(filename, 'rb') as file:
                attributes = pickle.load(file)
        elif format == "json":
            import json
            with open(filename, 'r') as file:
                attributes = json.load(file)
        else:
            raise ValueError("Unsupported format. Use 'pickle' or 'json'.")

        for key, value in attributes.items():
            self.__dict__[key] = value

    def save_figure_to_disc(self, directory):
        # Handle base case
        if not self._fig or not self._ax:
            raise ValueError("No plot is available. Cannot export empty plot.")

        filepath = self._compose_filepath(directory)
        try:
            self._fig.savefig(f"{filepath}.png")
        except IOError:
            print(f"Warning: File '{filepath}' could not be saved. Continuing operation.")
            return

    """ Optimizer """

    def optimize(self, live_plot=True):
        self._validate()
        self._predict_gaussian_process()
        self._calculate_pareto_front()
        self._calculate_reference_point()
        self._minimize_acquisition_function()
        if live_plot:
            self._live_plot()

    def _predict_gaussian_process(self):
        # Instantiate a Gaussian process for each objective function
        for i in range(self._n_objectives):
            self._instantiate_a_gaussian_process()
        # Fit all Gaussian processes
        for i in range(self._n_objectives):
            self._model[i].fit(self._X, self._Y[:, i])
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

    def _instantiate_a_gaussian_process(self):
        gpr = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self._observation_noise,  # Noise added to the diagonal of the input matrix
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=self._n_restarts, )
        self._model.append(gpr)

    def _minimize_acquisition_function(self):
        """ Minimizes the acquisition function (EHVI) to find the next point to evaluate. """

        best_result = None
        best_value = float("inf")
        lower_bounds = np.array([self._bounds[i][0] for i in range(len(self._bounds))])
        upper_bounds = np.array([self._bounds[i][0] for i in range(len(self._bounds))])
        for _ in range(self._n_restarts):
            x0 = np.random.uniform(lower_bounds, upper_bounds, size=(self._X.shape[1],))
            res = minimize(fun=self._ehvi,
                           x0=x0,
                           args=(self._model, self._pareto_front, self._ref_point),
                           bounds=self._bounds,
                           method="L-BFGS-B")
            if res.fun < best_value:
                best_result = res
                best_value = res.fun

        self._new_X = best_result.x.reshape(1, -1)

    # TODO: write wrapper and check data type and shapes while running the example.
    def _ehvi_wrapper(self):
        self._ehvi(n_objectives=self._n_objectives,
                   mean=self._mu,
                   sigma=self._sigma,
                   pareto_front=self._pareto_front,
                   ref_point=self._ref_point)

    @staticmethod
    def _ehvi(x, n_objectives, mean, sigma, pareto_front, ref_point=None):
        """
        Calculates the Expected Hypervolume Improvement (EHVI) for a given point `x`.

        Args:
            x (np.ndarray): The point at which to evaluate EHVI.  Shape: (1, n_features).
            model: A trained Gaussian Process model (or other surrogate model)
                that can predict mean and covariance/variance.
            pareto_front (np.ndarray): The current Pareto front.
                Shape: (n_points_on_front, n_objectives).
            ref_point (np.ndarray, optional): The reference point for hypervolume
                calculation.  Shape: (n_objectives,).  If None, it is computed
                as slightly worse than the worst point in the Pareto front.

        Returns:
            float: The EHVI value at the point `x`.
        """

        # mean, cov = model.predict(x, return_cov=True)
        # mean = mean.flatten()
        # sigma = np.sqrt(np.diag(cov))  # Get standard deviations

        # 1. Define the reference point
        if ref_point is None:
            #  This is a *bad* way to calculate the reference point for real applications.
            #  It is inefficient and can lead to a poorly defined hypervolume.
            #  The reference point should be *outside* the Pareto front.
            min_values = np.min(pareto_front, axis=0)
            ref_point = min_values - 1  # Shift slightly below the minimum
            print(
                "WARNING: Reference point was calculated within the EHVI function. This is inefficient and can lead to suboptimal results.  Define the reference point *outside* this function and pass it in.")
        # 2. Calculate the EHVI
        ehvi = 0
        if n_objectives == 1:
            # Expected Improvement (EI) for single objective
            improvement = mean - pareto_front[0]  # pareto_front is just one value
            z = improvement / sigma
            ehvi = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
            ehvi = max(ehvi, 0)
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
            front_s = np.copy(front)
            front_s[:, 0] = (front_s[:, 0] - mu1) / sigma1
            front_s[:, 1] = (front_s[:, 1] - mu2) / sigma2
            r1_s = (r1 - mu1) / sigma1
            r2_s = (r2 - mu2) / sigma2

            ehvi = 0
            for i in range(len(front)):
                u1s = front_s[i, 0]
                u2s = front_s[i, 1]
                c1 = (r1_s - u1s) * (r2_s - u2s)
                c2 = sigma1 * sigma2
                t1 = c1 * norm.cdf(u1s) * norm.cdf(u2s)
                t2 = (r2 - mu2 - sigma2 * u2s) * sigma1 * norm.cdf(u1s) * norm.pdf(u2s)
                t3 = (r1 - mu1 - sigma1 * u1s) * sigma2 * norm.pdf(u1s) * norm.cdf(u2s)
                t4 = c2 * norm.pdf(u1s) * norm.pdf(u2s)
                ehvi += t1 + t2 + t3 + t4
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
        self._ref_point = np.min(self._pareto_front, axis=0)

    """ Plotters """

    def _live_plot(self):
        if len(self._bounds) == 1:
            self._live_1d_plot()
        elif len(self._bounds) == 2:
            self._live_2d_plot()
        else:
            raise ValueError("Only 1D and 2D plots are supported.")

    def _live_1d_plot(self):
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

    def _live_2d_plot(self):
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
            raise ValueError("The number of dimensions in bounds and X must match.")

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

    def _build_domain_from_bounds(self):
        domain = []
        for i in range(len(self._bounds)):
            domain.append(np.linspace(self._bounds[i][0], self._bounds[i][1], 100).reshape(-1, 1))
        self._domain = np.concatenate(domain, axis=1)