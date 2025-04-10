import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor


class BayesianOptimization:

    def __init__(self):

        # Domain attributes
        self._observation_noise = None
        self._callback = None
        self._n_restarts = None
        self._bounds: list[tuple[float, float]] | None = None
        self._n_iter = None
        self._acquisition_function: str or None = None
        self._f0 = None  # The objective function f0: R^N -> R
        self._kernel = None
        self._X = None  # The initial dataset X (n samples x N dimensions)
        self._Y = None  # The initial dataset Y (n samples x 1 dimensions)
        self._n = None  # Number of initial points used by the solver to calculate the first posterior

    """Setters and getters of object attributes."""

    def set_initial_dataset(self, X: np.ndarray, Y: np.ndarray):
        """
        Sets the training data samples.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X_train must be a numpy array.")

        if not isinstance(Y, np.ndarray):
            raise ValueError("Y_train must be a numpy array.")

        if X.shape[0] != Y.shape[0]:
            raise ValueError("The number of samples in X_train and Y_train must match.")

        self._X = X
        self._Y = Y

    def get_initial_samples(self):
        return self._X, self._Y

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

    def set_number_of_objective_function_calls(self, n_calls: int):
        if not isinstance(n_calls, int) or n_calls <= 0:
            raise ValueError("The number of objective function calls must be a positive integer.")
        self._n_iter = n_calls

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

    """ I/O """

    def import_data(self, filename):
        data = pd.read_csv(filename)
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values
        self.set_initial_dataset(X, Y)

    def export_data(self, filename):
        if self._X is None or self._Y is None:
            raise ValueError("No data available to export. Ensure that _X and _Y are set.")

        columns = ["i_0", "i_max", "t_on", "wear"]
        data = np.hstack((self._X, self._Y.reshape(-1, 1)))
        data = pd.DataFrame(data, columns=columns)
        data.to_csv(filename, index=False)

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
        return np.concatenate(domain, axis=1)

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
        domain = self._build_domain_from_bounds()
        gpr = self._instantiate_a_gaussian_process()

        fig = plt.figure()
        plt.show(block=False)

        j = 0
        while j < self._n_iter:
            # Define domain and fit the Gaussian process
            gpr.fit(self._X, self._Y)

            # Predict mean and standard deviation
            # If the domain is N-dimensional, we need to predict on a grid. So, let's build the grid:
            if domain.shape[1] > 1:
                x, y = np.meshgrid(domain[:, 0], domain[:, 1])
                grid = np.stack([x.flatten(), y.flatten()], axis=1)
                mu, sigma = gpr.predict(grid, return_std=True)
                mu = mu.reshape(100, 100)
                sigma = sigma.reshape(100, 100)
            else:
                mu, sigma = gpr.predict(domain, return_std=True)

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

            print(f"New best result: {best_result}")

            # Plot
            if self._X.shape[1] == 1:
                self._plot_1d(fig, domain, mu, sigma)
            elif self._X.shape[1] == 2:
                self._plot_2d(fig, domain, mu, sigma)

            # Add new observation
            self._X = np.concatenate((self._X, best_result.x.reshape(1, -1)), axis=0)
            # TODO: if 1d reshape(-1, 1)
            self._Y = np.concatenate((self._Y, self._f0(best_result.x.reshape(1, -1))), axis=0)

            j += 1

        plt.show()

    """ Plotters """

    # TODO: fix plots and use these methods in optimize
    def _plot_1d(self, fig, domain, mu, sigma):
        plt.clf()
        plt.show(block=False)
        plt.xlim(self._bounds[0][0] * 1.1, self._bounds[0][1] * 1.1)
        plt.title(r'Bayesian Optimization for $f_0:\mathbb{R} \rightarrow \mathbb{R}$')
        plt.xlabel("Input (X)")
        plt.ylabel("Output (f(X))")
        # Plot observations
        plt.scatter(self._X, self._Y, marker="x", s=100, color='red', label='Observations')
        # Plot the true objective function (if available)
        plt.plot(domain, self._f0(domain), color="black", linestyle="--", label="True objective function")
        # Plot mean and uncertainty
        plt.plot(domain, mu, label="Mean")
        for i in range(1, 4):
            plt.fill_between(x=domain.flatten(),
                             y1=mu - i * sigma,
                             y2=mu + i * sigma,
                             alpha=0.2 / i,
                             color="blue",
                             label=rf"{i}$\sigma$")
        plt.pause(1)

    def _plot_2d(self, fig, domain, mu, sigma):
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        # Add labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R}^2 \rightarrow \mathbb{R}$')
        ax.set_xlim(self._bounds[0][0], self._bounds[0][1])
        ax.set_ylim(self._bounds[1][0], self._bounds[1][1])

        # Plot observations
        ax.scatter(self._X[:, 0], self._X[:, 1], self._Y, c='red', marker='X', s=100, label='Observations')
        # Plot the true objective function (if available)
        x_grid = domain[:, 0]
        y_grid = domain[:, 1]
        X, Y = np.meshgrid(x_grid, y_grid)
        points = np.c_[X.ravel(), Y.ravel()]
        Z = self._f0(points).reshape(X.shape)
        ax.plot_surface(X, Y, Z, lw=0.5, rstride=8, cstride=8, alpha=0.5, cmap='Greys',
                        label="True objective function")
        # Plot projections of the contours for each dimension.  By choosing offsets
        # that match the appropriate axes limits, the projected contours will sit on
        # the 'walls' of the graph.
        # ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='x', offset=self._bounds[0][0] * 1.1, cmap='coolwarm')
        # ax.contour(X, Y, Z, zdir='y', offset=self._bounds[1][1] * 1.1, cmap='coolwarm')

        # Plot mean and uncertainty
        for i in range(1, 4):
            ax.plot_surface(X, Y, mu.reshape(X.shape) - i * sigma.reshape(X.shape), lw=0.5, rstride=8, cstride=8,
                            alpha=0.2, cmap='coolwarm',
                            label="True objective function")
            ax.plot_surface(X, Y, mu.reshape(X.shape) + i * sigma.reshape(X.shape), lw=0.5, rstride=8, cstride=8,
                            alpha=0.2, cmap='coolwarm',
                            label="True objective function")
        plt.pause(1)

    """ Validators """

    def _validate(self):
        self._validate_bounds()
        self._validate_number_of_iterations()
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

    def _validate_number_of_iterations(self):
        if self._n_iter is None:
            raise ValueError("Please specify the number of iterations.")

    def _validate_bounds(self):
        # Handle base case
        if self._bounds is None:
            raise ValueError("Bounds for the input domain must be specified.")
        if isinstance(self._X, np.ndarray) and len(self._bounds) != self._X.shape[1]:
            raise ValueError("The number of dimensions in bounds and X_init must match.")