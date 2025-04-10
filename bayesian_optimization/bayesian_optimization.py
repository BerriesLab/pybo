import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor
from skopt.space import Real

from objective_functions import test_f0_1d


class BayesianOptimization:

    def __init__(self):

        # Domain attributes
        self._observation_noise = None
        self._callback = None
        self._n_restarts = None
        self._bounds: list[tuple[float, float]] | None = None
        self._n_iter = None
        self._acquisition_function: str or None = None
        self._f0 = None
        self._kernel = None
        self._X_init = None  # The initial dataset X (n samples x N dimensions)
        self._Y_init = None  # The initial dataset Y (n samples x 1 dimensions)
        self._n_init = None  # Number of initial points used by the solver to calculate the first posterior
        self._X = None  # Current dataset X
        self._Y = None  # Current dataset Y

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

        self._X_init = X
        self._Y_init = Y

    def get_initial_samples(self):
        return self._X_init, self._Y_init

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

    def set_kernel(self, kernel):
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

    """ Optimizer """

    def _instantiate_a_gaussian_process(self):
        # Define the regressor, i.e. the kernel and
        gpr = GaussianProcessRegressor(
            kernel=None,  # self._kernel,  # If None, uses 1.0 * RBF(1.0) as default
            alpha=self._observation_noise,  # Noise added to the diagonal of the input matrix
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=self._n_restarts)
        return gpr

    def _build_domain_from_bounds(self):
        domain = []
        for i in range(len(self._bounds)):
            domain.append(np.linspace(self._bounds[i][0], self._bounds[i][1], 1000).reshape(-1, 1))
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

    # TODO: the acquisition function should return a scalar
    def optimize(self):
        # Validate object attributes
        self._validate()

        self._X = self._X_init
        self._Y = self._Y_init

        # Consistent Gaussian process initialization
        gpr = self._instantiate_a_gaussian_process()

        # Plot initial data
        plt.show(block=False)
        plt.xlim(self._bounds[0][0], self._bounds[0][1])
        plt.title("Bayesian Optimization: Surrogate Function")
        plt.xlabel("Input (X)")
        plt.ylabel("Output (f(X))")
        plt.scatter(self._X, self._Y, marker="x", s=100, color='red', label='Initial Samples')
        plt.pause(0.2)

        j = 0
        while j < 20:
            # Define domain and fit the Gaussian process
            domain = self._build_domain_from_bounds()
            gpr.fit(self._X, self._Y)

            # Predict mean and standard deviation
            mu, sigma = gpr.predict(domain, return_std=True)

            # Plot mean and uncertainty
            plt.plot(domain, mu, label="Mean")
            for i in range(1, 4):
                plt.fill_between(domain.flatten(), mu - i * sigma, mu + i * sigma,
                                 alpha=0.2 / i, color="blue", label=rf"{i}$\sigma$")
            plt.pause(1)

            # Maximize acquisition function (Expected Improvement)
            y_best = max(self._Y)
            best_result = None
            best_value = float("inf")

            for _ in range(self._n_restarts):
                x0 = np.random.uniform(self._bounds[0][0], self._bounds[0][1], size=(1))
                res = minimize(fun=self._expected_improvement,
                               x0=x0,
                               args=(gpr, y_best),
                               bounds=self._bounds,
                               method="L-BFGS-B")
                if res.fun < best_value:
                    best_result = res
                    best_value = res.fun

            print(f"New best result: {best_result}")

            # Visualize new observation
            plt.scatter(best_result.x, self._f0(best_result.x.reshape(-1, 1)), marker="x", s=100, color='red')
            plt.pause(1)

            # Add new observation
            self._X = np.concatenate((self._X, best_result.x.reshape(1, -1)), axis=0)
            self._Y = np.concatenate((self._Y, self._f0(best_result.x.reshape(-1, 1))), axis=0)

            j += 1

        plt.show()

    """ Plotters """

    def plot_live(self, result):

        plt.clf()
        plt.xlim(self._bounds[0][0], self._bounds[0][1])
        X = np.linspace(self._bounds[0][0], self._bounds[0][1], 1000).reshape(-1, 1)
        plt.title("Bayesian Optimization: Surrogate Function")
        plt.xlabel("Input (X)")
        plt.ylabel("Output (f(X))")

        self.plot_objective_function(X)
        self.plot_initial_observations(X)
        # self.plot_prediction(X, result)
        # self.plot_surrogate_function(X, result)

        plt.pause(0.2)

    def plot_objective_function(self, X):
        """This is possible only if the objective is defined"""
        plt.plot(X, self._f0(X), label="Objective Function")

    def plot_initial_observations(self, X):
        if self._X_init is not None:
            plt.scatter(self._X_init, self._Y_init, marker="x", s=100, color='red', label='Initial Samples')

    def plot_prediction(self, X, results):
        plt.axvline(x=results.x, linestyle="--", color="black", label="Optimal Point")

        # plt.scatter(results.x, marker="o", s=100, color='blue', label='Optimal Point')

    # def plot_surrogate_function(self, results):
    #
    #     # Predict with the Gaussian Process model from the results
    #     model = results.models[-1]
    #     mean, std = model.predict(X, return_std=True)
    #
    #     # Plot the surrogate function (mean)
    #     plt.plot(X, mean, label="Surrogate Mean")
    #     plt.fill_between(
    #         X.flatten(),
    #         mean - std,
    #         mean + std,
    #         alpha=0.2,
    #         label="Surrogate +/− 1 std"
    #     )
    #
    #     # Plot the optimization samples
    #     sampled_X = np.array(results.x_iters)
    #     sampled_Y = np.array(results.func_vals)
    #     plt.scatter(sampled_X, sampled_Y, marker="x", color='red', label='Observed Samples')
    #
    #     # Plot optimal point
    #
    #     plt.legend()
    #     plt.pause(0.2)

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
        if self._X_init is None and self._Y_init is None:
            print("No initial dataset provided. No initial points will be used.")
        if (self._X_init is not None and self._Y_init is not None and
                self._X_init.shape[0] != self._Y_init.shape[0]):
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
        if isinstance(self._X_init, np.ndarray) and len(self._bounds) != self._X_init.shape[1]:
            raise ValueError("The number of dimensions in bounds and X_init must match.")


# 1D - X is a vector n x 1
# ( [ x^(1)_1 ], y^(1) )
# ( [ x^(2)_1 ], y^(2) )
# ...
# ( [ x^(n)_1 ], y^(n) )


# 2D - X is a matrix n x 2
# ( [ x^(1)_1, x^(1)_2 ], y^(1) )
# ( [ x^(2)_1, x^(2)_2 ], y^(2) )
# ...
# ( [ x^(n)_1, x^(n)_2 ], y^(n) )


# 3D - X is a matrix n x 3
# ( [ x^(1)_1, x^(1)_2, x^(1)_3 ], y^(1) )
# ( [ x^(2)_1, x^(2)_2, x^(2)_3 ], y^(2) )
# ...
# ( [ x^(n)_1, x^(n)_2, x^(n)_3 ], y^(n) )


X_init_1d = np.array([[-4], [-2], [0], [+2], [+4]])
Y_init_1d = np.array([[-1], [-2], [0], [2], [1]]).reshape(-1)
Y_init_1d = test_f0_1d(X_init_1d)
# Y_init_1d = test_f0_1d(X_init_1d).reshape(-1, 1)
bounds_1d = [(-5, 5)]

# X_init_3d = np.array([[-4, -3, -2], [-2, +0, +2]])
# Y_init_3d = np.sin(X_init_3d[:, 0]) + 2 * np.cos(X_init_3d[:, 1]) + np.tanh(X_init_3d[:, 2])

search_space = [
    Real(5, 10, name="i_0"),
    Real(80, 100, name="i_max"),
    Real(90, 120, name="t_on")
]

opt = BayesianOptimization()
opt.set_initial_dataset(X_init_1d, Y_init_1d)
opt.set_observation_noise(1)
opt.set_bounds(bounds_1d)
opt.set_acquisition_function("EI")
opt.set_kernel(None)
opt.set_objective_function(test_f0_1d)
opt.set_number_of_objective_function_calls(10)
opt.set_number_of_optimizer_restarts(50)
opt.set_callback(None)
opt.set_callback(opt.plot_live)
print(opt.get_attributes())
opt.optimize()