import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv

from kernel_functions import rbf


class BayesianOptimization:

    def __init__(self):
        # Set random seed to ensure reproducibility
        np.random.seed(8)

        # Settings number of input locations which approximates a function
        n = 100


# Define domain vector
X_test = np.linspace(-5, 5, n).reshape(-1, 1)
# Define mean as a column vector and validate its shape
mu = np.zeros(X_test.shape)
# Calculate pairwise distance and add regularization for numerical stability
K = rbf(X_test, X_test) + 1e-10 * np.eye(n)


def sample(mu, cov, size):
    samples = np.random.multivariate_normal(mean=mu.ravel(), cov=cov, size=size)
    return samples


def plot_gp(mu, cov, x, x_train=None, y_train=None, samples=None):
    x = x.ravel()
    mu = mu.ravel()

    # 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(x, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(x, mu, label='Mean')
    plt.plot(x, mu + uncertainty, label=r"$\mu + 1.96 \sigma$")
    plt.plot(x, mu - uncertainty, label=r"$\mu - 1.96 \sigma$")

    for i, sample in enumerate(samples):
        plt.plot(x, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    # Plot observations if available

    if x_train is not None and y_train is not None:
        plt.plot(x_train, y_train, 'rx')
    plt.legend()


def update_posterior(X_s, X_train, Y_train, l=1.0, sigma=1.0, sigma_y=1e-8):
    """
    :parameter X: new input location (n x d), where n is the number of samples and the d their dimensions
    :parameter X_train: training locations (m x d), where m is the number of samples and the d their dimensions
    :parameter Y_train: training outputs (m x 1)
    :parameter sigma_y: observation noise
    """
    # Calculate the covariance matrix for the observed inputs.
    K = rbf(X_train, X_train, l, sigma) + sigma_y ** 2 * np.eye(len(X_train))
    # Calculate the cross-variance between previously observed and new inputs.
    K_s = rbf(X_train, X_s, l, sigma)
    # calculate the covariance matrix for the new inputs
    K_ss = rbf(X_s, X_s)
    # Calculate inverse of covariance matrix
    K_inv = inv(K)

    # Calculate the posterior mean vector and covariance matrix
    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
Y_train = np.cos(X_train)
mu_s, cov_s = update_posterior(X_test, X_train, Y_train)
samples = sample(mu_s.ravel(), cov_s, 5)
plot_gp(mu_s, cov_s, X_test, X_train, Y_train, samples=samples)
plt.show()