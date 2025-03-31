import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def multivariate_gaussian_likelihood(X, mu, Sigma):
    """
    Compute the likelihood and log-likelihood for a multivariate normal distribution.

    Parameters:
    X (numpy array): Observed data points.
    mu (numpy array): Mean vector.
    Sigma (numpy array): Covariance matrix.

    Returns:
    likelihood (float): Probability density value.
    log_likelihood (float): Log of the probability density.
    """
    # Compute likelihood using the multivariate normal distribution
    mvn = multivariate_normal(mean=mu, cov=Sigma)
    likelihood = np.prod(mvn.pdf(X))  # Likelihood is the product of individual PDFs

    # Compute log-likelihood
    log_likelihood = np.sum(mvn.logpdf(X))  # Log-likelihood is the sum of log PDFs

    return likelihood, log_likelihood


def maximum_likelihood_estimates(X):
    """
    Compute the Maximum Likelihood Estimates (MLE) for the mean and covariance matrix of a multivariate normal distribution.

    Parameters:
    X (numpy array): Observed data points.

    Returns:
    mu_hat (numpy array): MLE of the mean vector.
    Sigma_hat (numpy array): MLE of the covariance matrix.
    """
    # MLE for the mean is the sample mean
    mu_hat = np.mean(X, axis=0)

    # MLE for the covariance matrix is the sample covariance matrix
    Sigma_hat = np.cov(X, rowvar=False)

    return mu_hat, Sigma_hat


# Generate random data from a multivariate normal distribution
mu_true = np.array([3.0, 4.0])  # True mean vector
Sigma_true = np.array([[1.0, 0.8], [0.8, 1.0]])  # True covariance matrix
n_samples = 500  # Number of samples to generate

# Generate random samples
X_random = np.random.multivariate_normal(mu_true, Sigma_true, n_samples)

# Maximum Likelihood Estimates (MLE) for mean and covariance matrix
mu_hat, Sigma_hat = maximum_likelihood_estimates(X_random)

# Compute likelihood and log-likelihood with MLE
likelihood, log_likelihood = multivariate_gaussian_likelihood(X_random, mu_hat, Sigma_hat)

# Print results
print(f"MLE for mean (mu): {mu_hat}")
print(f"MLE for covariance matrix (Sigma): \n{Sigma_hat}")
print(f"Likelihood: {likelihood}")
print(f"Log-Likelihood: {log_likelihood}")

# Plotting the multivariate normal distribution with the MLE parameters and the random data
x_min, x_max = X_random[:, 0].min() - 1, X_random[:, 0].max() + 1
y_min, y_max = X_random[:, 1].min() - 1, X_random[:, 1].max() + 1

x_vals, y_vals = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
pos = np.dstack((x_vals, y_vals))

# Create the multivariate normal distribution with the MLE parameters
mvn = multivariate_normal(mean=mu_hat, cov=Sigma_hat)
z_vals = mvn.pdf(pos)

# Plot the contour plot of the Gaussian distribution
plt.contour(x_vals, y_vals, z_vals, cmap='Blues')

# Plot the observed data points
plt.scatter(X_random[:, 0], X_random[:, 1], color='red', s=5, label='Random Data', alpha=0.6)

# Adding labels and title
plt.title("Multivariate Normal Distribution with Random Data")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()

# Display the plot
plt.show()
