import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def univariate_normal_likelihood(X, mu, sigma_squared):
    """
    Compute the likelihood and log-likelihood for a univariate normal distribution.

    Parameters:
    X (numpy array): Observed data points.
    mu (float): Mean of the distribution.
    sigma_squared (float): Variance of the distribution (sigma squared).

    Returns:
    likelihood (float): Probability density value.
    log_likelihood (float): Log of the probability density.
    """
    n = len(X)

    # Compute likelihood
    likelihood = np.prod(norm.pdf(X, mu, np.sqrt(sigma_squared)))

    # Compute log-likelihood
    log_likelihood = -n / 2 * np.log(2 * np.pi * sigma_squared) - np.sum((X - mu) ** 2) / (2 * sigma_squared)

    return likelihood, log_likelihood


def maximum_likelihood_estimates(X):
    """
    Compute the Maximum Likelihood Estimates (MLE) for the mean and variance of a univariate normal distribution.

    Parameters:
    X (numpy array): Observed data points.

    Returns:
    mu_hat (float): MLE of the mean.
    sigma_squared_hat (float): MLE of the variance.
    """
    n = len(X)

    # MLE for the mean is the sample mean
    mu_hat = np.mean(X)

    # MLE for the variance is the sample variance
    sigma_squared_hat = np.var(X)

    return mu_hat, sigma_squared_hat


# Generate random data from a normal distribution
mu_true = 3.0  # True mean
sigma_true = 1.0  # True standard deviation
n_samples = 1000  # Number of samples to generate

# Generate random samples
X_random = np.random.normal(mu_true, sigma_true, n_samples)

# Maximum Likelihood Estimates (MLE) for mean and variance
mu_hat, sigma_squared_hat = maximum_likelihood_estimates(X_random)

# Compute likelihood and log-likelihood with MLE
likelihood, log_likelihood = univariate_normal_likelihood(X_random, mu_hat, sigma_squared_hat)

# Print results
print(f"MLE for mean (mu): {mu_hat}")
print(f"MLE for variance (sigma^2): {sigma_squared_hat}")
print(f"Likelihood: {likelihood}")
print(f"Log-Likelihood: {log_likelihood}")

# Plotting the normal distribution with the MLE parameters and the random data
x_vals = np.linspace(min(X_random) - 1, max(X_random) + 1, 1000)
y_vals = norm.pdf(x_vals, mu_hat, np.sqrt(sigma_squared_hat))

# Plot the normal distribution curve
plt.plot(x_vals, y_vals, label=f'Normal Distribution ($\mu={mu_hat:.2f}$, $\sigma^2={sigma_squared_hat:.2f}$)',
         color='blue')

# Plot the observed data points
plt.hist(X_random, bins=30, density=True, alpha=0.6, color='red', label='Random Data Histogram')

# Adding labels and title
plt.title("Univariate Normal Distribution with Random Data")
plt.xlabel("Data Value")
plt.ylabel("Density")
plt.legend()

# Display the plot
plt.show()
