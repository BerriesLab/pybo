# define a kernel function to return a squared exponential distance
import numpy as np


def exponential(a, b):
    """
    Computes the exponential of the negative squared Euclidean distance
    between two sets of vectors.

    This function calculates the pairwise squared Euclidean distance matrix
    between two matrices `a` and `b`. It then applies the exponential function
    to the negative values of the distance matrix.

    :param a: The first matrix of shape (n, N), where `n` is the number of
              vectors and `N` is the dimensionality of each vector.
    :param b: The second matrix of shape (m, N), where `m` is the number of
              vectors and `N` is the dimensionality of each vector.
    :return: A matrix of shape (n, m), where each element represents the
             exponential result of the negative squared distance between a vector
             in `a` and a vector in `b`.
    :rtype: numpy.ndarray
    """
    sq_dist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-sq_dist)


def rbf(x1, x2, l=1.0, sigma=1.0):
    sq_dist = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
    return sigma ** 2 * np.exp(-sq_dist / (2 * l ** 2))


def matern(x1, x2, nu=1.5, l=1.0, sigma=1.0):
    """
    Computes the Matern kernel between two sets of vectors.

    :param x1: First input matrix (n, d) where `n` = number of data points, `d` = dimensions.
    :param x2: Second input matrix (m, d) where `m` = number of data points, `d` = dimensions.
    :param nu: Smoothness parameter, determines the differentiability of the kernel.
               Common values are 0.5, 1.5, 2.5, etc.
    :param l: Length scale parameter. Default is 1.0.
    :param sigma: Output scale parameter. Default is 1.0.
    :return: A kernel matrix computed between x1 and x2.
    :rtype: numpy.ndarray
    """
    # Pairwise Euclidean distance
    dists = np.sqrt(np.sum(x1 ** 2, axis=1).reshape(-1, 1) + np.sum(x2 ** 2, axis=1) - 2 * np.dot(x1, x2.T))
    dists = np.maximum(dists, 1e-12)  # Prevent numerical instability for very small distances

    if nu == 0.5:
        # Special case: Matern kernel with nu=0.5 is equivalent to the exponential kernel
        return sigma ** 2 * np.exp(-dists / l)
    elif nu == 1.5:
        # Special case: Matern kernel with nu=1.5
        sqrt_3 = np.sqrt(3)
        return sigma ** 2 * (1 + sqrt_3 * dists / l) * np.exp(-sqrt_3 * dists / l)
    elif nu == 2.5:
        # Special case: Matern kernel with nu=2.5
        sqrt_5 = np.sqrt(5)
        return sigma ** 2 * (1 + sqrt_5 * dists / l + (5 / 3) * (dists ** 2) / (l ** 2)) * np.exp(-sqrt_5 * dists / l)
    else:
        # General case: use the scipy implementation for arbitrary nu
        from scipy.special import kv, gamma
        factor = (2 ** (1 - nu)) / gamma(nu)
        scaled_dists = np.sqrt(2 * nu) * dists / l
        return sigma ** 2 * factor * (scaled_dists ** nu) * kv(nu, scaled_dists)