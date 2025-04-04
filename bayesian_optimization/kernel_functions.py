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