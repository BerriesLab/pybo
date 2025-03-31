import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Define multi-dimensional input data (R^2 space)
X = np.array([[1.0, 2.0], [3.0, 1.5], [4.0, 3.2]])  # 3 points in R^2
y = np.array([3.1, 2.9, 3.5])  # Function values

# Define an ARD kernel (different length scales for each dimension)
kernel = C(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(noise_level=1.0)

# Fit the GP model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X, y)

# Print the optimized kernel parameters
print("Optimized Kernel Parameters:", gp.kernel_)
