import numpy as np

from bayesian_optimization.bayesian_optimization import BayesianOptimization
from bayesian_optimization.enums import Kernel
from bayesian_optimization.objective_functions import test_f0_1d

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


# Y_init_1d = test_f0_1d(X_init_1d).reshape(-1, 1)


# X_init_3d = np.array([[-4, -3, -2], [-2, +0, +2]])
# Y_init_3d = np.sin(X_init_3d[:, 0]) + 2 * np.cos(X_init_3d[:, 1]) + np.tanh(X_init_3d[:, 2])


opt = BayesianOptimization()
opt.import_data("../data/data_format.csv")
opt.export_data("../data/data_format_out.csv")

# For demo purposes only
X_init_1d = np.array([[-4], [-2], [0], [+2], [+4]])
Y_init_1d = test_f0_1d(X_init_1d)
bounds_1d = [(-5, 5)]
# TODO: add grid sampling and LHS
opt.set_initial_dataset(X_init_1d, Y_init_1d)
opt.set_bounds(bounds_1d)
opt.set_objective_function(test_f0_1d)

opt.set_observation_noise(1e-6)
opt.set_acquisition_function("EI")
opt.set_kernel(Kernel.RBF.value(length_scale=1.0))
opt.set_number_of_objective_function_calls(50)
opt.set_number_of_optimizer_restarts(50)
opt.set_callback(None)
print(opt.get_attributes())
opt.optimize()