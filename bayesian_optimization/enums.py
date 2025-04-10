from enum import Enum

import sklearn.gaussian_process.kernels


class Kernel(Enum):
    RBF = sklearn.gaussian_process.kernels.RBF
    Matern = sklearn.gaussian_process.kernels.Matern
    Constant = sklearn.gaussian_process.kernels.ConstantKernel


class AcquisitionFunction(Enum):
    UCB = "ucb"
    EI = "ei"
    PI = "pi"
    POI = "poi"