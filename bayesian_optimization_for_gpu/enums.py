from enum import Enum

import botorch.acquisition
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, ConstantKernel
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement


class AcquisitionFunction(Enum):
    qEHVI = qExpectedHypervolumeImprovement
    qNEHVI = qNoisyExpectedHypervolumeImprovement


class Kernel(Enum):
    # ScaledRBF = ScaleKernel(RBFKernel)
    RBF = RBFKernel
    Matern = MaternKernel
    Constant = ConstantKernel
