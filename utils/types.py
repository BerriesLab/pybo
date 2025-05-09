from enum import Enum

from botorch.acquisition.multi_objective import qNoisyExpectedHypervolumeImprovement


class AcquisitionFunctionType(Enum):
    qEHVI = "qEHVI"
    qLogEHVI = "qLogEHVI"
    qNEHVI = "qNEHVI"
    qLogNEHVI = "qLogNEHVI"

    @classmethod
    def values(cls):
        values = []
        for item in cls:
            values.append(item.value)
        return values


class Kernel(Enum):
    RBF = "RBF"
    Matern = "Matern"
    Constant = "Constant"


class SamplerType(Enum):
    Sobol = "Sobol"
    LatinHypercube = "Latin Hypercube"


class OptimizationProblemType(Enum):
    Minimization = 0
    Maximization = 1


class TorchDeviceType(Enum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"
