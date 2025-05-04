from enum import Enum


class AcquisitionFunctionType(Enum):
    qEHVI = "qEHVI"
    qLogEHVI = "qLogEHVI"
    qNEHVI = "qNEHVI"

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
    Minimization = "min"
    Maximization = "max"


class TorchDeviceType(Enum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"
