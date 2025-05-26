from enum import Enum


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

    @staticmethod
    def require_partitioning():
        return ["qEHVI", "qLogEHVI"]


class Kernel(Enum):
    RBF = "RBF"
    Matern = "Matern"
    Constant = "Constant"


class SamplerType(Enum):
    Sobol = "Sobol"
    LatinHypercube = "Latin Hypercube"


class OptimizationProblemType(Enum):
    """ Maximization is set to True as BoTorch uses maximization by default."""
    Minimization = False
    Maximization = True


class TorchDeviceType(Enum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"
