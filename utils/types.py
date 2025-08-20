from enum import Enum
import torch


class AcquisitionFunctionType(Enum):
    qEHVI = "qEHVI"
    qLogEHVI = "qLogEHVI"
    qNEHVI = "qNEHVI"
    qLogNEHVI = "qLogNEHVI"
    qEWNEHVI = "qEWNEHVI"
    qDWNEHVI = "qDWNEHVI"
    qNParEGO = "qNParEGO"

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


class TorchDeviceType(Enum):
    CPU = "cpu"
    GPU = "cuda"
    MPS = "mps"

# class ParameterType(Enum):
#     objective = 0
#     constraint = 1
#     tracker = 2
#
#
# class Parameter:
#     def __init__(
#             self,
#             name: str,
#             value: torch.Tensor,
#             type: ParameterType,
#     ):
#         self.name = name
#         self.value = value
#         self.type = type
