from enum import Enum


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


class SamplerType(Enum):
    Sobol = "Sobol"
    LatinHypercube = "Latin Hypercube"
