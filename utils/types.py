from enum import Enum


class AcquisitionFunctionType(Enum):
    # Single-objective acquisition functions
    qEI = "qEI"
    qLogEI = "qLogEI"
    qNEI = "qNEI"
    qLogNEI = "qLogNEI"
    qPI = "qPI"
    qUCB = "qUCB"

    # Multi-objective acquisition functions
    qEHVI = "qEHVI"
    qLogEHVI = "qLogEHVI"
    qNEHVI = "qNEHVI"
    qLogNEHVI = "qLogNEHVI"
    qEWNEHVI = "qEWNEHVI"
    qDWNEHVI = "qDWNEHVI"
    qNParEGO = "qNParEGO"

    @classmethod
    def values(cls):
        return [e.value for e in cls]

    @classmethod
    def single_objective_types(cls):
        return [cls.qEI, cls.qLogEI, cls.qNEI, cls.qLogNEI, cls.qPI, cls.qUCB]

    @classmethod
    def multi_objective_types(cls):
        return [cls.qEHVI, cls.qLogEHVI, cls.qNEHVI, cls.qLogNEHVI, cls.qEWNEHVI, cls.qDWNEHVI, cls.qNParEGO]

    @classmethod
    def types_requiring_best_f(cls):
        return [cls.qEI, cls.qLogEI, cls.qPI]

    def is_single_objective(self):
        return self in self.single_objective_types()

    def is_multi_objective(self):
        return self in self.multi_objective_types()

    def requires_best_f(self):
        """Check if this acquisition function requires best_f."""
        return self in self.types_requiring_best_f()


class SamplerType(Enum):
    Sobol = "Sobol"
    LatinHypercube = "Latin Hypercube"

    @classmethod
    def values(cls):
        return [e.value for e in cls]
