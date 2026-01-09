from enum import Enum


class AcquisitionFunctionType(Enum):
    """Acquisition function types."""

    # === Single-Objective: Analytical (q=1 only, no sampler needed) ===
    EI = "EI"
    LogEI = "LogEI"
    PI = "PI"
    UCB = "UCB"

    # === Single-Objective: Monte Carlo (supports q>=1) ===
    qEI = "qEI"
    qLogEI = "qLogEI"
    qNEI = "qNEI"
    qLogNEI = "qLogNEI"
    qPI = "qPI"
    qUCB = "qUCB"

    # === Multi-Objective ===
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
    def analytical_types(cls):
        """Analytical acquisition functions (q=1 only, no sampler)."""
        return {cls.EI, cls.LogEI, cls.PI, cls.UCB}

    @classmethod
    def single_objective_types(cls):
        """All single-objective acquisition functions."""
        return {cls.EI, cls.LogEI, cls.PI, cls.UCB,
                cls.qEI, cls.qLogEI, cls.qNEI, cls.qLogNEI, cls.qPI, cls.qUCB}

    @classmethod
    def multi_objective_types(cls):
        """All multi-objective acquisition functions."""
        return {cls.qEHVI, cls.qLogEHVI, cls.qNEHVI, cls.qLogNEHVI,
                cls.qEWNEHVI, cls.qDWNEHVI, cls.qNParEGO}

    @classmethod
    def types_requiring_best_f(cls):
        """Acquisition functions that require best_f."""
        return {cls.EI, cls.LogEI, cls.PI, cls.qEI, cls.qLogEI, cls.qPI}

    @classmethod
    def types_requiring_sampler(cls):
        """Acquisition functions that require a sampler."""
        return (cls.single_objective_types() - cls.analytical_types()) | cls.multi_objective_types()

    def is_analytical(self):
        return self in self.analytical_types()

    def is_single_objective(self):
        return self in self.single_objective_types()

    def is_multi_objective(self):
        return self in self.multi_objective_types()

    def requires_best_f(self):
        return self in self.types_requiring_best_f()

    def requires_sampler(self):
        return self in self.types_requiring_sampler()


class SamplerType(Enum):
    Sobol = "Sobol"
    LatinHypercube = "Latin Hypercube"

    @classmethod
    def values(cls):
        return [e.value for e in cls]


class KernelType(Enum):
    """Supported kernel types for the Gaussian Process.

    Basic kernels:
        RBF: Smooth, infinitely differentiable (default)
        MATERN: Adjustable smoothness (nu=2.5 default)
        PERIODIC: For sinusoidal/seasonal patterns
        RQ: Rational Quadratic - mixture of RBFs
        SPECTRAL_MIXTURE: Learns periodicities from data
        LINEAR: Linear relationships
        POLYNOMIAL: Polynomial relationships
        COSINE: Pure cosine kernel

    Composite kernels:
        RBF_PLUS_PERIODIC: Trend + seasonality
        RBF_TIMES_PERIODIC: Locally periodic patterns
        MATERN_PLUS_PERIODIC: Rougher trend + seasonality
    """

    RBF = "RBF"
    MATERN = "Matern"
    PERIODIC = "Periodic"
    RQ = "RQ"
    SPECTRAL_MIXTURE = "Spectral_mixture"
    LINEAR = "Linear"
    POLYNOMIAL = "Polynomial"
    COSINE = "Cosine"

    # Composite kernels
    RBF_PLUS_PERIODIC = "RBF_plus_Periodic"
    RBF_TIMES_PERIODIC = "RBF_times_Periodic"
    MATERN_PLUS_PERIODIC = "Matern_plus_Periodic"
