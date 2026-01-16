from dataclasses import dataclass, field
from typing import Optional
from utils.bo_types import KernelType
from gpytorch.constraints import Interval
from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel as StandardRBFKernel,
    MaternKernel,
    PeriodicKernel,
    RQKernel,
    SpectralMixtureKernel,
    LinearKernel,
    PolynomialKernel,
    CosineKernel,
)


class Kernel:
    def __init__():
        cfg = None

    def __call__(self):


# === Basic Kernel Configs ===

@dataclass
class ScaleConfig:
    outputscale: Optional[float] = None
    outputscale_constraint: Optional[Interval] = None


@dataclass
class RBFConfig:
    lengthscale: Optional[float] = None
    lengthscale_constraint: Optional[Interval] = None
    lengthscale_prior: Optional[Interval] = None
    outputscale_constraint: Optional[Interval] = None


@dataclass
class MaternConfig:
    nu: float = 2.5
    lengthscale: Optional[float] = None
    lengthscale_constraint: Optional[Interval] = None


@dataclass
class CosineConfig(ScaleConfig):
    def __init__(self):
        super().__init__()

    period_length: Optional[float] = None
    period_length_constraint: Optional[Interval] = None


@dataclass
class PeriodicConfig:
    period_length: Optional[float] = None
    period_length_constraint: Optional[Interval] = None
    lengthscale: Optional[float] = None
    lengthscale_constraint: Optional[Interval] = None


# === Composite Kernel Configs ===

@dataclass
class RBFPlusPeriodicConfig:
    rbf: RBFConfig = field(default_factory=RBFConfig)
    periodic: PeriodicConfig = field(default_factory=PeriodicConfig)


@dataclass
class RBFTimesPeriodicConfig:
    rbf: RBFConfig = field(default_factory=RBFConfig)
    periodic: PeriodicConfig = field(default_factory=PeriodicConfig)


@dataclass
class MaternPlusPeriodicConfig:
    matern: MaternConfig = field(default_factory=MaternConfig)
    periodic: PeriodicConfig = field(default_factory=PeriodicConfig)


class KernelFactory:
    def __init__(self, kernel_type: KernelType, ard_num_dims: int, config=None):
        self.kernel_type = kernel_type
        self.ard_num_dims = ard_num_dims
        self.config = config

    def __call__(self):

        # === Basic Kernels ===
        if self.kernel_type == KernelType.RBF:
            cfg = self.config or RBFConfig()
            rbf = StandardRBFKernel(
                ard_num_dims=self.ard_num_dims,
                lengthscale_constraint=cfg.lengthscale_constraint,
            )
            if cfg.lengthscale is not None:
                rbf.lengthscale = cfg.lengthscale
            return ScaleKernel(rbf)

        elif self.kernel_type == KernelType.MATERN:
            cfg = self.config or MaternConfig()
            matern = MaternKernel(
                nu=cfg.nu,
                ard_num_dims=self.ard_num_dims,
                lengthscale_constraint=cfg.lengthscale_constraint,
            )
            if cfg.lengthscale is not None:
                matern.lengthscale = cfg.lengthscale
            return ScaleKernel(matern)

        elif self.kernel_type == KernelType.COSINE:
            cfg = self.config or CosineConfig()
            cosine = CosineKernel(
                ard_num_dims=self.ard_num_dims,
                periodic_length=cfg.periodic_length,
            )
            return ScaleKernel(
                base_kernel=cosine,
                cfg=ScaleConfig(
                    outputscale=cfg.outputscale,
                    outputscale_constraint=cfg.outputscale_constraint,
                )
            )

        elif self.kernel_type == KernelType.PERIODIC:
            cfg = self.config or PeriodicConfig()
            periodic = PeriodicKernel(
                period_length_constraint=cfg.period_length_constraint,
                lengthscale_constraint=cfg.lengthscale_constraint,
            )
            if cfg.period_length is not None:
                periodic.period_length = cfg.period_length
            if cfg.lengthscale is not None:
                periodic.lengthscale = cfg.lengthscale
            return ScaleKernel(periodic)

        # === Composite Kernels ===
        elif self.kernel_type == KernelType.RBF_PLUS_PERIODIC:
            cfg = self.config or RBFPlusPeriodicConfig()

            rbf = StandardRBFKernel(
                ard_num_dims=self.ard_num_dims,
                lengthscale_constraint=cfg.rbf.lengthscale_constraint,
            )
            if cfg.rbf.lengthscale is not None:
                rbf.lengthscale = cfg.rbf.lengthscale

            periodic = PeriodicKernel(
                period_length_constraint=cfg.periodic.period_length_constraint,
                lengthscale_constraint=cfg.periodic.lengthscale_constraint,
            )
            if cfg.periodic.period_length is not None:
                periodic.period_length = cfg.periodic.period_length
            if cfg.periodic.lengthscale is not None:
                periodic.lengthscale = cfg.periodic.lengthscale

            return ScaleKernel(rbf) + ScaleKernel(periodic)

        elif self.kernel_type == KernelType.RBF_TIMES_PERIODIC:
            cfg = self.config or RBFTimesPeriodicConfig()

            rbf = StandardRBFKernel(
                ard_num_dims=self.ard_num_dims,
                lengthscale_constraint=cfg.rbf.lengthscale_constraint,
            )
            if cfg.rbf.lengthscale is not None:
                rbf.lengthscale = cfg.rbf.lengthscale

            periodic = PeriodicKernel(
                period_length_constraint=cfg.periodic.period_length_constraint,
                lengthscale_constraint=cfg.periodic.lengthscale_constraint,
            )
            if cfg.periodic.period_length is not None:
                periodic.period_length = cfg.periodic.period_length
            if cfg.periodic.lengthscale is not None:
                periodic.lengthscale = cfg.periodic.lengthscale

            return ScaleKernel(rbf * periodic)

        elif self.kernel_type == KernelType.MATERN_PLUS_PERIODIC:
            cfg = self.config or MaternPlusPeriodicConfig()

            matern = MaternKernel(
                nu=cfg.matern.nu,
                ard_num_dims=self.ard_num_dims,
                lengthscale_constraint=cfg.matern.lengthscale_constraint,
            )
            if cfg.matern.lengthscale is not None:
                matern.lengthscale = cfg.matern.lengthscale

            periodic = PeriodicKernel(
                period_length_constraint=cfg.periodic.period_length_constraint,
                lengthscale_constraint=cfg.periodic.lengthscale_constraint,
            )
            if cfg.periodic.period_length is not None:
                periodic.period_length = cfg.periodic.period_length
            if cfg.periodic.lengthscale is not None:
                periodic.lengthscale = cfg.periodic.lengthscale

            return ScaleKernel(matern) + ScaleKernel(periodic)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
