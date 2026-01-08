from dataclasses import dataclass, field
from typing import Optional
from utils.types import KernelType
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


# === Basic Kernel Configs ===

@dataclass
class RBFConfig:
    lengthscale: Optional[float] = None
    lengthscale_constraint: Optional[Interval] = None


@dataclass
class MaternConfig:
    nu: float = 2.5
    lengthscale: Optional[float] = None
    lengthscale_constraint: Optional[Interval] = None


@dataclass
class PeriodicConfig:
    period_length: Optional[float] = None
    period_length_constraint: Optional[Interval] = None
    lengthscale: Optional[float] = None
    lengthscale_constraint: Optional[Interval] = None


# @dataclass
# class RQConfig:
#     lengthscale: Optional[float] = None
#     lengthscale_constraint: Optional[Interval] = None
#     alpha: Optional[float] = None
#     alpha_constraint: Optional[Interval] = None
#
#
# @dataclass
# class SpectralMixtureConfig:
#     num_mixtures: int = 4
#
#
# @dataclass
# class LinearConfig:
#     variance: Optional[float] = None
#     variance_constraint: Optional[Interval] = None
#
#
# @dataclass
# class PolynomialConfig:
#     power: int = 2
#     offset: Optional[float] = None
#     offset_constraint: Optional[Interval] = None
#
#
# @dataclass
# class CosineConfig:
#     period_length: Optional[float] = None
#     period_length_constraint: Optional[Interval] = None


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

        # elif self.kernel_type == KernelType.RQ:
        #     cfg = self.config or RQConfig()
        #     rq = RQKernel(
        #         ard_num_dims=self.ard_num_dims,
        #         lengthscale_constraint=cfg.lengthscale_constraint,
        #         alpha_constraint=cfg.alpha_constraint,
        #     )
        #     if cfg.lengthscale is not None:
        #         rq.lengthscale = cfg.lengthscale
        #     if cfg.alpha is not None:
        #         rq.alpha = cfg.alpha
        #     return ScaleKernel(rq)
        #
        # elif self.kernel_type == KernelType.SPECTRAL_MIXTURE:
        #     cfg = self.config or SpectralMixtureConfig()
        #     return SpectralMixtureKernel(
        #         num_mixtures=cfg.num_mixtures,
        #         ard_num_dims=self.ard_num_dims,
        #     )
        #
        # elif self.kernel_type == KernelType.LINEAR:
        #     cfg = self.config or LinearConfig()
        #     linear = LinearKernel(
        #         ard_num_dims=self.ard_num_dims,
        #         variance_constraint=cfg.variance_constraint,
        #     )
        #     if cfg.variance is not None:
        #         linear.variance = cfg.variance
        #     return ScaleKernel(linear)
        #
        # elif self.kernel_type == KernelType.POLYNOMIAL:
        #     cfg = self.config or PolynomialConfig()
        #     poly = PolynomialKernel(
        #         power=cfg.power,
        #         ard_num_dims=self.ard_num_dims,
        #         offset_constraint=cfg.offset_constraint,
        #     )
        #     if cfg.offset is not None:
        #         poly.offset = cfg.offset
        #     return ScaleKernel(poly)
        #
        # elif self.kernel_type == KernelType.COSINE:
        #     cfg = self.config or CosineConfig()
        #     cosine = CosineKernel(
        #         period_length_constraint=cfg.period_length_constraint,
        #     )
        #     if cfg.period_length is not None:
        #         cosine.period_length = cfg.period_length
        #     return ScaleKernel(cosine)

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
