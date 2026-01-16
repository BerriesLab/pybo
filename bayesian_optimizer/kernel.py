from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from gpytorch.kernels import Kernel as GPyKernel, CosineKernel, PeriodicKernel
from gpytorch.kernels import RBFKernel as StandardRBFKernel, ScaleKernel
from gpytorch.priors import Prior


@dataclass
class ScaleKernelConfig:
    outputscale: Optional[float] = None
    outputscale_prior: Optional[Prior] = None
    outputscale_constraint: Optional = None


@dataclass
class RBFKernelConfig:
    lengthscale: Optional[float] = None
    lengthscale_prior: Optional[Prior] = None
    lengthscale_constraint: Optional = None


@dataclass
class CosineKernelConfig:
    period_length: Optional[float] = None
    period_length_prior: Optional[Prior] = None
    period_length_constraint: Optional = None


@dataclass
class PeriodicKernelConfig:
    period_length: Optional[float] = None
    period_length_prior: Optional[Prior] = None
    period_length_constraint: Optional = None


class KernelBuilderBase(ABC):
    """Base builder: includes cfg and builds a gpytorch Kernel."""

    def __init__(self, ard_num_dims: int, base_cfg=None, scale_cfg=None):
        self.ard_num_dims = ard_num_dims
        self.base_cfg = base_cfg
        self.scale_cfg = scale_cfg

    @abstractmethod
    def build(self) -> GPyKernel:
        pass


class RBFKernelBuilder(KernelBuilderBase):
    def __init__(self, ard_num_dims: int, base_cfg: RBFKernelConfig, scale_cfg: ScaleKernelConfig):
        super().__init__(ard_num_dims=ard_num_dims, base_cfg=base_cfg, scale_cfg=scale_cfg)

    def build(self) -> GPyKernel:
        # Instantiate Kernel
        base = StandardRBFKernel(
            ard_num_dims=self.ard_num_dims,
            lengthscale_constraint=self.base_cfg.lengthscale_constraint,
        )
        scaled = ScaleKernel(
            base_kernel=base,
            outputscale_constraint=self.scale_cfg.outputscale_constraint,
        )

        # Set initial conditions (including prior)
        if self.base_cfg.lengthscale is not None:
            base.lengthscale = self.base_cfg.lengthscale

        if self.base_cfg.lengthscale_prior is not None:
            base.lengthscale_prior = self.base_cfg.lengthscale_prior

        if self.scale_cfg.outputscale is not None:
            scaled.outputscale = self.scale_cfg.outputscale

        if self.scale_cfg.outputscale_prior is not None:
            scaled.outputscale_prior = self.scale_cfg.outputscale_prior

        return scaled


class CosineKernelBuilder(KernelBuilderBase):
    def __init__(self, ard_num_dims: int, base_cfg: CosineKernelConfig, scale_cfg: ScaleKernelConfig):
        super().__init__(ard_num_dims=ard_num_dims, base_cfg=base_cfg, scale_cfg=scale_cfg)

    def build(self) -> GPyKernel:
        base = CosineKernel(
            ard_num_dims=self.ard_num_dims,
            period_length_constraint=self.base_cfg.period_length_constraint,
        )
        scaled = ScaleKernel(
            base_kernel=base,
            outputscale_constraint=self.scale_cfg.outputscale_constraint,
        )
        if self.base_cfg.period_length is not None:
            base.period_length = self.base_cfg.period_length

        if self.base_cfg.period_length_prior is not None:
            base.period_length_prior = self.base_cfg.period_length_prior

        if self.scale_cfg.outputscale is not None:
            base.period_scale = self.scale_cfg.outputscale

        if self.scale_cfg.outputscale_prior is not None:
            base.period_scale_prior = self.scale_cfg.outputscale_prior

        return scaled


class PeriodicKernelBuilder(KernelBuilderBase):
    def __init__(self, ard_num_dims: int, base_cfg: PeriodicKernelConfig, scale_cfg: ScaleKernelConfig):
        super().__init__(ard_num_dims=ard_num_dims, base_cfg=base_cfg, scale_cfg=scale_cfg)

    def build(self) -> GPyKernel:
        base = PeriodicKernel(
            ard_num_dims=self.ard_num_dims,
            period_length_prior=self.base_cfg.period_length_prior,
            period_length_constraint=self.base_cfg.period_length_constraint,
        )
        scaled = ScaleKernel(
            base_kernel=base,
            outputscale_constraint=self.scale_cfg.outputscale_constraint,
        )
        base.period_length = self.base_cfg.period_length
        scaled.outputscale = self.scale_cfg.outputscale

        return scaled
