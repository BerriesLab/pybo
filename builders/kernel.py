import inspect
from abc import ABC
from dataclasses import make_dataclass, asdict
from typing import Any, Optional, Type
from gpytorch.kernels import Kernel as GPyKernel, ScaleKernel, RBFKernel, CosineKernel, PeriodicKernel


class KernelBuilderBase(ABC):
    _base_kernel_class: Type[GPyKernel] = None

    def __init__(self):
        if self._base_kernel_class is None:
            raise ValueError("_base_kernel_class must be set in subclass.")

        self._generate_scale_runtime_params_dataclass()
        self._generate_base_runtime_params_dataclass()

        self._base_params = self.BaseKernelRuntimeParams()
        self._scale_params = self.ScaleKernelRuntimeParams()

    def _generate_scale_runtime_params_dataclass(self):
        sig = inspect.signature(ScaleKernel.__init__)
        field_list = self._get_field_list(sig)
        self.ScaleKernelRuntimeParams = make_dataclass(
            f"{ScaleKernel.__name__}Params", field_list, slots=True
        )

    def _generate_base_runtime_params_dataclass(self):
        sig = inspect.signature(self._base_kernel_class.__init__)
        field_list = self._get_field_list(sig)

        if not any(f[0] == "ard_num_dims" for f in field_list):
            field_list.append(("ard_num_dims", Optional[int], None))

        self.BaseKernelRuntimeParams = make_dataclass(
            f"{self._base_kernel_class.__name__}Params", field_list, slots=True
        )

    @staticmethod
    def _get_field_list(sig):
        field_list = []
        exclude = ("self", "base_kernel", "kwargs", "args")
        for name, param in sig.parameters.items():
            if name in exclude:
                continue
            default = param.default if param.default is not inspect.Parameter.empty else None
            field_list.append((name, Any, default))
        return field_list

    @property
    def base_params(self):
        return self._base_params

    @property
    def scale_params(self):
        return self._scale_params

    def build(self) -> GPyKernel:
        base_kwargs = {k: v for k, v in asdict(self._base_params).items() if v is not None}
        base_kernel = self._base_kernel_class(**base_kwargs)
        scale_kwargs = {k: v for k, v in asdict(self._scale_params).items() if v is not None}
        return ScaleKernel(base_kernel, **scale_kwargs)


# =========================
# Concrete Builders
# =========================

class RBFKernelBuilder(KernelBuilderBase):
    _base_kernel_class = RBFKernel


class CosineKernelBuilder(KernelBuilderBase):
    _base_kernel_class = CosineKernel


class PeriodicKernelBuilder(KernelBuilderBase):
    _base_kernel_class = PeriodicKernel
