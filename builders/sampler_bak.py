import inspect
from typing import Type, Callable, Any
from abc import ABC
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from dataclasses import dataclass, fields, make_dataclass
from torch import Tensor


@dataclass
class SamplerConfig:
    """ Public parameters """
    n_dimensions: int
    device: torch.device
    dtype: torch.dtype
    bounds: Optional[Tensor] = None
    seed: Optional[int] = None
    scramble: bool = True
    linear_equality_constraints: Optional[list] = None
    linear_inequality_constraints: Optional[list] = None
    nonlinear_inequality_constraints: Optional[list] = None


PARAM_MAP = {
    "SobolSampler": {
        "n_dimensions": "dimension",
        "seed": "seed",
        "scramble": "scramble"
    },
    "LHSSampler": {
        "n_dimensions": "d",
        "seed": "seed",
        "optimization": "optimization"
    },
    "GridSampler": {
        "n_dimensions": "dims"
    }
}


class SamplerBuilder(ABC):
    _sampler_class: Type

    @dataclass
    class Constraints:
        linear_equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        linear_inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
        nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,

    def __init__(self):
        self._runtime_params = None
        self._generate_runtime_params_dataclass()

    def _generate_runtime_params_dataclass(self):
        """Extracts the parameters from the constructor of the target class
        and generate a dataclass with the target class' attributes. """
        if self._sampler_class is None:
            raise RuntimeError("Must define a sampler class first.")

        sig = inspect.signature(self._sampler_class.__init__)
        params = []

        mapping = PARAM_MAP.get(self._class_name, {})
        inverse_mapping = {v: k for k, v in mapping.items()}

        for name, param in sig.parameters.items():
            if name in ("self",):
                continue

            # Default to None if no default value is provided in the signature
            public_name = inverse_mapping.get(name, name)
            default_value = param.default if param.default is not inspect.Parameter.empty else None
            params.append((public_name, Any, default_value))

        # Dynamically create the dataclass
        self.RuntimeParams = make_dataclass(
            f"{self._sampler_class.__name__}Params",
            fields=params,
            bases=(self.Constraints,)
        )

    def build(self):
        """ Generic implementation that instantiates the sampler using
        the stored runtime parameters. """
        if self._runtime_params is None:
            raise ValueError("Runtime parameters have not been built yet.")
        params_dict = {f.name: getattr(self._runtime_params, f.name)
                       for f in fields(self._runtime_params)}
        return self._sampler_class(**params_dict)

    @property
    def runtime_params(self):
        return self._runtime_params


class SobolSamplerBuilder(SamplerBuilder):
    _sampler_class = SobolEngine

    def __init__(self):
        super().__init__()


class LatinHypercubeSamplerBuilder(SamplerBuilder):
    _sampler_class = LatinHypercube

    def __init__(self):
        super().__init__()
