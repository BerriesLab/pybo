import torch
import inspect
import numpy as np
from torch import Tensor
from dataclasses import dataclass, fields
from typing import Optional, Any, Callable, Type
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine


# ----------------------------------------------------------------#
# 1. PUBLIC INTERFACE (CONFIG)
# ----------------------------------------------------------------#

@dataclass
class SamplerConfig:
    """ User public interface. """
    n_dimensions: int
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    bounds: Optional[Tensor] = None
    seed: Optional[int] = None
    scramble: bool = True
    normalize: bool = True

    linear_equality_constraints: Optional[list] = None
    linear_inequality_constraints: Optional[list] = None
    nonlinear_inequality_constraints: Optional[list] = None

    def __post_init__(self):
        """Valida la configurazione prima della costruzione."""
        if self.bounds is not None:
            if self.bounds.shape != (2, self.n_dimensions):
                raise ValueError(
                    f"I bounds devono avere forma (2, {self.n_dimensions}), "
                    f"ma hanno forma {list(self.bounds.shape)}."
                )


# ----------------------------------------------------------------#
# 2. BASE SAMPLER & ENGINES
# ----------------------------------------------------------------#

class BaseSampler:
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            n_dimensions: int,
            bounds: Optional[Tensor] = None,
            linear_equality_constraints: Optional[list] = None,
            linear_inequality_constraints: Optional[list] = None,
            nonlinear_inequality_constraints: Optional[list] = None,
            **kwargs
    ):
        self.device = device
        self.dtype = dtype
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        self.linear_equality_constraints = linear_equality_constraints
        self.linear_inequality_constraints = linear_inequality_constraints
        self.nonlinear_inequality_constraints = nonlinear_inequality_constraints

    def _apply_constraints(self, X: Tensor) -> Tensor:
        """Applica proiezioni e filtri ai campioni generati."""
        # Logica di proiezione su uguaglianze (omessa per brevità, inserisci la tua funzione qui)
        # Logica di filtraggio su disuguaglianze (omessa per brevità)
        return X

    def draw(self, n: int) -> Tensor:
        raise NotImplementedError("Sottoclassi devono implementare draw()")


class SobolSampler(BaseSampler):
    def __init__(self, dimension: int, scramble: bool = True, seed: Optional[int] = None, **kwargs):
        super().__init__(n_dimensions=dimension, **kwargs)
        self.engine = SobolEngine(dimension=dimension, scramble=scramble, seed=seed)

    def draw(self, n: int) -> Tensor:
        X = self.engine.draw(n).to(device=self.device, dtype=self.dtype)
        return self._apply_constraints(X)


class LHSSampler(BaseSampler):
    def __init__(self, d: int, seed: Optional[int] = None, **kwargs):
        super().__init__(n_dimensions=d, **kwargs)
        self.engine = LatinHypercube(d=d, seed=seed)

    def draw(self, n: int) -> Tensor:
        samples = self.engine.random(n)
        X = torch.tensor(samples, device=self.device, dtype=self.dtype)
        return self._apply_constraints(X)


class GridSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def draw(self, n: int) -> Tensor:
        points_per_dim = int(round(n ** (1 / self.n_dimensions)))
        points_per_dim = max(1, points_per_dim)

        grid_axes = [torch.linspace(0, 1, points_per_dim, device=self.device, dtype=self.dtype)
                     for _ in range(self.n_dimensions)]
        grid = torch.meshgrid(*grid_axes, indexing='ij')
        X = torch.stack(grid, dim=-1).reshape(-1, self.n_dimensions)
        return self._apply_constraints(X)


# ----------------------------------------------------------------#
# 3. BUILDER & MAPPING
# ----------------------------------------------------------------#

# Mappa per tradurre i nomi pubblici in nomi richiesti dalle librerie (PyTorch/SciPy)
SAMPLER_MAPPING = {
    "SobolSampler": {"n_dimensions": "dimension"},
    "LHSSampler": {"n_dimensions": "d"},
    "GridSampler": {"n_dimensions": "n_dimensions"}
}


class SamplerBuilder:
    def __init__(self, sampler_class: Type[BaseSampler]):
        self._sampler_class = sampler_class
        self._class_name = sampler_class.__name__

    def build(self, config: SamplerConfig) -> BaseSampler:
        mapping = SAMPLER_MAPPING.get(self._class_name, {})
        config_dict = {f.name: getattr(config, f.name) for f in fields(config)}

        # Traduzione parametri
        constructor_params = {}
        for pub_name, value in config_dict.items():
            internal_name = mapping.get(pub_name, pub_name)
            constructor_params[internal_name] = value

        # Ispezione per passare solo i parametri validi
        sig = inspect.signature(self._sampler_class.__init__)
        valid_params = {
            k: v for k, v in constructor_params.items()
            if k in sig.parameters or sig.parameters.get('kwargs')
        }

        return self._sampler_class(**valid_params)
