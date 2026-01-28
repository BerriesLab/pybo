from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Optional, Callable


class ConstraintType(StrEnum):
    INPUT_LINEAR_EQUALITY = "input linear equality"
    INPUT_LINEAR_INEQUALITY = "input linear inequality"
    INPUT_NONLINEAR_INEQUALITY = "input nonlinear inequality"
    OUTPUT_INEQUALITY = "output inequality"


@dataclass(frozen=True)
class Cfg:
    """Base configuration class"""
    label: str
    index: int


@dataclass(frozen=True)
class ObjectiveCfg(Cfg):
    """Configuration for objectives"""
    f: Callable
    bounds: tuple[float, float] | None
    to_minimize: bool = False
    ref_point: float | None = None


@dataclass(frozen=True)
class ParameterCfg(Cfg):
    """Configuration for parameters"""
    bounds: tuple[float, float] | None
    pass


@dataclass(frozen=True)
class LinearEqualityInputConstraintCfg(Cfg):
    """Configuration for linear equality input constraints"""
    f: tuple[int, float, float]


@dataclass(frozen=True)
class LinearInequalityInputConstraintCfg(Cfg):
    """Configuration for linear inequality input constraints"""
    f: tuple[int, float, float]


@dataclass(frozen=True)
class NonLinearInequalityInputConstraintCfg(Cfg):
    """Configuration for nonlinear inequality input constraints"""
    f: (Callable, bool)


@dataclass(frozen=True)
class OutputConstraintCfg(Cfg):
    """Configuration for output constraints"""
    f: Callable


@dataclass(frozen=True)
class TrkCfg(Cfg):
    """Configuration for trackers """
    bounds: tuple[float, float] | None
    f: Callable


class VariableRegistry(Enum):
    @property
    def cfg(self) -> Cfg:
        return self.value  # type: ignore[return-value]

    @property
    def label(self) -> str: return self.cfg.label

    @property
    def index(self) -> int: return self.cfg.index

    @property
    def bounds(self) -> tuple[float, float] | None: return self.cfg.bounds

    @property
    def dtype(self) -> Any: return self.cfg.dtype

    @property
    def to_minimize(self) -> bool: return self.cfg.to_minimize

    @property
    def ref_point(self) -> float: return self.cfg.ref_point
