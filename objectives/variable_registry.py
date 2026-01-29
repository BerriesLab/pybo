from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Any, Optional, Callable


class VariableRegistry(Enum):
    pass


@dataclass(frozen=True)
class Cfg:
    """Base configuration class"""
    label: str
    index: int


@dataclass(frozen=True)
class ObjCfg(Cfg):
    """Configuration for objectives"""
    f: Callable
    bounds: tuple[float, float] | None
    to_minimize: bool = False
    ref_point: float | None = None


@dataclass(frozen=True)
class ParCfg(Cfg):
    """Configuration for parameters"""
    bounds: tuple[float, float] | None


@dataclass(frozen=True)
class LinEqXConCfg(Cfg):
    r"""Configuration for linear equality X constraints.
    Example: if you want the constraint 3*X[0] + 2*X[2] = 5, you'd pass the
    tuple ([0, 2], [3, 2], 5), where idxs=[0, 2], coeff=[3, 2], and rhs=5"""
    idxs: list[int]
    coeff: list[float]
    rhs: float


@dataclass(frozen=True)
class LinIneqXConCfg(Cfg):
    r"""Configuration for linear inequality input constraints.

    Intra-point constraints are applied to each candidate individually.
    Example: if you want the constraint 3*X[0] + 2*X[2] >= 5, you'd pass the
    tuple ([0, 2], [3, 2], 5), where idxs=[0, 2], coeff=[3, 2], and rhs=5.

    Inter-point constraints are applied between features of candidate pairs
    Example: we have a batch of 3 candidates, each with d=5 features. The 2nd feature
    of candidate 0 plus twice the 4th feature of candidate 1 must be at least 20.
    In formula, the constraint reads X[0, 1] + 2 * X[1, 3] >= 20, corresponding to
    indxs=[[0, 1], [1, 3]], coeff=[1, 2], and rhs=20. """
    idxs: list[int]
    coeff: list[float]
    rhs: float


@dataclass(frozen=True)
class NonLinIneqXConCfg(Cfg):
    """Configuration for nonlinear inequality input ineq_Y_con_cfg"""
    f: Callable
    intra: bool


@dataclass(frozen=True)
class IneqYConCfg(Cfg):
    """Configuration for output ineq_Y_con_cfg"""
    f: Callable


@dataclass(frozen=True)
class TrkCfg(Cfg):
    """Configuration for trackers """
    bounds: tuple[float, float] | None
    f: Callable
