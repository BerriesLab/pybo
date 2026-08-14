from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Optional, Callable


class VariableRegistry(Enum):
    pass


@dataclass(kw_only=True)
class CfgBase:
    """Base configuration class"""
    label: str | None = None
    unit: str | None = None
    index: int | None = field(default=None, init=False)

    @property
    def axis_label(self) -> str:
        """Label and unit as one string, for an axis title."""
        return f"{self.label} ({self.unit})" if self.unit else self.label


@dataclass(kw_only=True)
class ObjCfg(CfgBase):
    """Configuration for objectives"""
    to_minimize: bool
    ref_point: float | None = None
    bounds: tuple[float, float] | None = None


@dataclass(kw_only=True)
class ParCfg(CfgBase):
    """Configuration for parameters. """
    bounds: tuple[float, float]
    resolution: float | None = None  # rig's smallest step, in this parameter's own units

    def __post_init__(self):
        # A step of zero is not a step. It reads as "whole units" but would silently mean
        # "unknown" everywhere downstream, so it is rejected rather than guessed at: write
        # 1 for whole units, None for unknown.
        if self.resolution is not None and self.resolution <= 0:
            raise ValueError(
                f"{self.label or 'parameter'}: resolution must be positive or None, got "
                f"{self.resolution}. Use 1 for whole units, None when it is unknown.")  # E.g. 0.001 V. None means unknown.


@dataclass(kw_only=True)
class LinEqXConCfg(CfgBase):
    r"""Configuration for linear equality X constraints.
    Example: if you want the constraint 3*X[0] + 2*X[2] = 5, you'd pass the
    tuple ([0, 2], [3, 2], 5), where idxs=[0, 2], coeff=[3, 2], and rhs=5"""
    idxs: list[int]
    coeff: list[float]
    rhs: float


@dataclass(kw_only=True)
class LinIneqXConCfg(CfgBase):
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


@dataclass(kw_only=True)
class NonLinIneqXConCfg(CfgBase):
    """Configuration for nonlinear inequality input constraints"""
    f: Callable
    intra: bool


@dataclass(kw_only=True)
class IneqYConCfg(CfgBase):
    """Configuration for output constraints"""
    f: Callable


@dataclass(kw_only=True)
class TrkCfg(CfgBase):
    """Configuration for trackers """
    bounds: tuple[float, float] | None = None
