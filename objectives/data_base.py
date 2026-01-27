from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
import torch


@dataclass(frozen=True, slots=True)
class Config:
    label: str
    bounds: tuple[float, float] | Optional[None]
    dtype: torch.dtype
    index: int = 0
    to_minimize: bool = False
    ref_point: Optional[float] = None


class VariableRegistry(Enum):
    @property
    def cfg(self) -> Config:
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
