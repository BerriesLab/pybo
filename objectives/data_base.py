from dataclasses import dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    label: str
    index: int
    bounds: tuple[float, float] | None
    dtype: Any
    to_minimize: bool = False


class DataBase(Enum):
    # The enum value *is* the Config for that key.
    @property
    def cfg(self) -> Config:
        return self.value  # type: ignore[return-value]

    # Optional convenience shortcuts:
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
