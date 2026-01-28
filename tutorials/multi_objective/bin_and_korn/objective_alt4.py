import torch
from objectives.base_class import MCMultiObjectiveBase
from objectives.variable_registry import *


class BinhAndKorn(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions. """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)

    def evaluate_true_objective(self, X: torch.Tensor, add_noise=False) -> torch.Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        return torch.stack([f1, f2], dim=-1)
