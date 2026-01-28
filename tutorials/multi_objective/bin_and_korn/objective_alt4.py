import torch
from objectives.base_class import MCMultiObjectiveBase
from objectives.variable_registry import *


class BinhAndKorn(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions. """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., self.par_cfg[0].index]
        x2 = X[..., self.par_cfg[1].index]
        return 4 * x1 ** 2 + 4 * x2 ** 2

    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return (x1 - 5) ** 2 + (x2 - 5) ** 2

    def _input_c1(self, X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return 25 - ((x1 - 5) ** 2 + x2 ** 2)

    def _input_c2(self, X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return (x1 - 8) ** 2 + (x2 + 3) ** 2 - 7.7

    def evaluate_true_objective(self, X: torch.Tensor, add_noise=False) -> torch.Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        return torch.stack([f1, f2], dim=-1)
