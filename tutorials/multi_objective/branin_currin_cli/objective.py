import torch
import math
from pybo.objectives.base_class import MCMultiObjectiveBase
from torch import Tensor

from pybo.objectives.variable_registry import *


class BraninCurrin(MCMultiObjectiveBase):
    """
    Two objective problem composed of the Branin and Currin functions.

    Note:
    - Branin Function: Originally intended for minimization.
    - Currin Exponential Function: Originally intended for maximization.
    - Both the Branin and Currin functions are positive on their domain.

    For testing purposes, and following BoTorch authors implementation,
    both test functions are intended for minimization, against their original
    nature.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(0.0, 1.0)),
                ParCfg(bounds=(0.0, 1.0))
            ],
            obj_cfg=[
                ObjCfg(label="Branin", bounds=(0, 300), to_minimize=True, ref_point=18.0),
                ObjCfg(label="Currin", bounds=(0, 14), to_minimize=True, ref_point=6.0),
            ],
            max_hv=59.36011874867746,  # this is approximated using NSGA-II
        )

    @staticmethod
    def _branin(X: Tensor) -> Tensor:
        x0 = X[..., 0]
        x1 = X[..., 1]
        t1 = (x1 - 5.1 / (4 * math.pi ** 2) * x0.pow(2) + 5 / math.pi * x0 - 6)
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x0)
        return t1.pow(2) + t2 + 10

    def _rescaled_branin(self, X: Tensor) -> Tensor:
        """ For a visual reference visit: https://www.sfu.ca/~ssurjano/branin.html"""
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        return self._branin(torch.stack([x_0, x_1], dim=-1))

    @staticmethod
    def _currin(X: Tensor) -> Tensor:
        """ For a visual reference visit: https://www.sfu.ca/~ssurjano/curretal88exp.html """
        x0 = X[..., 0]
        x1 = X[..., 1]
        factor1 = 1 - torch.exp(-1 / (2 * x1))
        numer = 2300 * x0.pow(3) + 1900 * x0.pow(2) + 2092 * x0 + 60
        denom = 100 * x0.pow(3) + 500 * x0.pow(2) + 4 * x0 + 20
        return factor1 * numer / denom

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor | None:
        branin = self._rescaled_branin(X=X)
        currin = self._currin(X=X)
        f = torch.stack([branin, currin], dim=-1)
        return f
