import torch
from pybo.objectives.base_class import MCMultiObjectiveBase
from torch import Tensor

from pybo.objectives.variable_registry import ParCfg, ObjCfg, LinIneqXConCfg


class LinearInequalityTest(MCMultiObjectiveBase):
    r"""
    Multi-objective optimization problem with two objectives and a linear constraint.

    Parameters:
        0 <= x1 <= 3
        0 <= x2 <= 3

    Objectives:
        1. Minimize distance from the origin:
           f1(x) = x1^2 + x2^2
        2. Minimize distance from the point (2,1):
           f2(x) = (x1 - 2)^2 + (x2 - 1)^2

    Constraint:
        Linear inequality constraint:
            x1 + x2 <= 2
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(0.0, 3.0)),
                ParCfg(bounds=(0.0, 3.0)),
            ],
            obj_cfg=[
                ObjCfg(to_minimize=True, ref_point=10.0),
                ObjCfg(to_minimize=True, ref_point=6.0)
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1.0, -1.0], rhs=-2.0)
            ],
            gt_obj_noise_std=[0.03, 0.03]
        )

    @staticmethod
    def _f1(X: Tensor) -> Tensor:
        return (X[..., 0]).pow(2) + (X[..., 1]).pow(2)

    @staticmethod
    def _f2(X: Tensor) -> Tensor:
        return (X[..., 0] - 2).pow(2) + (X[..., 1] - 1).pow(2)

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._f1(X), self._f2(X)], dim=-1)
