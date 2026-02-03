import torch
from objectives.base_class import MCMultiObjectiveBase
from torch import Tensor

from objectives.variable_registry import ParCfg, ObjCfg, LinIneqXConCfg


class LinearEqualityTestProblem(MCMultiObjectiveBase):
    r"""
    Two-objective problem with a linear equality constraint.

    Parameters:
        0 <= x1 <= 1
        0 <= x2 <= 1

    Objectives:
        1. Minimize distance from the origin:
           f1(x) = (x1 - 1)^2 + x2^2
        2. Minimize distance from the point (2,1):
           f2(x) = x1^2 + (x2 - 1)^2

    Constraint:
        Linear inequality constraint:
            x1 + 2x2 <= 1

    Notes:
        - The feasible region is the triangle defined by the bounds and the linear constraint.
        - The Pareto front arises from the trade-off between the two objectives inside the feasible region.
        - The inequality must be passed as \sum_i (X[indices[i]] * coefficients[i]) >= rhs
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(0.0, 1.0)),
                ParCfg(label="P2", index=1, bounds=(0.0, 1.0))
            ],
            obj_cfg=[
                ObjCfg(label="F1", index=0, bounds=None, to_minimize=True, ref_point=2.1, f=self._f1),
                ObjCfg(label="F2", index=1, bounds=None, to_minimize=True, ref_point=2.1, f=self._f2)
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(label="C1", index=0, idxs=[0, 1], coeff=[-1.0, -2.0], rhs=-1.0)
            ],
        )

    @staticmethod
    def _f1(X: Tensor) -> Tensor:
        return (X[..., 0] - 1).pow(2) + (X[..., 1]).pow(2)

    @staticmethod
    def _f2(X: Tensor) -> Tensor:
        return (X[..., 0]).pow(2) + (X[..., 1] - 1).pow(2)
