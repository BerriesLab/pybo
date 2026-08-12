import torch
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, LinEqXConCfg


class LinearEqualityTest(MCMultiObjectiveBase):
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
            x1 + 2x2 = 1
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
                ObjCfg(to_minimize=True, ref_point=2.1),
                ObjCfg(to_minimize=True, ref_point=2.1)
            ],
            lin_eq_X_con_cfg=[
                LinEqXConCfg(idxs=[0, 1], coeff=[1.0, 2.0], rhs=1.0)
            ],
        )

    @staticmethod
    def _f1(X: torch.Tensor) -> torch.Tensor:
        return (X[..., 0] - 1).pow(2) + (X[..., 1]).pow(2)

    @staticmethod
    def _f2(X: torch.Tensor) -> torch.Tensor:
        return (X[..., 0]).pow(2) + (X[..., 1] - 1).pow(2)

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        Y = torch.stack([self._f1(X), self._f2(X)], dim=-1)
        if noisy:
            Y = Y + 0.02 * torch.randn_like(Y)
        return Y
