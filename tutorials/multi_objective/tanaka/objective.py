import torch
import math
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.constraints.output_constraints import Identity
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg


class Tanaka(MCMultiObjectiveBase):
    r"""
    TNK (Tanaka) test problem.
    Two objectives, two variables, and two non-linear constraints.
    The Pareto front is disconnected and non-convex.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(0.0, math.pi)),
                ParCfg(bounds=(0.0, math.pi)),
            ],
            obj_cfg=[
                ObjCfg(to_minimize=True, ref_point=1.5),
                ObjCfg(to_minimize=True, ref_point=1.5)
            ],
            ineq_Y_con_cfg=[
                IneqYConCfg(f=Identity(index=-1)),
                IneqYConCfg(f=Identity(index=-2))
            ],
        )

    @staticmethod
    def _f1(X: torch.Tensor) -> torch.Tensor:
        return X[..., 0]

    @staticmethod
    def _f2(X: torch.Tensor) -> torch.Tensor:
        return X[..., 1]

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._f1(X), self._f2(X)], dim=-1)

    @staticmethod
    def _c1(X: torch.Tensor) -> torch.Tensor:
        x0 = X[..., 0]
        x1 = X[..., 1]
        # Constraint 1: x0^2 + x1^2 - 1 - 0.1*cos(16*arctan(x0/x1)) >= 0
        # Re-arranged for BoTorch (val <= 0 is feasible):
        theta = torch.atan2(x0, x1)
        return 1.0 + 0.1 * torch.cos(16 * theta) - (x0.pow(2) + x1.pow(2))

    @staticmethod
    def _c2(X: torch.Tensor) -> torch.Tensor:
        x0 = X[..., 0]
        x1 = X[..., 1]
        # Constraint 2: (x0 - 0.5)^2 + (x1 - 0.5)^2 <= 0.5
        return (x0 - 0.5).pow(2) + (x1 - 0.5).pow(2) - 0.5

    def evaluate_true_constraint(self, X: torch.Tensor) -> torch.Tensor:
        c1 = self._c1(X)
        c2 = self._c2(X)
        return torch.stack([c1, c2], dim=-1)
