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

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        # Deterministic ground truth: there is no measurement here to be noisy.
        if noisy:
            raise ValueError(f"{type(self).__name__} declares no ground-truth "
                             f"noise. Run with --noise false.")
        return torch.stack([self._f1(X), self._f2(X)], dim=-1)

    # Both TNK constraints are conditions on (f1, f2), which is why they take the
    # objectives rather than X: a constraint is read off measurements, so whatever
    # the objectives report is what it has to be judged against.
    @staticmethod
    def _c1(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # Constraint 1: f1^2 + f2^2 - 1 - 0.1*cos(16*arctan(f1/f2)) >= 0
        # Re-arranged for BoTorch (val <= 0 is feasible):
        theta = torch.atan2(f1, f2)
        return 1.0 + 0.1 * torch.cos(16 * theta) - (f1.pow(2) + f2.pow(2))

    @staticmethod
    def _c2(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        # Constraint 2: (f1 - 0.5)^2 + (f2 - 0.5)^2 <= 0.5
        return (f1 - 0.5).pow(2) + (f2 - 0.5).pow(2) - 0.5

    def evaluate_true_constraint(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        # No refusal of its own: asking for noise reaches the objectives, and they
        # are the ones with nothing to give. Were a std ever added to them, these
        # would inherit it here instead of needing one.
        Y = self.evaluate_true_objective(X, noisy=noisy)
        c1 = self._c1(Y[..., 0], Y[..., 1])
        c2 = self._c2(Y[..., 0], Y[..., 1])
        return torch.stack([c1, c2], dim=-1)
