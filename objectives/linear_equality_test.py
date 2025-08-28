import torch
from objectives.base_class import MCMultiOutputBase
from torch import Tensor


class LinearEqualityTestProblem(MCMultiOutputBase):
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
            dim=2,
            num_objectives=2,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=torch.tensor(
                [True, True]
            ),
            bounds=torch.tensor(
                [[0.0, 0.0],
                 [1.0, 1.0]]
            ),
            ref_point=torch.tensor([2.1, 2.1]),
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=None,
            linear_equality_input_constraints=[(
                torch.tensor([0, 1], dtype=torch.long),  # Indices
                torch.tensor([-1.0, -2.0], dtype=torch.float),  # Coefficients
                -1.0,  # RHS
            )],
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
        )

    @staticmethod
    def _f1(X: Tensor) -> Tensor:
        return (X[..., 0] - 1).pow(2) + (X[..., 1]).pow(2)

    @staticmethod
    def _f2(X: Tensor) -> Tensor:
        return (X[..., 0]).pow(2) + (X[..., 1] - 1).pow(2)

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor | None:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        f = torch.stack([f1, f2], dim=-1)
        return f

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected
