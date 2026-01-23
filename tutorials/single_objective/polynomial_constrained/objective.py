from constraints.output_constraints import *
from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor


class PolynomialConstrained(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=1,
            num_objectives=1,
            num_constraints=1,
            num_trackers=0,
            obj_to_minimize=[False],
            bounds=[(0.0, 1.0)],
            outcomes=[0],
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=[Identity(-1)],
            add_noise_to_gt=False,
            best_value=None,
        )

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        return (X - 0.30) ** 2 + 0.08 * torch.sin(12.0 * X) + 0.02 * torch.cos(25.0 * X)

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        return self.evaluate_true_objective(X) - 0.08
