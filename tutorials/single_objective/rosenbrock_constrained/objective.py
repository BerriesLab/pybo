from objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase
import torch
from torch import Tensor


class RosenbrockConstrained(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=2,
            num_objectives=1,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[True],
            bounds=[(-2.0, 2.0), (-1.0, 3.0)],
            ref_point=None,
            outcomes=[0],
            num_outcomes=1,
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=[(self.disk_constraint, True)],
            output_constraints=None,
            add_noise_to_gt=False,
            best_value=0.0,
        )

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        # Use [..., 0] to get the first dimension for all points in the batch
        X0 = X[..., 0]
        X1 = X[..., 1]
        part1 = 100 * (X1 - X0 ** 2) ** 2
        part2 = (1 - X0) ** 2
        # Since X0 and X1 were shape (N,), the result is (N,)
        # BoTorch expects (N, 1), so we unsqueeze
        return (part1 + part2).unsqueeze(-1)

    @staticmethod
    def disk_constraint(X: Tensor) -> Tensor:
        # (x^2 + y^2) <= 2
        return 2.0 - (X[..., 0] ** 2 + X[..., 1] ** 2)
