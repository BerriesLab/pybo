from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor


class Periodic(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=1,
            num_objectives=1,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[True],
            bounds=[(-3.0, 3.0)],
            outcomes=[0],
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            add_noise_to_gt=False,
            best_value=None,
        )

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        return torch.sin(2 * torch.pi * 2 * X) + 0.5 * torch.sin(4 * torch.pi * 2 * X)
