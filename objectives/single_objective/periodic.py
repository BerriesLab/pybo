from objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase
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
            ref_point=[-1.0],
            outcomes=[0],
            num_outcomes=1,
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            add_noise_to_gt=False,
            best_value=None,
        )

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        # return X ** 2 + 10 * (1 - torch.cos(2 * torch.pi * X))
        return torch.cos(2 * torch.pi * X)
