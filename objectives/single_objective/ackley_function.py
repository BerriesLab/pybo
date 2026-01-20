from objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase
import torch
from torch import Tensor


class Ackley(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=2,
            num_objectives=1,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[True],
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            ref_point=[-2.0],
            outcomes=[0],
            num_outcomes=1,
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            add_noise_to_gt=False,
            best_value=0,
        )

    @staticmethod
    def term1(X: Tensor) -> Tensor:
        arg = -0.2 * torch.sqrt(0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2))
        return -20 * torch.exp(arg)

    @staticmethod
    def term2(X: Tensor) -> Tensor:
        arg = 0.5 * (torch.cos(2 * torch.pi * X[:, 0]) + torch.cos(2 * torch.pi * X[:, 1]))
        return - torch.exp(arg)

    @staticmethod
    def term3() -> Tensor:
        return torch.e + 20

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        return (self.term1(X) + self.term2(X) + self.term3()).unsqueeze(-1)
