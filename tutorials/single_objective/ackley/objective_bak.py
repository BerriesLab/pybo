from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor
from objectives.variable_registry import Cfg, VariableRegistry


class Ackley(MCSingleObjectiveBase):
    class Obj(VariableRegistry):
        ACKLEY = Cfg(label="Ackley", index=0, bounds=(-5.0, 0.0), dtype=torch.float64, to_minimize=True,
                     best_value=0)

    class Par(VariableRegistry):
        P1 = Cfg(label="P1", index=0, bounds=(-5.0, 5.0), dtype=torch.float64)
        P2 = Cfg(label="P2", index=1, bounds=(-5.0, 5.0), dtype=torch.float64)

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            # dim=2,
            # num_objectives=1,
            # num_constraints=0,
            # num_trackers=0,
            # obj_to_minimize=[True],
            # bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            # outcomes=[0],
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            add_noise_to_gt=False,
            # best_value=0,
        )

    @staticmethod
    def term1(X: Tensor) -> Tensor:
        x1 = X[:, 0]
        x2 = X[:, 1]
        arg = -0.2 * torch.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
        return -20 * torch.exp(arg)

    @staticmethod
    def term2(X: Tensor) -> Tensor:
        x1 = X[:, 0]
        x2 = X[:, 1]
        arg = 0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))
        return - torch.exp(arg)

    @staticmethod
    def term3() -> Tensor:
        return torch.e + 20

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        return (self.term1(X) + self.term2(X) + self.term3()).unsqueeze(-1)
