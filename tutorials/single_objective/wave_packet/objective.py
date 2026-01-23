from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor


class WavePacket(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=1,
            num_objectives=1,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[True],
            bounds=[(-1.0, 1.0)],
            outcomes=[0],
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            add_noise_to_gt=False,
        )

        self.p = 1 / 2
        self.sigma = 0.4
        self.k0 = 2 * torch.pi / self.p
        self.x0 = 0

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((X - self.x0) / self.sigma) ** 2)

    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.k0 * X)

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        f = f1 * f2
        return f
