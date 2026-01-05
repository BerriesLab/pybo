from objectives.multi_objective.base_class import MCObjectiveBase
import torch
from torch import Tensor


class WavePacketTestFunction(MCObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=1,
            num_objectives=1,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[False],
            bounds=[(-1.0, 1.0)],
            ref_point=[0.0],
            outcomes=[0],
            num_outcomes=1,
            gt_noise_std=0.0,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            add_noise_to_gt=False,
        )

        self.sigma = 0.1
        self.k0 = 5
        self.x0 = 0

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((X - self.x0) / self.sigma) ** 2)

    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * self.k0 * X)

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        f = f1 * f2
        return f

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        """ Transform Monte Carlo samples from the model's posterior according to the specified
        objective configuration. This method selects the relevant output dimensions (if `outcomes` are specified),
        and optionally applies negation if the objective is formulated as a minimization problem but needs to
        be maximized internally (as is common in acquisition functions like qNEHVI)."""
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected
