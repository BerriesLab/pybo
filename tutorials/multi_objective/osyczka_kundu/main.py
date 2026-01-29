import torch
from objectives.base_class import MCMultiOutputBase
from torch import Tensor


class OsyczkaKundu(MCMultiOutputBase):
    r"""
    Two-objective problem with a set of linear inequality ineq_Y_con_cfg.
    ref: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=6,
            num_objectives=2,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=torch.tensor(
                [True, True]
            ),
            bounds=torch.tensor(
                [[0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                 [10.0, 10.0, 5.0, 6.0, 5.0, 10.0]]
            ),
            ref_point=torch.tensor([0.0, 160.0]),
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=None,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=[
                (torch.tensor([0, 1], dtype=torch.long), torch.tensor([1.0, 1.0], dtype=torch.float), 2.0),
                (torch.tensor([0, 1], dtype=torch.long), torch.tensor([-1.0, -1.0], dtype=torch.float), -6.0),
                (torch.tensor([0, 1], dtype=torch.long), torch.tensor([1.0, -1.0], dtype=torch.float), -2.0),
                (torch.tensor([0, 1], dtype=torch.long), torch.tensor([-1.0, 3.0], dtype=torch.float), -2.0),
            ],
            nonlinear_inequality_input_constraints=[
                (self._nonlinear_c1, True),
                (self._nonlinear_c2, True)
            ],
            output_constraints=None,
        )

    @staticmethod
    def _f1(X: Tensor) -> Tensor:
        return (
                - 25 * (X[..., 0] - 2).pow(2)
                - (X[..., 1] - 2).pow(2)
                - (X[..., 2] - 1).pow(2)
                - (X[..., 3] - 4).pow(2)
                - (X[..., 4] - 1).pow(2)
        )

    @staticmethod
    def _f2(X: Tensor) -> Tensor:
        return (
                + X[..., 0].pow(2)
                + X[..., 1].pow(2)
                + X[..., 2].pow(2)
                + X[..., 3].pow(2)
                + X[..., 4].pow(2)
                + X[..., 5].pow(2)
        )

    @staticmethod
    def _nonlinear_c1(X: Tensor) -> Tensor:
        return 4 - (X[..., 2] - 3).pow(2) - X[..., 3]

    @staticmethod
    def _nonlinear_c2(X: Tensor) -> Tensor:
        return (X[..., 4] - 3).pow(2) + X[..., 5] - 4

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
