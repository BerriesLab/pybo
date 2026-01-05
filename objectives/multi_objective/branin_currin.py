import torch
import math
from objectives.multi_objective.base_class import MCMultiOutputBase
from torch import Tensor


class BraninCurrinMCMultiOutputObjective(MCMultiOutputBase):
    """
    Two objective problem composed of the Branin and Currin functions.

    Note:
    - Branin Function: Originally intended for minimization.
    - Currin Exponential Function: Originally intended for maximization.
    - Both the Branin and Currin functions are positive on their domain.

    For testing purposes, and following BoTorch authors implementation,
    both test functions are intended for minimization, against their original
    nature.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=2,
            num_objectives=2,
            objective_names=[
                "Branin",
                "Currin"
            ],
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[True, True],
            bounds=[(0.0, 1.0), (0.0, 1.0)],
            ref_point=[18.0, 6.0],
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=59.36011874867746,  # this is approximated using NSGA-II
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
        )

    @staticmethod
    def _branin(X: Tensor) -> Tensor:
        t1 = (
                X[..., 1]
                - 5.1 / (4 * math.pi ** 2) * X[..., 0].pow(2)
                + 5 / math.pi * X[..., 0]
                - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1.pow(2) + t2 + 10

    def _rescaled_branin(self, X: Tensor) -> Tensor:
        """ For a visual reference visit: https://www.sfu.ca/~ssurjano/branin.html"""
        # return to Branin bounds
        x_0 = 15 * X[..., 0] - 5
        x_1 = 15 * X[..., 1]
        return self._branin(torch.stack([x_0, x_1], dim=-1))

    @staticmethod
    def _currin(X: Tensor) -> Tensor:
        """ For a visual reference visit: https://www.sfu.ca/~ssurjano/curretal88exp.html """
        x_0 = X[..., 0]
        x_1 = X[..., 1]
        factor1 = 1 - torch.exp(-1 / (2 * x_1))
        numer = 2300 * x_0.pow(3) + 1900 * x_0.pow(2) + 2092 * x_0 + 60
        denom = 100 * x_0.pow(3) + 500 * x_0.pow(2) + 4 * x_0 + 20
        return factor1 * numer / denom

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor | None:
        branin = self._rescaled_branin(X=X)
        currin = self._currin(X=X)
        f = torch.stack([branin, currin], dim=-1)
        return f

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected
