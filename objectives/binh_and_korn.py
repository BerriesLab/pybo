import torch
from torch import Tensor
from objectives.base_class import MCMultiOutputBase
from constraints.output_constraints import Identity


class BinhAndKornMCMultiOutputObjective(MCMultiOutputBase):
    """ Two objective problem composed of the Binh and Korn functions.

    Notes:
        - Both functions are originally intended for minimization.
        - The constraints are on the input.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=2,
            num_objectives=2,
            num_constraints=0,
            obj_to_minimize=[True, True],
            bounds=[(0.0, 5.0), (0.0, 3.0)],
            ref_point=[80.0, 30.0],
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=None,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=[(self.c1, True), (self.c2, True)],
            output_constraints=[Identity(index=-1)],
            estimate_max_hv=False,
        )

    @staticmethod
    def _f1(X: torch.Tensor) -> torch.Tensor:
        return 4 * X[..., 0] ** 2 + 4 * X[..., 1] ** 2

    @staticmethod
    def _f2(X: torch.Tensor) -> torch.Tensor:
        return (X[..., 0] - 5) ** 2 + (X[..., 1] - 5) ** 2

    @staticmethod
    def c1(X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
        return ((X[..., 0] - 5) ** 2 + X[..., 1] ** 2) - 25

    @staticmethod
    def c2(X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
        return 7.7 - ((X[..., 0] - 8) ** 2 + (X[..., 1] + 3) ** 2)

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        f = torch.stack([f1, f2], dim=-1)
        f = super().evaluate_true_objective(f)
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
