import torch
from torch import Tensor
from objectives.base_class import MCMultiObjectiveBase


class BinhAndKornMCMultiOutputObjective(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions.

    Notes:
        - Both functions are originally intended for minimization.
        - The non-linear constraints must be cast in the form tuple[callable(x) >= 0, bool].
        The first element in the tuple is a callable representing a constraint of the form `callable(x) >= 0`.
        In case of an intra-point constraint, "callable()" takes in a one-dimensional tensor of
        shape "d" and returns a scalar. In case of an inter-point constraint, "callable()"
        takes a two-dimensional tensor of shape "q x d" and again returns a scalar. The second
        element is a boolean, indicating if it is an intra-point or inter-point constraint
        ("True" for intra-point. "False" for inter-point).
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=2,
            num_objectives=2,
            num_constraints=0,
            num_trackers=0,
            obj_to_minimize=[True, True],
            bounds=torch.tensor(
                [[0.0, 0.0],
                 [5.0, 3.0]],
            ),
            ref_point=torch.tensor(
                [130.0, 50.0]
            ),
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=None,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=[
                (self._input_c1, True),
                (self._input_c2, True)
            ],
            output_constraints=None,
        )

    @staticmethod
    def _f1(X: torch.Tensor) -> torch.Tensor:
        return 4 * X[..., 0] ** 2 + 4 * X[..., 1] ** 2

    @staticmethod
    def _f2(X: torch.Tensor) -> torch.Tensor:
        return (X[..., 0] - 5) ** 2 + (X[..., 1] - 5) ** 2

    @staticmethod
    def _input_c1(X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
        return 25 - ((X[..., 0] - 5) ** 2 + X[..., 1] ** 2)

    @staticmethod
    def _input_c2(X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
        return (X[..., 0] - 8) ** 2 + (X[..., 1] + 3) ** 2 - 7.7

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        return torch.stack([f1, f2], dim=-1)

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
