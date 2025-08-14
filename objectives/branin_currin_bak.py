import torch
import math
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.exceptions import BotorchTensorDimensionError, BotorchError, InputDataError
from botorch.utils.transforms import normalize_indices


class BraninCurrinMCMultiOutputObjective(MCMultiOutputObjective, ABC):
    """
    Two objective problem composed of the Branin and Currin functions.

    Note:
    - Branin Function: Originally intended for minimization.
    - Currin Exponential Function: Originally intended for maximization.
    - Both the Branin and Currin functions are positive on their domain.

    For testing purposes, and following BoTorch authors implementation,
    both test functions are intended for minimization, against their original
    nature. Therefore, here it is important to negate both.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype,):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = 2
        self.num_objectives = 2
        self.num_constraints = 0
        self.negate = [True, True]
        self.ref_point = 18.0, 6.0
        self.noise_std: float or list[float] or None = None
        self.bounds = [(0.0, 1.0), (0.0, 1.0)]
        self.outcomes = [0, 1]
        self.num_outcomes = 2
        self.max_hv = 59.36011874867746  # this is approximated using NSGA-II

        # Bounds validation and registration
        if len(self.bounds) != self.dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {len(self.bounds)=}."
            )

        # Outcomes validation and index conversion
        if self.outcomes is not None:
            if len(self.outcomes) < 2:
                raise BotorchTensorDimensionError("Must specify at least two outcomes for MOO.")
            if any(i < 0 for i in self.outcomes):
                if self._num_outcomes is None:
                    raise BotorchError("num_outcomes is required if any outcomes are less than 0.")
                self._outcomes = normalize_indices(self.outcomes, self.num_outcomes)

        # Noise validation and conversion
        if self.noise_std is not None:
            if isinstance(self.noise_std, list):
                if len(self.noise_std) != self.num_objectives:
                    raise ValueError(
                        f"noise_std list must have length {self.num_objectives}, "
                        f"but got {len(self.noise_std)}"
                    )
                self.noise_std = torch.tensor(self.noise_std)
            elif isinstance(self.noise_std, (float, int)):
                self.noise_std = torch.tensor([self.noise_std] * self.num_objectives)
            else:
                raise TypeError("noise_std must be a float, int, or list of floats.")
        else:
            self.noise_std = None

        # Create tensors
        self.negate = torch.tensor(data=self.negate, device=self.device, dtype=torch.bool)
        self.ref_point = torch.tensor(data=self.ref_point, device=self.device, dtype=self.dtype)
        self.outcomes = torch.tensor(self.outcomes, device=self.device)
        self.bounds = torch.tensor(self.bounds).transpose(-1, -2)

    @staticmethod
    def _branin(X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi**2) * X[..., 0].pow(2)
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

    # Optional — not required by MCMultiOutputObjective
    def evaluate_true(self, X: Tensor) -> Tensor:
        """ Evaluate the true (noise-free, unnegated) objective functions at the given input locations.
        This method returns the underlying objective values without any modifications such as noise addition or
        sign flipping. It serves as the ground-truth evaluation of the problem and is typically used for benchmarking,
        visualization (e.g., plotting the true Pareto front), or performance assessment of optimization algorithms.
        It should never be used for optimization as it does not take into account any possibly necessary sign flip. """
        # branin rescaled with inputs to [0,1]^2
        branin = self._rescaled_branin(X=X)
        currin = self._currin(X=X)
        return torch.stack([branin, currin], dim=-1)

    # Optional — not required by MCMultiOutputObjective
    def add_noise(self, Y: Tensor) -> Tensor:
        """ A method to add noise to the observations. """
        if self.noise_std is not None:
            noise = self.noise_std.to(Y.device) * torch.randn_like(Y)
            return Y + noise
        return Y

    # Optional — not required by MCMultiOutputObjective
    def evaluate_in_optimization_space(self, X: Tensor, noise: bool = True) -> Tensor:
        """ Evaluate the objective function (w/o noise) on input X in the optimization space.
        It can be conveniently used to evaluate the function in the optimization space for e.g.,
        debugging purposes. """
        Y = self.evaluate_true(X)
        Y = self.add_noise(Y) if noise else Y
        Y[..., self.negate] *= -1
        return Y

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        """ Transform Monte Carlo samples from the model's posterior according to the specified
        objective configuration. This method selects the relevant output dimensions (if `outcomes` are specified),
        and optionally applies negation if the objective is formulated as a minimization problem but needs to
        be maximized internally (as is common in acquisition functions like qNEHVI)."""
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.negate] *= -1
        return selected

    # TODO: implement the following as methods or properties
    def linear_equality_constraints(self) -> Tensor:
        raise ValueError("No linear equality constraints for this objective.")

    def linear_inequality_constraints(self) -> Tensor:
        raise ValueError("No linear inequality constraints for this objective.")

    def nonlinear_inequality_constraints(self) -> Tensor:
        raise ValueError("No nonlinear inequality constraints for this objective.")
