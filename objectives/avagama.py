import torch
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.exceptions import BotorchTensorDimensionError, BotorchError, InputDataError
from botorch.utils.transforms import normalize_indices


class AvagamaMCMultiOutputObjective(MCMultiOutputObjective, ABC):
    """
    3x objectives:
        - Machining Time: Originally intended for minimization.
        - Electrode Wear: Originally intended for minimization.
        - Orbiting Time: Originally intended for values > 40 mins.

    3x parameters:
        - Maximum Current []
        - Pedestal Current []
        - Maximum Ramp Time [], expressed as a fraction of the ON time = 78 us

    Reference point:
        - Machining Time: 5 h * 60 min/h * 60s/min = 5h * 3600 s/h = 18000
        - Electrode Wear: 1000 um
        - Orbiting Time: 5 h * 60 min/h
    """

    def __init__(self, device: torch.device, dtype: torch.dtype,):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = 3
        self.num_objectives = 3
        self.num_constraints = 0
        self.negate = [True, True, True]
        self.ref_point = [10000, 200, 10000]
        self.noise_std: float or list[float] or None = None
        self.bounds = [(7.5, 15), (3, 7.5), (0.1 * 78, 78)]  # 2 x d: I_MAX, I_P, tau 78 us
        self.outcomes = [0, 1, 2]
        self.num_outcomes = 3
        self.max_hv = None

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
    def _machining_time(X: Tensor) -> Tensor:
        """ The identity function. """
        return X[..., 0]

    @staticmethod
    def _electrode_wear(X: Tensor) -> Tensor:
        """ The identity function. """
        return X[..., 1]

    @staticmethod
    def _orbiting_time(X: Tensor) -> Tensor:
        """ A penalty function that penalizes orbiting times shorter than 40 mins. """
        return X[..., 2]

    def _orbiting_time_penalty(self, X: Tensor) -> Tensor:
        orbiting_time = self._orbiting_time(X)
        penalty = torch.clamp(input=2400.0 - orbiting_time, min=0.0)
        return penalty

    # Optional — not required by MCMultiOutputObjective
    def evaluate_true(self, X: Tensor) -> Tensor:
        """ Evaluate the true (noise-free, unnegated) objective functions at the given input locations.
        This method returns the underlying objective values without any modifications such as noise addition or
        sign flipping. It serves as the ground-truth evaluation of the problem and is typically used for benchmarking,
        visualization (e.g., plotting the true Pareto front), or performance assessment of optimization algorithms.
        It should never be used for optimization as it does not take into account any possibly necessary sign flip. """
        machining_time = self._machining_time(X=X)
        electrode_wear = self._electrode_wear(X=X)
        orbiting_time = self._orbiting_time(X=X)
        return torch.stack([machining_time, electrode_wear, orbiting_time], dim=-1)

    # Optional — not required by MCMultiOutputObjective
    def add_noise(self, Y: Tensor) -> Tensor:
        """ A method to add noise to the observations. """
        if self.noise_std is not None:
            noise = self.noise_std.to(Y.device) * torch.randn_like(Y)
            return Y + noise
        return Y

    # Optional — not required by MCMultiOutputObjective
    def evaluate(self, X: Tensor, noise: bool = True) -> Tensor:
        """ Evaluate the objective function on input X, optionally adding noise and applying negation
        if specified. This method returns the version of the objective used for optimization. While this
        method is not used for optimization, it can be conveniently used to generate the initial
        population in test problems. """
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