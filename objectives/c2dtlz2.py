import torch
import math

from sympy import false
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.exceptions import BotorchTensorDimensionError, BotorchError, InputDataError
from botorch.utils.transforms import normalize_indices


class C2DTLZ2MCMultiOutputObjective(MCMultiOutputObjective, ABC):
    r"""
    DLTZ2 test problem.

    d-dimensional problem evaluated on `[0, 1]^d`:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{i=m}^{d-1} (x_i - 0.5)^2

    The pareto front is given by the unit hypersphere \sum{i} f_i^2 = 1.
    Note: the pareto front is completely concave. The goal is to minimize
    both objectives.

    The constraint computes the minimum distance to two types of structures in objective space:
    Notes: negative constraint values imply feasibility in botorch.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype,):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = 4
        self.num_objectives = 2
        self.num_constraints = 1
        self.negate = [True, True]
        self.ref_point = [1.1 for _ in range(self.num_objectives)]
        self.noise_std: float or list[float] or None = None
        self.bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        self.outcomes = [0, 1]
        self.num_outcomes = 2
        self.max_hv = 0.3996406303723544

        self.k = self.dim - self.num_objectives + 1
        self._r = 0.2

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

    # Optional — not required by MCMultiOutputObjective
    def evaluate_true(self, X: Tensor) -> Tensor:
        """ Evaluate the true (noise-free, unnegated) objective functions at the given input locations.
        This method returns the underlying objective values without any modifications such as noise addition or
        sign flipping. It serves as the ground-truth evaluation of the problem and is typically used for benchmarking,
        visualization (e.g., plotting the true Pareto front), or performance assessment of optimization algorithms.
        It should never be used for optimization as it does not take into account any possibly necessary sign flip. """
        X_m = X[..., -self.k:]
        g_X = (X_m - 0.5).pow(2).sum(dim=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_plus1.clone()
            f_i *= torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        """Evaluate the constraint slack (w/o observation noise) on a set of points.
        This constraint has been changed with respect to the original problem to account for
        the fact that negative values imply feasibility in botorch"""
        if X.ndim > 2:
            raise NotImplementedError("Batch X is not supported.")
        f_X = self.evaluate_true(X)
        term1 = (f_X - 1).pow(2)
        mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
        indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
        indexer = indices[mask].view(f_X.shape[1], f_X.shape[-1] - 1)
        term2_inner = (
            f_X.unsqueeze(1)
            .expand(f_X.shape[0], f_X.shape[-1], f_X.shape[-1])
            .gather(dim=-1, index=indexer.repeat(f_X.shape[0], 1, 1))
        )
        term2 = (term2_inner.pow(2) - self._r**2).sum(dim=-1)
        min1 = (term1 + term2).min(dim=-1).values
        min2 = ((f_X - 1 / math.sqrt(f_X.shape[-1])).pow(2) - self._r**2).sum(dim=-1)
        slack_true = -torch.min(min1, min2).unsqueeze(-1)
        return -slack_true

    # Optional — not required by MCMultiOutputObjective
    def add_noise(self, Y: Tensor) -> Tensor:
        """ A method to add noise to the observations. """
        if self.noise_std is not None:
            noise = self.noise_std.to(Y.device) * torch.randn_like(Y)
            return Y + noise
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

    def evaluate_slack(self):
        return

    def is_feasible(self):
        return