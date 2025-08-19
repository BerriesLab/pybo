from collections.abc import Callable
from botorch.utils.multi_objective import is_non_dominated, Hypervolume
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.exceptions import BotorchTensorDimensionError, BotorchError, InputDataError
from botorch.utils.transforms import normalize_indices
from constraints.output_constraints import *


class MCMultiOutputBase(MCMultiOutputObjective, ABC):
    """
    Base class for Monte Carlo multi-output objectives. Read the description
    of each attribute's setter to understand their data type and structure.
    """

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            dim: int,
            num_objectives: int,
            num_constraints: int,
            obj_to_minimize: list[bool],
            bounds: list[float],
            ref_point: list[float],
            outcomes: list[int],
            num_outcomes: int,
            gt_noise_std: float | list[float],
            max_hv: float | None,
            linear_equality_input_constraints: list[tuple[Tensor, Tensor, float]] | None,
            linear_inequality_input_constraints: list[tuple[Tensor, Tensor, float]] | None,
            nonlinear_inequality_input_constraints: list[tuple[Callable, bool]] | None,
            output_constraints: list[Callable] | None,
            add_noise_to_gt: bool = False,
            estimate_max_hv: bool = False,
            parameter_names: list[str] | None = None,
            objective_names: list[str] | None = None,
            constraint_names: list[str] | None = None,

    ):

        super().__init__()
        # === CUDA attributes ===
        self.device = device
        self.dtype = dtype

        # === Objective Attributes ===
        self.dim = dim
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.obj_to_minimize = obj_to_minimize
        self.ref_point = ref_point
        self.bounds = bounds
        self.outcomes = outcomes
        self.num_outcomes = num_outcomes
        self.gt_noise_std = gt_noise_std
        self.add_noise_to_gt = add_noise_to_gt
        self.max_hv = max_hv
        if estimate_max_hv is True:
            self.estimate_max_hv()
        self.parameter_names = parameter_names
        self.objective_names = objective_names
        self.constraint_names = constraint_names

        # === Constraints ===
        self.linear_equality_input_constraints = linear_equality_input_constraints
        self.linear_inequality_input_constraints = linear_inequality_input_constraints
        self.nonlinear_inequality_input_constraints = nonlinear_inequality_input_constraints
        self.output_constraints = output_constraints

    # === CUDA Properties ===
    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        if not isinstance(device, torch.device):
            raise ValueError("Device must be of type torch.device")
        self._device = device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype):
        if not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be of type torch.dtype")
        self._dtype = dtype

    # === Objective Properties ===
    @property
    def dim(self) -> int:
        return self._dim

    @dim.setter
    def dim(self, dim: int):
        if not isinstance(dim, int) or dim < 0:
            raise ValueError("'dim' must be a positive integer")
        self._dim = dim

    @property
    def num_objectives(self) -> int:
        return self._num_objectives

    @num_objectives.setter
    def num_objectives(self, num_objectives: int):
        if not isinstance(num_objectives, int) or num_objectives < 0:
            raise ValueError("'num_objectives' must be a positive integer")
        self._num_objectives = num_objectives

    @property
    def num_constraints(self) -> int:
        return self._num_constraints

    @num_constraints.setter
    def num_constraints(self, num_constraints: int):
        if not isinstance(num_constraints, int) or num_constraints < 0:
            raise ValueError("'num_constraints' must be a positive integer")
        self._num_constraints = num_constraints

    @property
    def obj_to_minimize(self) -> Tensor:
        return self._obj_to_minimize

    @obj_to_minimize.setter
    def obj_to_minimize(self, value: list[bool] | torch.Tensor):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=torch.bool, device=self.device)
        elif isinstance(value, torch.Tensor):
            value = value.to(self.device, dtype=torch.bool)
        else:
            raise TypeError("negate must be a list of bools or a torch.Tensor")
        self._obj_to_minimize = value

    @property
    def ref_point(self) -> Tensor:
        return self._ref_point

    @ref_point.setter
    def ref_point(self, value: list[float] | torch.Tensor):
        if isinstance(value, list):
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        elif isinstance(value, torch.Tensor):
            value = value.to(self.device, dtype=self.dtype)
        else:
            raise TypeError("ref_point must be a list of floats or a torch.Tensor")
        self._ref_point = value

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @bounds.setter
    def bounds(self, value: list[tuple[float, float]] | torch.Tensor):
        # value can either be a list of tuples of floats, where the 1st and the
        # 2nd floats of the n-th tuple are the lower and upper bounds, respectively,
        # of the input n-th dimension; or a tensor of shape 2 x n, where the 1st
        # row includes the lower bounds and the 2nd row the upper bounds.
        if len(value) != self.dim:
            raise InputDataError(
                f"Expected bounds to match dimensionality of the domain. "
                f"Got dim={self.dim}, bounds_len={len(value)}"
            )
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=self.dtype)
        self._bounds = value.transpose(-1, -2)

    @property
    def outcomes(self) -> torch.Tensor:
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: list[int] | torch.Tensor | None):
        if value is not None:
            if len(value) < 2:
                raise BotorchTensorDimensionError(
                    "Must specify at least two outcomes for MOO."
                )
            if any(i < 0 for i in value):
                if self.num_outcomes is None:
                    raise BotorchError(
                        "num_outcomes is required if any outcomes are less than 0."
                    )
                value = normalize_indices(value, self.num_outcomes)

            if not torch.is_tensor(value):
                value = torch.tensor(value, device=self.device)
        self._outcomes = value

    @property
    def num_outcomes(self) -> int | None:
        return self._num_outcomes

    @num_outcomes.setter
    def num_outcomes(self, value: int | None):
        self._num_outcomes = value

    @property
    def gt_noise_std(self) -> torch.Tensor | None:
        return self._noise_std

    @gt_noise_std.setter
    def gt_noise_std(self, value: float | int | list[float] | None):
        if value is None:
            self._noise_std = None
            return

        if isinstance(value, list):
            if len(value) != self.num_objectives:
                raise ValueError(
                    f"noise_std list must have length {self.num_objectives}, "
                    f"but got {len(value)}"
                )
            value = torch.tensor(value, dtype=self.dtype)
        elif isinstance(value, (float, int)):
            value = torch.tensor([value] * self.num_objectives, dtype=self.dtype)
        else:
            raise TypeError("noise_std must be a float, int, or list of floats.")

        self._noise_std = value

    # == Constraints Properties ===
    @property
    def linear_equality_input_constraints(self):
        return self._linear_equality_input_constraints

    @linear_equality_input_constraints.setter
    def linear_equality_input_constraints(self, value):
        r"""
        Linear constraints are passed as a list of tuples. Each tuple corresponds
        to a constraint, and includes 3 elements (indices, coefficients, rhs),
        Each tuple encodes an inequality constraint of the form:
        \sum_i (X[indices[i]] * coefficients[i]) = rh
        """
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("linear_equality_input_constraints must be a list of tuples")
            for item in value:
                if not (isinstance(item, tuple) and len(item) == 3 and
                        isinstance(item[0], torch.Tensor) and
                        isinstance(item[1], torch.Tensor) and
                        isinstance(item[2], float)):
                    raise TypeError(
                        "Each linear_equality_input_constraints item must be a tuple of (Tensor, Tensor, float)"
                    )
        self._linear_equality_input_constraints = value

    @property
    def linear_inequality_input_constraints(self):
        return self._linear_inequality_input_constraints

    @linear_inequality_input_constraints.setter
    def linear_inequality_input_constraints(self, value):
        r"""
        Linear constraints are passed as a list of tuples. Each tuple corresponds
        to a constraint, and includes 3 elements (indices, coefficients, rhs).
        Each tuple encodes an inequality constraint of the form:
        \sum_i (X[indices[i]] * coefficients[i]) >= rh
        """
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("linear_inequality_input_constraints must be a list of tuples")
            for item in value:
                if not (isinstance(item, tuple) and len(item) == 3 and
                        isinstance(item[0], torch.Tensor) and
                        isinstance(item[1], torch.Tensor) and
                        isinstance(item[2], float)):
                    raise TypeError(
                        "Each linear_inequality_input_constraints item must be a tuple of (Tensor, Tensor, float)"
                    )
        self._linear_inequality_input_constraints = value

    @property
    def nonlinear_inequality_input_constraints(self):
        return self._nonlinear_inequality_input_constraints

    @nonlinear_inequality_input_constraints.setter
    def nonlinear_inequality_input_constraints(self, value):
        r"""
        A list of tuples representing the nonlinear
        inequality constraints. The first element in the tuple is a callable
        representing a constraint of the form `callable(x) >= 0`. In the case of an
        intra-point constraint (single candidate or q=1), `callable()` takes in a
        one-dimensional tensor of shape `d` and returns a scalar. In the case of an
        inter-point constraint (multiple candidates or q >1), `callable()` takes a
        two-dimensional tensor of shape `q x d` and again returns a scalar.
        The second element is a boolean, indicating if it is an
        intra-point or inter-point constraint (`True` for intra-point. `False` for
        inter-point). For more information on intra-point vs. inter-point
        constraints, see the docstring of the `inequality_constraints` argument to
        `optimize_acqf()`. The constraints will later be passed to the scipy
        solver. You need to pass in `batch_initial_conditions` in this case.
        Using non-linear inequality constraints also requires that `batch_limit`
        is set to 1, which will be done automatically if not specified in
        `options`.
        """
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("nonlinear_inequality_input_constraints must be a list of tuples")
            for item in value:
                if not (isinstance(item, tuple) and len(item) == 2 and callable(item[0]) and isinstance(item[1],
                                                                                                        bool)):
                    raise TypeError(
                        "Each nonlinear_inequality_input_constraints item must be a tuple of (Callable, bool)"
                    )
        self._nonlinear_inequality_input_constraints = value

    # === Output Constraints ===
    @property
    def output_constraints(self):
        return self._output_constraints

    @output_constraints.setter
    def output_constraints(self, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("output_constraints must be a list of Callables")
            for item in value:
                if not callable(item):
                    raise TypeError("Each output_constraint must be callable")
        self._output_constraints = value

    # === GROUND TRUTH METHODS ===
    def evaluate_true_objective(self, X: Tensor) -> Tensor:
        """
        Evaluate the true, unnegated objective function at the given input
        locations X. This method serves as the ground-truth evaluation of the
        problem and is typically used for benchmarking, visualization (e.g.,
        plotting the true Pareto front), or performance assessment of
        optimization algorithms.
        """
        # X = f(X)...
        if self.add_noise_to_gt and self.gt_noise_std is not None:
            X = self.add_noise(X)
        return X

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        """
        Evaluate the true, unnegated constraint at the given input
        locations X. This method serves as the ground-truth evaluation of the
        problem and is typically used for benchmarking, visualization (e.g.,
        plotting the true Pareto front), or performance assessment of
        optimization algorithms.
        """
        # X = f(X)...
        if self.add_noise_to_gt and self.gt_noise_std is not None:
            X = self.add_noise(X)
        return X

    def evaluate_true_slack(self, X: Tensor, slack: float = 0) -> Tensor:
        """
        Evaluate the relaxed constraint at X, where "slack >= 0" allows
        "c(X) <= slack" instead of the strict "c(X) <= 0".
        Returns "c(X) - slack", which is <= 0 when feasible.
        """
        if slack < 0:
            raise ValueError("slack must be positive")
        return self.evaluate_true_constraint(X=X) - slack

    def add_noise(self, Y: Tensor) -> Tensor:
        """
        A method to add noise to the observations.
        """
        if self.gt_noise_std is None:
            raise ValueError("noise_std is required to add_noise.")
        noise = self._noise_std.to(Y.device) * torch.randn_like(Y)
        return Y + noise

    def estimate_max_hv(self, n_samples: int = 1_000, verbose=True):
        """ Estimates the maximum theoretical hypervolume using Sobol-based Monte Carlo sampling.
            Parameters:
                - n_samples (int): Number of MC samples for objective space.
                - verbose (bool): If True, prints progress and results.
            Sets:
                - self._max_hv: Estimated maximum hypervolume value.
            """

        if verbose:
            print("Estimating maximum theoretical hypervolume... ", end="")

        # 1. Sample input space to get Pareto front
        lb, ub = self.bounds[0], self.bounds[1]
        X = (ub - lb) * torch.rand(n_samples, self.dim, device=self.device, dtype=self.dtype) + lb

        # 2. Evaluate and filter feasible points only if constraints exist
        Y = self.evaluate_true_objective(X)
        if self.num_constraints > 0:
            constraint_slack = self.evaluate_true_constraint(X)  # shape: (n_samples, num_constraints)
            feasible_mask = constraint_slack <= 0  # all constraints satisfied
            feasible_mask = feasible_mask.all(dim=-1)
            feasible_Y = Y[feasible_mask]
        else:
            feasible_Y = Y
        if feasible_Y.shape[0] == 0:
            raise ValueError("No feasible samples found to estimate hypervolume.")

        # 3. Work in maximization space: flip only objectives marked as "negate"
        feasible_Y_max = feasible_Y.clone()
        feasible_Y_max[..., self.obj_to_minimize] *= -1

        # 4. Compute Pareto front among feasible samples (in maximization space)
        pareto_mask = is_non_dominated(feasible_Y_max)
        pareto_front_max = feasible_Y_max[pareto_mask]

        # 5. Compute hypervolume in maximization space
        ref_point_max = self.ref_point.clone()
        ref_point_max[..., self.obj_to_minimize] *= -1
        hv = Hypervolume(ref_point_max).compute(pareto_front_max)
        self.max_hv = hv

        if verbose:
            print(f"{self.max_hv:.4f}")

    # === MONTE CARLO METHODS ===
    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        """
        Transform Monte Carlo samples from the model's posterior according
        to the specified objective configuration. This method selects the
        relevant output dimensions (if `outcomes` are specified), and optionally
        applies negation if the objective is formulated as a minimization
        problem but needs to be maximized internally.
        """
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.negate] *= -1
        return selected
