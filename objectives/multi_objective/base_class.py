from collections.abc import Callable

import torch
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.exceptions import BotorchTensorDimensionError, BotorchError, InputDataError
from botorch.utils.transforms import normalize_indices
from constraints.output_constraints import *


class MCObjectiveBase(ABC):

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            dim: int,
            num_objectives: int,
            num_constraints: int,
            num_trackers: int,
            obj_to_minimize: torch.Tensor | list[bool],
            bounds: torch.Tensor | list[float],
            ref_point: torch.Tensor | list[float],
            outcomes: list[int],
            num_outcomes: int,
            linear_equality_input_constraints: list[tuple[Tensor, Tensor, float]] | None,
            linear_inequality_input_constraints: list[tuple[Tensor, Tensor, float]] | None,
            nonlinear_inequality_input_constraints: list[tuple[Callable, bool]] | None,
            output_constraints: list[Callable] | None,
            max_hv: float | None = None,
            gt_noise_std: float | list[float] | None = None,
            add_noise_to_gt: bool = False,
            parameter_names: list[str] | None = None,
            objective_names: list[str] | None = None,
            constraint_names: list[str] | None = None,
            tracker_names: list[str] | None = None,
    ):
        super().__init__()

        # === CUDA attributes ===
        self.device = device
        self.dtype = dtype

        # === Objective Attributes ===
        self.dim = dim
        self.num_objectives = num_objectives
        self.num_constraints = num_constraints
        self.num_trackers = num_trackers
        self.obj_to_minimize = obj_to_minimize
        self.ref_point = ref_point
        self.bounds = bounds
        self.outcomes = outcomes
        self.num_outcomes = num_outcomes
        self.gt_noise_std = gt_noise_std
        self.add_noise_to_gt = add_noise_to_gt
        self.max_hv = max_hv
        self.parameter_names = parameter_names
        self.objective_names = objective_names
        self.constraint_names = constraint_names
        self.tracker_names = tracker_names

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
            if len(value) != self.num_objectives:
                raise ValueError("The length of obj_to_minimize must be equal to the number of objectives.")
            value = torch.tensor(value, dtype=torch.bool, device=self.device)
        elif isinstance(value, torch.Tensor):
            if value.size(0) != self.num_objectives:
                raise ValueError("The length of obj_to_minimize must be equal to the number of objectives.")
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
            if len(value) != self.num_objectives:
                raise ValueError("The number of objectives must match the dimensions of the reference point.")
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        elif isinstance(value, torch.Tensor):
            if value.shape[0] != self.num_objectives:
                raise ValueError("The number of objectives must match the dimensions of the reference point.")
            value = value.to(self.device, dtype=self.dtype)
        else:
            raise TypeError("ref_point must be a list of floats or a torch.Tensor")
        self._ref_point = value

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    @bounds.setter
    def bounds(self, value: list[tuple[float, float]] | torch.Tensor):
        # Convert list of tuples to tensor
        if not torch.is_tensor(value):
            value = torch.tensor(value, device=self._device, dtype=self.dtype)
            if value.shape != (self.dim, 2):
                raise InputDataError(
                    f"Expected list of {self.dim} (lb, ub) tuples, got shape {value.shape}"
                )
            self._bounds = value.transpose(0, 1)  # shape -> (2, dim)
        else:
            # Tensor input: expect shape (2, dim)
            if value.shape != (2, self.dim):
                raise InputDataError(
                    f"Expected tensor of shape (2, {self.dim}), got {value.shape}"
                )
            self._bounds = value.to(device=self.device, dtype=self.dtype)

    @property
    def outcomes(self) -> torch.Tensor:
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value: list[int] | torch.Tensor):
        """
        Indices of model outputs used as objectives. If None, use all model outputs.
        """
        if torch.is_tensor(value):
            value = value.tolist()
        elif isinstance(value, list):
            pass
        else:
            raise TypeError("outcomes must be a list of ints or a torch.Tensor")

        if len(value) < self.num_objectives:
            raise ValueError("The number of objectives must match the number of model outputs.")

        # if any(i < 0 for i in value):
        #     if self.num_outcomes is None:
        #         raise BotorchError(
        #             "num_outcomes is required if any outcomes are less than 0."
        #         )
        #     value = normalize_indices(value, self.num_outcomes)

        value = torch.tensor(value, device=self.device, dtype=torch.long)
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
    def linear_equality_input_constraints(self, values: list[tuple[Tensor, Tensor, float]]):
        r"""
        Linear constraints are passed as a list of tuples. Each tuple corresponds
        to a constraint, and includes 3 elements (indices, coefficients, rhs),
        Each tuple encodes an equality constraint of the form:
        \sum_i (X[indices[idx]] * coefficients[idx]) = rh
        """
        if values is not None:
            if not isinstance(values, list):
                raise TypeError("linear_equality_input_constraints must be a list of tuples")

            new_values = []
            for c in values:
                # Type checks
                if not (isinstance(c, tuple) and len(c) == 3):
                    raise TypeError("Each linear_equality_input_constraints item must be a tuple of 3 elements")
                if not isinstance(c[0], torch.Tensor):
                    raise TypeError("The 1st element must be a torch.Tensor")
                if not isinstance(c[1], torch.Tensor):
                    raise TypeError("The 2nd element must be a torch.Tensor")
                if not isinstance(c[2], (float, int)):
                    raise TypeError("The 3rd element must be a float or an integer.")

                indices_tensor = c[0].to(device=self.device, dtype=torch.long)
                coefficients_tensor = c[1].to(device=self.device, dtype=self.dtype)
                new_values.append((indices_tensor, coefficients_tensor, c[2]))

            self._linear_equality_input_constraints = new_values
        else:
            self._linear_equality_input_constraints = None

    @property
    def linear_inequality_input_constraints(self):
        return self._linear_inequality_input_constraints

    @linear_inequality_input_constraints.setter
    def linear_inequality_input_constraints(self, values: list[tuple[Tensor, Tensor, float]]):
        r"""
        Linear constraints are passed as a list of tuples. Each tuple corresponds
        to a constraint, and includes 3 elements (indices, coefficients, rhs).
        Each tuple encodes an inequality constraint of the form:
        \sum_i (X[indices[idx]] * coefficients[idx]) >= rh
        """
        if values is not None:
            if not isinstance(values, list):
                raise TypeError("linear_equality_input_constraints must be a list of tuples")

            new_values = []
            for c in values:
                # Type checks
                if not (isinstance(c, tuple) and len(c) == 3):
                    raise TypeError("Each linear_equality_input_constraints item must be a tuple of 3 elements")
                if not isinstance(c[0], torch.Tensor):
                    raise TypeError("The 1st element must be a torch.Tensor")
                if not isinstance(c[1], torch.Tensor):
                    raise TypeError("The 2nd element must be a torch.Tensor")
                if not isinstance(c[2], (float, int)):
                    raise TypeError("The 3rd element must be a float or an integer.")

                indices_tensor = c[0].to(device=self.device, dtype=torch.long)
                coefficients_tensor = c[1].to(device=self.device, dtype=self.dtype)
                new_values.append((indices_tensor, coefficients_tensor, c[2]))

            self._linear_inequality_input_constraints = new_values
        else:
            self._linear_inequality_input_constraints = None

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
        pass

    def evaluate_true_objective_with_noise(self, X: Tensor) -> Tensor:
        Y = self.evaluate_true_objective(X)
        Y = self.add_noise(Y)
        return Y

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        """
        Evaluate the true, unnegated constraint at the given input
        locations X. This method serves as the ground-truth evaluation of the
        problem and is typically used for benchmarking, visualization (e.g.,
        plotting the true Pareto front), or performance assessment of
        optimization algorithms.
        """
        pass

    def evaluate_true_constraint_with_noise(self, X: Tensor) -> Tensor:
        Y = self.evaluate_true_constraint(X)
        Y = self.add_noise(Y)
        return Y

    def evaluate_trackers(self, X: Tensor) -> Tensor:
        """
        Evaluate values to monitor but not to optimize.
        """
        pass

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
        # selected[..., self.negate] *= -1
        selected[..., self.obj_to_minimize] *= -1
        return selected


class MCSingleObjectiveBase(MCAcquisitionObjective, MCObjectiveBase, ABC):
    def __init__(self, *args, **kwargs):
        ABC.__init__(self)
        MCAcquisitionObjective.__init__(self)
        MCObjectiveBase.__init__(self, *args, **kwargs)


class MCMultiObjectiveBase(MCMultiOutputObjective, MCObjectiveBase, ABC):
    def __init__(self, *args, **kwargs):
        ABC.__init__(self)
        MCMultiOutputObjective.__init__(self)
        MCObjectiveBase.__init__(self, *args, **kwargs)
