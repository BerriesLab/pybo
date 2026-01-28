import inspect
from collections.abc import Callable
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.exceptions import InputDataError
from constraints.output_constraints import *
from objectives.variable_registry import VariableRegistry


class MCObjectiveBase(ABC):
    class Obj(VariableRegistry):
        pass

    class Par(VariableRegistry):
        pass

    class Trk(VariableRegistry):
        pass

    class Con(VariableRegistry):
        pass

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            linear_equality_input_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
            linear_inequality_input_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
            nonlinear_inequality_input_constraints: list[tuple[Callable, bool]] | None = None,
            output_constraints: list[Callable] | None = None,
            gt_noise_std: float | list[float] | None = None,
            add_noise_to_gt: bool = False,
    ):
        r"""Args:
        - Linear equality input constraints must be cast in matrix form and are satisfied for Ax - b = 0.
          The user must pass a list of tuples (indices, coefficients, rhs), with each tuple encoding an equality
          constraint of the form `\sum_i (X[indices[i]] * coefficients[i]) = rhs`. See the docstring of
          `make_scipy_linear_constraints` for an example.
        - Linear inequality input constraints must be cast in matrix form Ax - b >= 0 (feasible if non-negative).
          The user must pass a list of tuples (indices, coefficients, rhs), with each tuple encoding an inequality
          constraint of the form `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`. `indices` and `coefficients`
          should be torch tensors. See the docstring of `make_scipy_linear_constraints` for an example.
          When q=1, or when applying the same constraint to each candidate in the batch (intra-point constraint),
          `indices` should be a 1-d tensor. For inter-point constraints, in which the constraint is applied to the
          whole batch of candidates, `indices` must be a 2-d tensor. Here, in each row `indices[i] = (k_i, l_i)`
          the first index `k_i` corresponds to the `k_i`-th element of the `q`-batch and the second index `l_i`
          corresponds to the `l_i`-th feature of that element.
        - Non-linear inequality input constraints must be cast in callable form f(x) >= 0 (feasible if non-negative).
          A list of tuples representing the nonlinear inequality constraints. The first element in the tuple is a
          callable representing a constraint of the form `callable(x) >= 0`. In case of an intra-point constraint,
          `callable()` takes in a one-dimensional tensor of shape `d` and returns a scalar.
          In case of an inter-point constraint, `callable()` takes a two-dimensional tensor of shape `q x d`
          and again returns a scalar. The second element is a boolean, indicating if it is an intra-point or
          inter-point constraint (`True` for intra-point. `False` for inter-point). For more information on
          intra-point vs inter-point constraints, see the docstring of the `inequality_constraints` argument to
          `optimize_acqf()`. The constraints will later be passed to the scipy solver. You need to pass in
          `batch_initial_conditions` in this case. Using non-linear inequality constraints also requires that
          `batch_limit` is set to 1, which will be done automatically if not specified in `options`.
        - Output constraints must be cast in callable form and are satisfied if f(x) >= 0 (feasible if non-negative).
        """
        super().__init__()

        # === CUDA attributes ===
        self.device = device
        self.dtype = dtype

        # Accessing the Enums via the Class (more robust)
        cls = self.__class__

        # === Derived Objective Attributes ===
        self.dim = len(cls.Par)
        self.num_objectives = len(cls.Obj)
        self.num_constraints = len(cls.Con)
        self.num_trackers = len(cls.Trk)

        # === Assigned Attributes ===
        self.to_minimize = [item.to_minimize for item in cls.Obj]
        self.bounds = self.get_bounds_from_pars()
        self.outcomes = [item.index for item in cls.Obj]
        self.num_outcomes = len(cls.Obj)
        self.gt_noise_std = gt_noise_std
        self.add_noise_to_gt = add_noise_to_gt

        # === Constraints ===
        self.linear_equality_input_constraints = linear_equality_input_constraints
        self.linear_inequality_input_constraints = linear_inequality_input_constraints
        self.nonlinear_inequality_input_constraints = nonlinear_inequality_input_constraints
        self.constraints = output_constraints

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip the check if the class is intended to be Abstract
        if ABC in cls.__bases__ or inspect.isabstract(cls):
            return

        # The mandatory "contract"
        for attr in ['Obj', 'Par', 'Con', 'Trk']:
            if not hasattr(cls, attr):
                raise TypeError(f"Class {cls.__name__} must define an inner '{attr}' Enum.")

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
    def to_minimize(self) -> Tensor:
        return self._obj_to_minimize

    @to_minimize.setter
    def to_minimize(self, value: list[bool] | torch.Tensor):
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
    def bounds(self) -> torch.Tensor:
        return self._bounds

    def get_bounds_from_pars(self) -> torch.Tensor:
        """
        Extracts bounds from Par enum members, sorted by their defined index.
        Returns a tensor of shape (dim, 2).
        """
        # Sort pars by their index attribute to ensure correct order
        sorted_pars = sorted(self.Par, key=lambda p: p.index)

        # Create the list of tuples: [(lb, ub), (lb, ub), ...]
        bounds_list = [p.bounds for p in sorted_pars]

        return torch.tensor(
            bounds_list,
            device=self._device,
            dtype=self.dtype
        ).transpose(0, 1)

    @bounds.setter
    def bounds(self, value: list[tuple[float, float]] | torch.Tensor):
        # Converts list of tuples to tensor
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

        value = torch.tensor(value, device=self.device, dtype=torch.long)
        self._outcomes = value

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
    def constraints(self):
        return self._output_constraints

    @constraints.setter
    def constraints(self, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError("constraints must be a list of Callables")
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

    # === FEASIBILITY ===

    def is_input_feasible(self, X: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        """
        Supports X (..., d) and q-btach X (..., q, d) for intra-point constraints only (1D indices).
        Returns mask of shape (...) (one per batch item / restart).
        """
        has_q = (X.dim() >= 3)  # interpret last two dims as (q, d)
        base_shape = X.shape[:-2] if has_q else X.shape[:-1]
        feasible_mask = torch.ones(base_shape, dtype=torch.bool, device=X.device)

        lower, upper = self.bounds[0], self.bounds[1]

        # bounds
        if has_q:
            in_bounds = (X >= lower).all(dim=-1) & (X <= upper).all(dim=-1)  # (..., q)
            feasible_mask &= in_bounds.all(dim=-1)  # (...,)
        else:
            feasible_mask &= (X >= lower).all(dim=-1) & (X <= upper).all(dim=-1)  # (...,)

        # linear equalities (1D indices only)
        if self.linear_equality_input_constraints:
            for indices, coeffs, rhs in self.linear_equality_input_constraints:
                if indices.dim() != 1:
                    raise ValueError("2D indices constraints require inter-point handling (not implemented here).")
                lhs = (X[..., :, indices] * coeffs).sum(dim=-1) if has_q else (X[..., indices] * coeffs).sum(dim=-1)
                feasible_mask &= ((lhs - rhs).abs() <= atol).all(dim=-1) if has_q else (lhs - rhs).abs() <= atol

        # linear inequalities, 1D indices only
        if self.linear_inequality_input_constraints:
            for indices, coeffs, rhs in self.linear_inequality_input_constraints:
                if indices.dim() != 1:
                    raise ValueError("2D indices constraints require inter-point handling (not implemented here).")
                lhs = (X[..., :, indices] * coeffs).sum(dim=-1) if has_q else (X[..., indices] * coeffs).sum(dim=-1)
                feasible_mask &= (lhs >= (rhs - atol)).all(dim=-1) if has_q else lhs >= (rhs - atol)

        # nonlinear inequalities: c(x) >= 0
        if self.nonlinear_inequality_input_constraints:
            for c_func, _ in self.nonlinear_inequality_input_constraints:
                cval = c_func(X).squeeze(-1)
                feasible_mask &= (cval >= -atol).all(dim=-1) if has_q else (cval >= -atol)

        return feasible_mask

    def is_output_feasible(self, Y: Tensor, atol=1e-6) -> Tensor:
        """ Checks if the objective/constraint outputs Y satisfy the performance constraints.
            Y is expected to be [batch_shape, num_objectives + num_constraints]. """

        if not hasattr(self, 'constraints') or self.constraints is None:
            return torch.ones(Y.shape[:-1], dtype=torch.bool, device=Y.device)

        # Each c in self.constraints is a callable that returns <= 0 for feasible
        # We stack them and check if all are satisfied
        feasible_mask = torch.stack(
            [c(Y) >= -atol for c in self.constraints]
        ).all(dim=0).squeeze(-1)

        return feasible_mask


class MCSingleObjectiveBase(MCAcquisitionObjective, MCObjectiveBase, ABC):
    def __init__(self, best_value: float | None = None, *args, **kwargs):
        ABC.__init__(self)
        MCAcquisitionObjective.__init__(self)
        MCObjectiveBase.__init__(self, *args, **kwargs)
        self.best_value = best_value

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
        selected[..., self.to_minimize] *= -1
        return selected.squeeze(-1)


class MCMultiObjectiveBase(MCMultiOutputObjective, MCObjectiveBase, ABC):
    def __init__(self, max_hv: float | None = None, *args, **kwargs):
        ABC.__init__(self)
        MCMultiOutputObjective.__init__(self)
        MCObjectiveBase.__init__(self, *args, **kwargs)
        self.ref_point = [item.ref_point for item in self.Obj]
        self.max_hv = max_hv

    """
    - The reference point is used only for multi objective problems to compute the Hypervolume.
    - The max hypervolume may be known a priori. When passed, it is plotted by the Plotter as a target optimization value.
    """

    @property
    def ref_point(self) -> Tensor:
        return self._ref_point

    @ref_point.setter
    def ref_point(self, value: list[float] | torch.Tensor):
        if isinstance(value, list):
            if len(value) != self.num_objectives:
                raise ValueError("The number of objectives must match the dimensions of the reference point.")
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        if isinstance(value, torch.Tensor):
            if value.shape[0] != self.num_objectives:
                raise ValueError("The number of objectives must match the dimensions of the reference point.")
            value = value.to(self.device, dtype=self.dtype)
        self._ref_point = value

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
        selected[..., self.obj_to_minimize] *= -1
        return selected
