import inspect
from collections.abc import Callable
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.acquisition.objective import MCAcquisitionObjective
from botorch.exceptions import InputDataError
from constraints.output_constraints import *
from objectives.variable_registry import *


class MCObjectiveBase(ABC):

    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            obj_cfg: list[ObjCfg],
            par_cfg: list[ParCfg],
            trk_cfg: list[TrkCfg] = None,
            lin_eq_X_con_cfg: list[LinEqXConCfg] = None,
            lin_ineq_X_con_cfg: list[LinIneqXConCfg] = None,
            nonlin_ineq_X_con_cfg: list[NonLinIneqXConCfg] = None,
            ineq_Y_con_cfg: list[IneqYConCfg] = None,
            gt_noise_std: float | list[float] | None = None,
    ):
        r"""
        Args:
        -obj_cfg: Configuration for objectives. It receives a list of ObjCfg.
        These objects are internally stored for optimization and visualization.

        -par_cfg: Configuration for parameters. It receives a list of ParCfg.
        These objects are internally stored for optimization and visualization.

        -trk_cfg: Configuration for trackers. It receives a list of TrkCfg.
        These objects are internally stored for optimization and visualization.

        - lin_eq_X_con_cfg: Configuration for linear equality input constraints.
        It receives a list of LinEqXConCfg. These objects are internally converted
        to a list of tuples (idxs,coeffs, rhs), with each tuple encoding
        an equality constraint of the form `\sum_i (X[idxs[i]] * coeffs[i]) = rhs`.

        - lin_ineq_X_con_cfg: Configuration for linear inequality input constraints.
        It receives a list of LinIneqXConCfg. These objects are internally converted
        to a list of tuples (idxs, coeffs, rhs), with each tuple encoding
        an inequality constraint of the form `\sum_i (X[idxs[i]] * coeffs[i]) >= rhs`.
        For intra-point constraints, idxs[i] is a 1D tensor where each index determines
        the feature of the point. For intra-point constraints, idxs[i] = (k_i, l_i) is
        a 2D tensor, where the first index `k_i` corresponds to the `k_i`-th point
        and the second index `l_i`corresponds to the `l_i`-th feature of that point.

        - nonlin_ineq_X_con_cfg: Configuration for non-linear inequality input constrsints.
        It receives a list of NonLinIneqXConCfg. These objects are internally converted
        to a list of tuples (f, intra). Here, the first element is a callable
        representing a constraint of the form `callable(x) >= 0` (feasible if non-negative).
        The second element is a boolean defining intra-point constraints (True) or
        intra-point constraints (False). In the case of an intra-point constraint,
        `callable()` takes in a one-dimensional tensor of shape `d` and returns a scalar.
        In the case of an intra-point constraint, `callable()` takes a two-dimensional
        tensor of shape `q x d` and again returns a scalar.

        - ineq_Y_con_cfg: Configuretion for inequality output constraints.
        It receives a list of IneqYConfg. These objects are internally converted to
        a list of functions representing a constraint of the form `callable(x) >= 0`
        (feasible if non-negative).

        -gt_noise_std: ...
        """
        super().__init__()

        # === CUDA attributes ===
        self.device = device
        self.dtype = dtype

        # === Store cfg ===
        self.par_cfg = par_cfg
        self.obj_cfg = obj_cfg
        self.trk_cfg = trk_cfg
        self.lin_eq_input_con_cfg = lin_eq_X_con_cfg
        self.lin_ineq_input_con_cfg = lin_ineq_X_con_cfg
        self.nonlin_ineq_input_con_cfg = nonlin_ineq_X_con_cfg
        self.output_con_cfg = ineq_Y_con_cfg

        # === Derived Objective Attributes ===
        self._dim = len(self.par_cfg) if self.par_cfg is not None else 0
        self._num_objectives = len(self.obj_cfg) if self.obj_cfg is not None else 0
        self._num_constraints = len(self.output_con_cfg) if self.output_con_cfg is not None else 0
        self._num_trackers = len(self.trk_cfg) if trk_cfg is not None else 0

        # === Assigned Attributes ===
        self._to_minimize = [cfg.to_minimize for cfg in self.obj_cfg]
        self._bounds = self.get_bounds_from_pars()
        self._outcomes = [cfg.index for cfg in self.obj_cfg]
        self._num_outcomes = len(self.obj_cfg)
        self._gt_noise_std = gt_noise_std

        # TODO: convert setters to methods
        # === Constraints ===
        self.lin_eq_X_con = lin_eq_X_con_cfg
        self.lin_ineq_X_con = lin_ineq_X_con_cfg
        self.nonlin_ineq_X_con = nonlin_ineq_X_con_cfg
        self.ineq_Y_con = ineq_Y_con_cfg

    # === Properties: CUDA ===

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

    # ===== Properties: General =====

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_objectives(self) -> int:
        return self._num_objectives

    @property
    def num_constraints(self) -> int:
        return self._num_constraints

    @property
    def to_minimize(self) -> Tensor:
        return self._obj_to_minimize

    @property
    def bounds(self) -> torch.Tensor:
        return self._bounds

    def get_bounds_from_pars(self) -> torch.Tensor:
        """ Extracts bounds from ParCfg objects, sorted by their defined index.
        Returns a tensor of shape (2, dim). """
        sorted_pars = sorted(self.par_cfg, key=lambda p: p.index)
        lb = [p.bounds[0] for p in sorted_pars]
        ub = [p.bounds[1] for p in sorted_pars]
        return torch.tensor([lb, ub], device=self.device, dtype=self.dtype)

    @property
    def outcomes(self) -> torch.Tensor:
        return self._outcomes

    @property
    def num_outcomes(self) -> int:
        return self._num_outcomes

    @property
    def gt_noise_std(self) -> torch.Tensor | None:
        return self._noise_std

    # ===== Properties: Constraints on X =====

    @property
    def lin_eq_X_con(self):
        return self._lin_eq_X_con

    @lin_eq_X_con.setter
    def lin_eq_X_con(self, cfg_list: list[LinEqXConCfg] | None):
        """ Parses LinEqXConCfg into internal tuples: (indices_tensor, coeff_tensor, rhs_float) """
        if cfg_list is not None:
            if not isinstance(cfg_list, list):
                raise TypeError("lin_eq_X_con must be a list of LinEqXConCfg")

            processed = []
            for cfg in cfg_list:
                if not isinstance(cfg, LinEqXConCfg):
                    raise TypeError("lin_eq_X_con must be a list of LinEqXConCfg")

                indices_t = torch.tensor(cfg.idxs, device=self.device, dtype=torch.long)
                coeff_t = torch.tensor(cfg.coeff, device=self.device, dtype=self.dtype)
                rhs = torch.tensor(cfg.rhs, device=self.device, dtype=self.dtype)
                processed.append((indices_t, coeff_t, rhs))

            self._lin_eq_X_con = processed
        else:
            self._lin_eq_X_con = None

    @property
    def lin_ineq_X_con(self):
        return self._lin_ineq_X_con

    @lin_ineq_X_con.setter
    def lin_ineq_X_con(self, cfg_list: list[LinIneqXConCfg] | None):
        """ Parses LinIneqXConCfg into internal tuples: (indices_tensor, coeff_tensor, rhs_float)"""
        if cfg_list is not None:
            if not isinstance(cfg_list, list):
                raise TypeError("lin_ineq_X_con must be a list of LinIneqXConCfg")

            processed = []
            for cfg in cfg_list:
                if not isinstance(cfg, LinIneqXConCfg):
                    raise TypeError("lin_ineq_X_con must be a list of LinIneqXConCfg")
                indices_t = torch.tensor(cfg.idxs, device=self.device, dtype=torch.long)
                coeff_t = torch.tensor(cfg.coeff, device=self.device, dtype=self.dtype)
                rhs = torch.tensor(cfg.rhs, device=self.device, dtype=self.dtype)
                processed.append((indices_t, coeff_t, rhs))

            self._lin_ineq_X_con = processed
        else:
            self._lin_ineq_X_con = None

    @property
    def nonlin_ineq_X_con(self):
        return self._nonlin_ineq_X_con

    @nonlin_ineq_X_con.setter
    def nonlin_ineq_X_con(self, cfg_list):
        """ Parses NonLinIneqXConCfg into internal tuples: (Callable, bool).
        The boolean 'intra' flag determines if it is intra-point (True) or
        inter-point (False). """
        if cfg_list is not None:
            if not isinstance(cfg_list, list):
                raise TypeError("nonlin_ineq_X_con must be a list of NonLinIneqXConCfg")

            processed = []
            for cfg in cfg_list:
                if not isinstance(cfg, NonLinIneqXConCfg):
                    raise TypeError("nonlin_ineq_X_con must be a list of NonLinIneqXConCfg")
                if not callable(cfg.f):
                    raise TypeError("Constraint 'f' must be callable")
                if not isinstance(cfg.intra, bool):
                    raise TypeError("Constraint 'intra' must be a boolean")
                processed.append((cfg.f, cfg.intra))

            self._nonlin_ineq_X_con = processed
        else:
            self._nonlin_ineq_X_con = None

    # ===== Properties: Constraints on Y =====

    @property
    def ineq_Y_con(self):
        return self._ineq_Y_con

    @ineq_Y_con.setter
    def ineq_Y_con(self, cfg_list: list[IneqYConCfg] | None):
        if cfg_list is not None:
            if not isinstance(cfg_list, list):
                raise TypeError(f"ineq_Y_con_cfg must be a list of {IneqYConCfg.__name__}")

            processed = []
            for cfg in cfg_list:
                if not isinstance(cfg, IneqYConCfg):
                    raise TypeError(f"ineq_Y_con_cfg must be a list of {IneqYConCfg.__name__}")
                if not callable(cfg.f):
                    raise TypeError(f"Each {IneqYConCfg.f.__name__} must be callable")
                processed.append(cfg.f)
            self._ineq_Y_con = processed
        else:
            self._ineq_Y_con = None

    # ===== Methods =====

    def evaluate_true_objective(self, X: Tensor) -> Tensor:
        """
        Evaluate the true objective at the given input locations X.
        """
        pass

    def evaluate_true_objective_with_noise(self, X: Tensor) -> Tensor:
        Y = self.evaluate_true_objective(X)
        Y = self.add_noise(Y)
        return Y

    def evaluate_true_constraints(self, X: Tensor) -> Tensor:
        """
        Evaluates and stacks all nonlinear input ineq_Y_con_cfg.
        """
        if not self.ineq_Y_con:
            return torch.empty((*X.shape[:-1], 0), device=self.device, dtype=self.dtype)
        return torch.stack([f(X) for f, _ in self.ineq_Y_con], dim=-1)

    def evaluate_true_constraints_with_noise(self, X: Tensor) -> Tensor:
        Y = self.evaluate_true_constraints(X)
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
        return self.evaluate_true_constraints(X=X) - slack

    def add_noise(self, Y: Tensor) -> Tensor:
        """
        A method to add noise to the observations.
        """
        if self._gt_noise_std is None:
            raise ValueError("noise_std is required to add_noise.")
        noise = self._noise_std.to(Y.device) * torch.randn_like(Y)
        return Y + noise

    def is_X_feasible(self, X: torch.Tensor, atol: float = 1e-6) -> torch.Tensor:
        """ Supports X (..., d) and q-btach X (..., q, d) for inter- and intra-point
        input constraints. Returns mask of shape (...) (one per batch item / restart)."""
        has_q = (X.dim() >= 3)  # interpret the last two dims as (q, d)
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
        if self.lin_eq_X_con:
            for indices, coeffs, rhs in self.lin_eq_X_con:
                if indices.dim() != 1:
                    raise ValueError("2D indices ineq_Y_con_cfg require intra-point handling (not implemented here).")
                lhs = (X[..., :, indices] * coeffs).sum(dim=-1) if has_q else (X[..., indices] * coeffs).sum(dim=-1)
                feasible_mask &= ((lhs - rhs).abs() <= atol).all(dim=-1) if has_q else (lhs - rhs).abs() <= atol

        # linear inequalities, 1D indices only
        if self.lin_ineq_X_con:
            for indices, coeffs, rhs in self.lin_ineq_X_con:
                if indices.dim() != 1:
                    raise ValueError("2D indices ineq_Y_con_cfg require intra-point handling (not implemented here).")
                lhs = (X[..., :, indices] * coeffs).sum(dim=-1) if has_q else (X[..., indices] * coeffs).sum(dim=-1)
                feasible_mask &= (lhs >= (rhs - atol)).all(dim=-1) if has_q else lhs >= (rhs - atol)

        # nonlinear inequalities: c(x) >= 0
        if self.nonlin_ineq_X_con:
            for c_func, _ in self.nonlin_ineq_X_con:
                cval = c_func(X).squeeze(-1)
                feasible_mask &= (cval >= -atol).all(dim=-1) if has_q else (cval >= -atol)

        return feasible_mask

    def is_Y_feasible(self, Y: Tensor, atol=1e-6) -> Tensor:
        """ Checks if the objective/constraint outputs Y satisfy the performance ineq_Y_con_cfg.
            Y is expected to be [batch_shape, num_objectives + num_constraints]. """

        if not hasattr(self, 'ineq_Y_con_cfg') or self.ineq_Y_con is None:
            return torch.ones(Y.shape[:-1], dtype=torch.bool, device=Y.device)

        # Each c in self.ineq_Y_con_cfg is a callable that returns <= 0 for feasible
        # We stack them and check if all are satisfied
        feasible_mask = torch.stack(
            [c(Y) >= -atol for c in self.ineq_Y_con]
        ).all(dim=0).squeeze(-1)

        return feasible_mask

    def get_idx(self, label: str) -> int:
        """ Returns the integer index for a given parameter label. """
        for item in self.par_cfg:
            if item.label == label:
                return item.index
        raise ValueError(f"Label '{label}' not found in par_cfg")


class MCSingleObjectiveBase(MCAcquisitionObjective, MCObjectiveBase, ABC):
    def __init__(
            self,
            best_value: float | None = None,
            device: torch.device = None,
            dtype: torch.dtype = None,
            obj_cfg: list[ObjCfg] = None,
            par_cfg: list[ParCfg] = None,
            trk_cfg: list[TrkCfg] = None,
            lin_eq_X_con_cfg: list[LinEqXConCfg] = None,
            lin_ineq_X_con_cfg: list[LinIneqXConCfg] = None,
            nonlin_ineq_X_con_cfg: list[NonLinIneqXConCfg] = None,
            ineq_Y_con_cfg: list[IneqYConCfg] = None,
            gt_noise_std: float | list[float] | None = None,
    ):
        ABC.__init__(self)
        MCAcquisitionObjective.__init__(self)
        MCObjectiveBase.__init__(
            self,
            device=device,
            dtype=dtype,
            obj_cfg=obj_cfg,
            par_cfg=par_cfg,
            trk_cfg=trk_cfg,
            lin_eq_X_con_cfg=lin_eq_X_con_cfg,
            lin_ineq_X_con_cfg=lin_ineq_X_con_cfg,
            nonlin_ineq_X_con_cfg=nonlin_ineq_X_con_cfg,
            ineq_Y_con_cfg=ineq_Y_con_cfg,
            gt_noise_std=gt_noise_std,
        )
        self.best_value = best_value

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
        selected[..., self._to_minimize] *= -1
        return selected.squeeze(-1)


class MCMultiObjectiveBase(MCMultiOutputObjective, MCObjectiveBase, ABC):
    def __init__(
            self,
            max_hv: float | None = None,
            device: torch.device = None,
            dtype: torch.dtype = None,
            obj_cfg: list[ObjCfg] = None,
            par_cfg: list[ParCfg] = None,
            trk_cfg: list[TrkCfg] = None,
            lin_eq_X_con_cfg: list[LinEqXConCfg] = None,
            lin_ineq_X_con_cfg: list[LinIneqXConCfg] = None,
            nonlin_ineq_X_con_cfg: list[NonLinIneqXConCfg] = None,
            ineq_Y_con_cfg: list[IneqYConCfg] = None,
            gt_noise_std: float | list[float] | None = None,
    ):
        ABC.__init__(self)
        MCMultiOutputObjective.__init__(self)
        MCObjectiveBase.__init__(
            self,
            device=device,
            dtype=dtype,
            obj_cfg=obj_cfg,
            par_cfg=par_cfg,
            trk_cfg=trk_cfg,
            lin_eq_X_con_cfg=lin_eq_X_con_cfg,
            lin_ineq_X_con_cfg=lin_ineq_X_con_cfg,
            nonlin_ineq_X_con_cfg=nonlin_ineq_X_con_cfg,
            ineq_Y_con_cfg=ineq_Y_con_cfg,
            gt_noise_std=gt_noise_std,
        )
        self.ref_point = [cfg.ref_point for cfg in self.obj_cfg]
        self.max_hv = max_hv

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

    def evaluate_true_objective(self, X: Tensor) -> Tensor:
        """ Evaluates and stacks all objectives in the order of their defined index. """
        # Sort the configuration objects by their index
        sorted_objs = sorted(self.obj_cfg, key=lambda obj: obj.index)
        # Evaluate and stack along the last dimension
        # This ensures column 0 is the obj with index 0, column 1 is index 1, etc.
        return torch.stack([obj.f(X) for obj in sorted_objs], dim=-1)

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
        selected[..., self._to_minimize] *= -1
        return selected
