import datetime
import glob
import json
import os
import pickle
import time
import torch
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union
from botorch.utils.multi_objective import is_non_dominated, Hypervolume
from torch import Tensor
from pybo.objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase, MCMultiObjectiveBase
from pybo.optimizer.validators import _set_tensor, _check_tensor


class OptimizerBase(ABC):
    """Data, scoring, metrics and I/O for an optimization run of any kind."""

    # Attributes dropped on pickling because they do not survive it.
    _unpicklable_attrs: tuple[str, ...] = ()

    # What proposed the observations - the arm, not whether it was simulated or run for
    # real (see to_json's "experiment_type", a separate axis). Written into every record
    # this optimizer saves.
    optimizer_type: str = "base"

    def __init__(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            objective: MCSingleObjectiveBase | MCMultiObjectiveBase | None = None,
            X: torch.Tensor | None = None,
            Y_obj: torch.Tensor | None = None,
            Y_obj_var: torch.Tensor | None = None,
            Y_con: torch.Tensor | None = None,
            Y_con_var: torch.Tensor | None = None,
            Y_trk: torch.Tensor | None = None,
            Y_trk_var: torch.Tensor | None = None,
            batch_size: int = 1,
    ):

        # ===== Initialize Private Attributes =====
        # Hardware attributes
        self._device = None
        self._dtype = None
        self._objective = None

        # Experiment attributes
        self._datetime = datetime.datetime.now()
        self._X = None
        self._Y_obj = None
        self._Y_obj_var = None
        self._Y_con = None
        self._Y_con_var = None
        self._Y_trk = None
        self._Y_trk_var = None
        self._batch_size = None
        self._source: list[str] = []  # What produced X (one entry per row), e.g. initial sampling or optimization.

        # State attributes
        self._new_X: torch.Tensor | None = None  # The new X proposed by the optimizer
        self._feasible_mask: torch.Tensor | None = None  # Boolean mask, true where observations are feasible

        # Single-objective Attributes
        self._best_f: torch.Tensor | None = None  # Y value of best feasible observation, cast in maximization space
        self._best_f_mask: torch.Tensor | None = None
        self._best_feasible_Y: torch.Tensor | None = None  # Y value of best feasible observation
        self._best_feasible_X: torch.Tensor | None = None  # X value of best feasible observation

        # Multi-objective Attributes
        self._ref_point: torch.Tensor | None = None
        self._pareto_front_mask: torch.Tensor | None = None  # Pareto mask, true for Pareto optimal points
        self._feasible_pareto_front_Y: torch.Tensor | None = None
        self._feasible_pareto_front_X: torch.Tensor | None = None

        # Metrics
        self._hypervolume: list[float] = []
        self._best_values: list[float] = []
        self._elapsed_time: list[float] = []

        # Warnings
        self._warnings = []

        # ===== Assign Attributes =====
        self.device = device
        self.dtype = dtype
        self.objective = objective
        self.X = X
        self.Y_obj = Y_obj
        self.Y_obj_var = Y_obj_var
        self.Y_con = Y_con
        self.Y_con_var = Y_con_var
        self.Y_trk = Y_trk
        self.Y_trk_var = Y_trk_var
        self.batch_size = batch_size

    """ =========================== """
    """ ===== PICKLING HELPER ===== """
    """ =========================== """

    def __getstate__(self):
        state = self.__dict__.copy()
        for attr in self._unpicklable_attrs:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._transient = None
        # Pickles written before provenance was recorded carry no _source. Without this
        # they unpickle but raise on first access, including from the X setter.
        if not hasattr(self, "_source"):
            self._source = []

    def _print_caught_warnings(self, flush=True):
        ORANGE = "\033[38;5;208m"
        RESET = "\033[0m"
        for w in self._warnings:
            if flush:
                print(f"{ORANGE}{w.category.__name__}: {w.message}{RESET}", flush=True)

    def _clear_warnings(self):
        self._warnings = []

    @staticmethod
    def _print_success(msg: str = "", flush=True):
        GREEN = "\033[32m"
        RESET = "\033[0m"
        print(f"{GREEN}✅ {msg}{RESET}", flush=flush)

    @staticmethod
    def _print_failed(msg: str = "", flush=True):
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}❌ {msg}{RESET}", flush=flush)

    """ =============================== """
    """ ===== HARDWARE PROPERTIES ===== """
    """ =============================== """

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | None):
        if device is None:
            self._device = torch.device("cpu")
        elif not isinstance(device, torch.device):
            raise ValueError("Device must be of type torch.device or None")
        else:
            self._device = device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype | None):
        if dtype is None:
            self._dtype = torch.float64
        elif not isinstance(dtype, torch.dtype):
            raise ValueError("dtype must be of type torch.dtype or None")
        else:
            self._dtype = dtype

    """ =================================== """
    """ ===== EXPERIMENTAL PROPERTIES ===== """
    """ =================================== """

    @property
    def datetime(self):
        """When this optimizer was constructed. Read-only: it stamps the run, so nothing
        outside should be able to rewrite it."""
        return self._datetime

    @property
    def objective(self) -> Union[MCMultiObjectiveBase, MCSingleObjectiveBase]:
        return self._objective

    @objective.setter
    def objective(self, objective: Union[MCObjectiveBase, None]):
        if objective is not None and not isinstance(objective, MCObjectiveBase):
            raise ValueError("Objective is not compatible.")
        self._objective = objective

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value: torch.Tensor | None):
        self._X = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)
        # Provenance is per row, so it follows X's length here - the one place every
        # assignment passes through, whether it comes from the constructor or update_X.
        # Rows that arrive without a stated source are the initial design; update_X
        # renames the ones it appends. Rows that disappear take their entry with them.
        n_rows = 0 if self._X is None else self._X.shape[0]
        del self._source[n_rows:]
        self._source += ["initial"] * (n_rows - len(self._source))

    @property
    def Y_obj(self):
        return self._Y_obj

    @Y_obj.setter
    def Y_obj(self, value: torch.Tensor | None):
        self._Y_obj = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)

    @property
    def Y_obj_var(self):
        return self._Y_obj_var

    @Y_obj_var.setter
    def Y_obj_var(self, value: torch.Tensor | None):
        self._Y_obj_var = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)

    @property
    def Y_con(self):
        return self._Y_con

    @Y_con.setter
    def Y_con(self, value: torch.Tensor | None):
        self._Y_con = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)

    @property
    def Y_con_var(self):
        return self._Y_con_var

    @Y_con_var.setter
    def Y_con_var(self, value: torch.Tensor | None):
        self._Y_con_var = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)

    @property
    def Y_trk(self):
        return self._Y_trk

    @Y_trk.setter
    def Y_trk(self, value: torch.Tensor | None):
        self._Y_trk = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)

    @property
    def Y_trk_var(self):
        return self._Y_trk_var

    @Y_trk_var.setter
    def Y_trk_var(self, value: torch.Tensor | None):
        self._Y_trk_var = _set_tensor(value=value, device=self.device, dtype=self.dtype, ndim=2)

    @property
    def source(self) -> list[str]:
        """What produced each observation, one entry per row of X: "initial" for the
        initial design, else whatever the caller named the batch (the tutorials use
        "proposed"), falling back to this optimizer's optimizer_type. A copy, so the
        history cannot be rewritten from outside."""
        return list(self._source)

    @property
    def n_initial_samples(self) -> int:
        """How many observations came from the initial design. Counted from the recorded
        provenance rather than declared, so a run that never took one reports 0."""
        return self._source.count("initial")

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        if not isinstance(batch_size, int):
            raise ValueError("batch_size must be of type int")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self._batch_size = batch_size

    """ ============================ """
    """ ===== STATE PROPERTIES ===== """
    """ ============================ """

    @property
    def model(self):
        """The surrogate, or None for an arm that fits none. Overridden where there is one. """
        return None

    @property
    def acqf(self):
        """The acquisition function class, or None for an arm that optimizes none. """
        return None

    @property
    def acqf_instance(self):
        """The acquisition function, or None for an arm that builds none."""
        return None

    @property
    def ref_point(self):
        return self._ref_point

    @property
    def pareto_front_mask(self) -> torch.Tensor | None:
        return self._pareto_front_mask

    @property
    def best_f_mask(self) -> torch.Tensor | None:
        return self._best_f_mask

    @property
    def new_X(self) -> Tensor:
        return self._new_X

    @property
    def hypervolume(self):
        return self._hypervolume

    @property
    def best_values(self):
        return self._best_values

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @property
    def feasible_mask(self) -> torch.Tensor | None:
        return self._feasible_mask

    @property
    def best_feasible_X(self) -> torch.Tensor | None:
        return self._best_feasible_X

    @property
    def best_feasible_Y(self) -> torch.Tensor | None:
        return self._best_feasible_Y

    @property
    def best_f(self) -> torch.Tensor | None:
        return self._best_f

    @property
    def feasible_pareto_front_X(self) -> torch.Tensor | None:
        return self._feasible_pareto_front_X

    @property
    def feasible_pareto_front_Y(self) -> torch.Tensor | None:
        return self._feasible_pareto_front_Y

    """ ===================== """
    """ ===== Optimizer ===== """
    """ ===================== """

    def optimize(self, verbose=True):
        """Score the data, then let the arm propose the next batch.

        The scoring phase is identical for every arm and runs first: mark which
        observations are feasible, compute the reference the proposal will be made
        against (best_f for single-objective, the hypervolume reference point for
        multi-objective), then the metrics, which read both. `_propose` follows.

        The elapsed time recorded covers the whole step, so an arm that proposes cheaply
        honestly reports that nearly all of its cost is metric computation."""
        self._validate_state()
        t0 = time.monotonic()

        # === 1. Compute Metrics ===
        self._compute_feasible_mask(verbose=verbose)
        self._compute_acquisition_function_reference(verbose=verbose)
        self._compute_metrics(verbose=verbose)

        # === 2. Propose the next batch ===
        self._propose(verbose=verbose)

        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)

        if verbose:
            print(f"Optimization step completed in {t1 - t0:.2f}s")

    def _validate_state(self):
        """ Validate data against objective """
        self._validate_objective()
        self._validate_X()
        self._validate_Y_obj()
        self._validate_Y_con()
        self._validate_Y_trk()

    def _validate_objective(self):
        if self.objective is None:
            raise RuntimeError("No objective to optimize: set objective before calling optimize().")

    def _validate_X(self):
        if self._X is None:
            raise RuntimeError("No data to optimize: set X before calling optimize().")
        # X sets the row count every other tensor is held to below.
        _check_tensor(self._X, name="X", last_dim=self._objective.dim)
        if len(self._source) != self._X.shape[0]:
            raise RuntimeError(f"Provenance is out of step with the data: {len(self._source)} "
                               f"source entries for {self._X.shape[0]} observations. Rows must "
                               f"be appended through update_X, which records where they came from.")

    def _validate_Y_obj(self):
        if self._Y_obj is None:
            raise RuntimeError("No data to optimize: set Y_obj before calling optimize().")
        n_rows = self._X.shape[0]
        _check_tensor(self._Y_obj, name="Y_obj", last_dim=self._objective.num_obj, n_rows=n_rows)
        if self._Y_obj_var is not None:
            _check_tensor(self._Y_obj_var, name="Y_obj_var", last_dim=self._objective.num_obj, n_rows=n_rows)

    def _validate_Y_con(self):
        if self._objective.num_con == 0:
            if self._Y_con is not None:
                raise ValueError("Y_con was provided, but the objective declares no output constraints.")
            if self._Y_con_var is not None:
                raise ValueError("Y_con_var was provided, but the objective declares no output constraints.")
            return
        if self._Y_con is None:
            raise RuntimeError(
                f"Objective declares {self._objective.num_con} output constraint(s): "
                f"set Y_con before calling optimize().")
        _check_tensor(self._Y_con, name="Y_con", last_dim=self._objective.num_con, n_rows=self._X.shape[0])
        if self._Y_con_var is not None:
            _check_tensor(self._Y_con_var, name="Y_con_var", last_dim=self._objective.num_con,
                          n_rows=self._X.shape[0])

    def _validate_Y_trk(self):
        if self._objective.num_trk == 0:
            if self._Y_trk is not None:
                raise ValueError("Y_trk was provided, but the objective declares no trackers.")
            if self._Y_trk_var is not None:
                raise ValueError("Y_trk_var was provided, but the objective declares no trackers.")
            return
        if self._Y_trk is None:
            raise RuntimeError(
                f"Objective declares {self._objective.num_trk} tracker(s): "
                f"set Y_trk before calling optimize().")
        _check_tensor(self._Y_trk, name="Y_trk", last_dim=self._objective.num_trk, n_rows=self._X.shape[0])
        if self._Y_trk_var is not None:
            _check_tensor(self._Y_trk_var, name="Y_trk_var", last_dim=self._objective.num_trk,
                          n_rows=self._X.shape[0])

    @abstractmethod
    def _propose(self, verbose=True):
        """Choose the next batch and assign it to `self._new_X`.

        Called by optimize() once the data has been scored, so the feasibility mask,
        best_f / reference point and metrics are all current."""
        pass

    def _compute_feasible_mask(self, verbose=True):
        """ Computes feasible mask for all observed data (X, Y_obj). A point is defined
        feasible only if it satisfies both Input and Output constraints on X and Y_obj,
        respectively. """
        if verbose:
            print("Computing feasibility mask... ", end="")

        feasible_input_mask = self._compute_feasible_input_mask()
        feasible_output_mask = self._compute_feasible_output_mask()
        self._feasible_mask = feasible_input_mask & feasible_output_mask

        if verbose:
            n_points = self.X.shape[0]
            # At this points, feasible_mask cannot be None
            assert self._feasible_mask is not None
            n_feasible = self._feasible_mask.sum().item()
            self._print_success(msg=f"({n_feasible}/{n_points} feasible)")

    def _compute_feasible_input_mask(self):
        return self.objective.is_X_feasible(self.X)

    def _compute_feasible_output_mask(self):
        if self.objective.ineq_Y_con is None:
            Y_feasible = torch.ones(self._Y_obj.shape[0], dtype=torch.bool, device=self._device)
        else:
            Y_full = torch.cat([self._Y_obj, self._Y_con], dim=-1)
            Y_feasible = self.objective.is_Y_feasible(Y_full)
        return Y_feasible

    def _compute_metrics(self, verbose=True):
        """ Compute optimization metrics depending on problem dimensionality
        Single-objective: best value
        Multi-objective: Pareto front, then the hypervolume it spans."""
        if self._objective.num_obj == 1:
            self._compute_best_value(verbose=verbose)
        else:
            self._compute_pareto_front(verbose=verbose)
            self._compute_hypervolume(verbose=verbose)

    def _compute_best_value(self, verbose=True):
        if verbose:
            print("Computing best value... ", end="")

        if self._best_feasible_Y is not None:
            best_value = self._best_feasible_Y.squeeze().item()
        else:
            best_value = float('inf') if self._objective.to_minimize[0] else float('-inf')

        self._best_values.append(best_value)

        if verbose:
            self._print_success(msg=f"({best_value:.4f})")

    def _compute_hypervolume(self, verbose=True):
        if verbose:
            print("Computing hypervolume... ", end="")

        if self._feasible_pareto_front_Y is None:
            hv = torch.nan
            if verbose:
                print("✗ Cannot compute hypervolume. No Pareto front found.")
            self._hypervolume.append(hv)
            return

        # Negate only the dimensions that are minimization objectives
        feasible_pareto_front_Y_maximized = self._feasible_pareto_front_Y.clone()
        feasible_pareto_front_Y_maximized[..., self._objective.to_minimize] *= -1
        hv = Hypervolume(self._ref_point).compute(feasible_pareto_front_Y_maximized)
        self._hypervolume.append(hv)

        if verbose:
            self._print_success(msg=f"({hv:.4f})")

    def _compute_acquisition_function_reference(self, verbose=True):
        """Compute the reference value the metrics are scored against.
        Single-objective: best value (_best_f)
        Multi-objective: reference point in high dimensional space.

        Named for the acquisition function because that is the other thing that reads it,
        but every arm needs it: the hypervolume is measured from the same reference point
        whether or not anything was modelled."""
        if self._objective.num_obj == 1:
            self._compute_best_f(verbose=verbose)
        else:
            self._compute_reference_point(verbose=verbose)

    def _compute_best_f(self, verbose=True):
        """Compute the optimal observation (for single-objective).
        This method assumes maximization, therefore, the Y must be cast into a
        maximization problem before computing the best observation."""
        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        if verbose:
            print("Computing best feasible... ", end="")

        feasible_X, feasible_Y = self._compute_feasible_XY(verbose=False)

        if isinstance(feasible_X, torch.Tensor) and isinstance(feasible_Y, torch.Tensor):
            feasible_Y_max = feasible_Y.clone()
            feasible_Y_max[..., self._objective.to_minimize] *= -1
            best_idx = feasible_Y_max.squeeze(-1).argmax()

            self._best_feasible_X = feasible_X[best_idx]
            self._best_feasible_Y = feasible_Y[best_idx]
            self._best_f = feasible_Y_max[best_idx]
            best_f_mask = torch.zeros_like(self._feasible_mask, dtype=torch.bool)
            best_f_mask[best_idx] = True
            self._best_f_mask = best_f_mask.unsqueeze(-1)

            if verbose:
                assert self._best_feasible_Y is not None
                self._print_success(msg=f"({self._best_feasible_Y.item():.4f} in orig. space)")

        else:
            self._best_feasible_Y = None
            self._best_feasible_X = None
            self._best_f = torch.full((1,), -float("inf"), device=self._device, dtype=self._dtype)
            self._best_f_mask = torch.zeros_like(self._feasible_mask, dtype=torch.bool).unsqueeze(-1)

            if verbose:
                self._print_failed(msg="(no feasible points)")

    def _compute_reference_point(self, verbose=True):
        """
        Compute and set the reference point in the maximization space.
        Note that the reference point in the original space must be
        provided explicitly by the objective ("self._objective.ref_point").
        """

        if verbose:
            print("Defining reference point... ", end="")

        self._ref_point = self.objective.ref_point.clone().to(self._device, self._dtype)
        self._ref_point[..., self._objective.to_minimize] *= -1

        if verbose:
            self._print_success(msg=f"{self._ref_point.detach().cpu().numpy()} in max. space.")

    def _compute_pareto_front(self, verbose=True):
        """
        Compute the Pareto front including constraints. Note that since
        "is_non_dominated" assumes maximization, the Y must be cast into a
        maximization problem before computing the pareto front.
        """

        if verbose:
            print("Finding Pareto front... ", end="")

        if self._objective.num_obj == 1:
            raise ValueError("Pareto front cannot be computed for single-objective problems.")

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        Y_obj_maximized = self._Y_obj.clone()
        Y_obj_maximized[..., self._objective.to_minimize] *= -1

        # Compute feasible Pareto front
        if self._feasible_mask.any():
            feasible_pareto_mask = torch.zeros_like(self._feasible_mask)
            feasible_pareto_mask[self._feasible_mask] = is_non_dominated(Y_obj_maximized[self._feasible_mask])
            self._pareto_front_mask = feasible_pareto_mask.unsqueeze(-1)
            self._feasible_pareto_front_X = self.X[feasible_pareto_mask]
            self._feasible_pareto_front_Y = self._Y_obj[feasible_pareto_mask]
        else:
            self._pareto_front_mask = torch.zeros_like(self._feasible_mask).unsqueeze(-1)
            self._feasible_pareto_front_X = None
            self._feasible_pareto_front_Y = None

        if verbose:
            self._print_success()

    def _compute_feasible_XY(self, verbose=False):
        """ Computes feasible X and Y """

        if verbose:
            print("Computing feasible X and Y... ", end="")

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        if self._feasible_mask.any():
            feasible_X = self.X[self._feasible_mask].clone()
            feasible_Y = self._Y_obj[self._feasible_mask].clone()
        else:
            feasible_X = None
            feasible_Y = None

        if verbose:
            print("✓")

        return feasible_X, feasible_Y

    def _compute_infeasible_XY(self, verbose=False):
        """ Computes infeasible X and Y. This method assumes maximization,
        therefore, the Y must be cast into a maximization problem. """

        if verbose:
            print("Computing infeasible X and Y... ", end="")

        if self._objective.num_obj != 1:
            raise ValueError("Only single objective is currently supported.")

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        infeasible_mask = torch.logical_not(self._feasible_mask)
        if infeasible_mask.any():
            infeasible_X = self.X[infeasible_mask].clone()
            infeasible_Y = self._Y_obj[infeasible_mask].clone()
        else:
            infeasible_X = None
            infeasible_Y = None

        if verbose:
            print("✓")

        return infeasible_X, infeasible_Y

    def is_converged(self, patience=10, tol=1e-3, verbose=True):

        if verbose:
            print("Checking convergence... ", end="")

        # Select the relevant metric
        if self._objective.num_obj == 1:
            metrics_list = self._best_values
        else:
            metrics_list = self._hypervolume

        # Need enough history to evaluate
        if not metrics_list or len(metrics_list) < patience:
            if verbose:
                self._print_failed()
            return False

        # Take the last `patience` values
        metrics = metrics_list[-patience:]
        # Compute improvements between consecutive steps
        improvements = [metrics[i + 1] - metrics[i] for i in range(len(metrics) - 1)]
        # Converged if all improvements are smaller than tolerance
        if verbose:
            self._print_success()
        return all(abs(impr) < tol for impr in improvements)

    """ ================== """
    """ ===== PROBES ===== """
    """ ================== """

    def compute_acquisition_function_value_at_X(self, X: torch.Tensor, verbose=True):
        if verbose:
            print("Acquisition function value: none - this run optimizes no acquisition function.")
        return None

    def compute_posterior_mean_at_X(self, X: torch.Tensor, verbose=True):
        if verbose:
            print("Posterior mean: none - this run fits no model.")
        return None

    def compute_posterior_variance_at_X(self, X: torch.Tensor, verbose=True):
        if verbose:
            print("Posterior variance: none - this run fits no model.")
        return None

    """ ================== """
    """ ===== UPDATE ===== """
    """ ================== """

    def update_XY(self, new_X: torch.Tensor, new_Y_obj: torch.Tensor, new_Y_trk: torch.Tensor | None = None,
                  new_Y_obj_var: torch.Tensor | None = None,
                  new_Y_con: torch.Tensor | None = None, new_Y_con_var=None,
                  new_Y_trk_var: torch.Tensor | None = None, source: str | None = None) -> None:
        self.update_X(new_X, source=source)
        self.update_Y_obj(new_Y_obj, new_Y_obj_var)
        self.update_Y_con(new_Y_con, new_Y_con_var)
        self.update_Y_trk(new_Y_trk, new_Y_trk_var)

    def update_X(self, new_X: torch.Tensor, source: str | None = None):
        if new_X is not None:
            new_X = _set_tensor(new_X, device=self._device, dtype=self._dtype, ndim=2)
            self.X = new_X if self._X is None else torch.cat([self._X, new_X], dim=0)
            # The setter has already grown _source to match; name the rows just appended.
            # Unnamed batches are the arm's own proposals, so only the initial design has
            # to say what it is.
            n_new = new_X.shape[0]
            self._source[len(self._source) - n_new:] = [source or self.optimizer_type] * n_new

    def update_Y_obj(self, new_Y_obj: torch.Tensor, new_Y_obj_var: torch.Tensor | None = None):
        if new_Y_obj is not None:
            new_Y_obj = _set_tensor(new_Y_obj, device=self._device, dtype=self._dtype, ndim=2)
            self.Y_obj = new_Y_obj if self._Y_obj is None else torch.cat([self._Y_obj, new_Y_obj], dim=0)
        if new_Y_obj_var is not None:
            new_Y_obj_var = _set_tensor(new_Y_obj_var, device=self._device, dtype=self._dtype, ndim=2)
            self.Y_obj_var = (new_Y_obj_var if self._Y_obj_var is None
                              else torch.cat([self._Y_obj_var, new_Y_obj_var], dim=0))

    def update_Y_con(self, new_Y_con: torch.Tensor | None, new_Y_con_var: torch.Tensor | None = None):
        if new_Y_con is not None:
            new_Y_con = _set_tensor(new_Y_con, device=self._device, dtype=self._dtype, ndim=2)
            self.Y_con = new_Y_con if self._Y_con is None else torch.cat([self._Y_con, new_Y_con], dim=0)
        if new_Y_con_var is not None:
            new_Y_con_var = _set_tensor(new_Y_con_var, device=self._device, dtype=self._dtype, ndim=2)
            self.Y_con_var = (new_Y_con_var if self._Y_con_var is None
                              else torch.cat([self._Y_con_var, new_Y_con_var], dim=0))

    def update_Y_trk(self, new_Y_track: torch.Tensor | None, new_Y_track_var: torch.Tensor | None = None):
        if new_Y_track is not None:
            new_Y_track = _set_tensor(new_Y_track, device=self._device, dtype=self._dtype, ndim=2)
            self.Y_trk = new_Y_track if self._Y_trk is None else torch.cat([self._Y_trk, new_Y_track], dim=0)
        if new_Y_track_var is not None:
            new_Y_track_var = _set_tensor(new_Y_track_var, device=self._device, dtype=self._dtype, ndim=2)
            self.Y_trk_var = (new_Y_track_var if self._Y_trk_var is None
                              else torch.cat([self._Y_trk_var, new_Y_track_var], dim=0))

    """ =============== """
    """ ===== I/O ===== """
    """ =============== """

    @staticmethod
    def _next_indexed_path(filepath: str | Path) -> Path:
        """Return the next free `<stem>_NNN<suffix>` path under the cwd, so
        repeated saves never overwrite (e.g. experiment_000.csv, _001, ...)."""
        p = Path(filepath)
        base = Path.cwd() / p.parent
        i = 0
        save_path = base / f"{p.stem}_{i:03d}{p.suffix}"
        while save_path.exists():
            i += 1
            save_path = base / f"{p.stem}_{i:03d}{p.suffix}"
        return save_path

    @staticmethod
    def _atomic_replace(tmp_path: Path, out_path: Path):
        """Swap a freshly written temp file over `out_path`, retrying the
        transient PermissionError Windows raises when another process holds a
        handle on either path.

        A virus scanner or the search indexer opens each file it just saw
        written for a few ms, and os.replace needs exclusive access to both, so
        it fails outright rather than blocking. Frequent snapshots provoke it: a
        Sobol trial saves ~160 times a second, where a BO trial's steps are
        seconds apart and never collide. The handle is always released long
        before the last attempt."""
        for attempt in range(10):
            try:
                os.replace(tmp_path, out_path)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * 2 ** attempt)

    def to_file(self, filepath: str | Path | None = None, verbose=True):
        if verbose:
            print("Saving optimizer to file... ", end="")

        # Pickle the whole optimizer, overwriting the target file atomically
        # (write to a temp file, then os.replace), so it can be called every
        # iteration as a running, crash-safe snapshot.
        out_path = Path(filepath or "data.dat")
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        with open(tmp_path, "wb") as file:
            pickle.dump(self, file)  # type: ignore
        self._atomic_replace(tmp_path, out_path)

        if verbose:
            self._print_success()

    def to_json_tensors(self, filepath: str | Path | None = None, verbose=True):
        """Write the observations of the latest step to JSON, as raw tensor rows.

        Superseded by to_json, which labels the columns from the problem definition.
        Kept as a fallback for reading back files written in this shape.
        """
        if verbose:
            print("Saving observations to JSON... ", end="")

        q = self._batch_size
        n_observations = self._X.shape[0]

        def _last(tensor):
            return tensor[-q:].detach().cpu().numpy().tolist() if tensor is not None else None

        payload = {
            "datetime": self._datetime.isoformat(),
            "experiment_type": self.optimizer_type,
            "observation_n": list(range(n_observations - q, n_observations)),
            "batch_size": q,
            "data": {
                "source": self._source[-q:],
                "X": _last(self._X),
                "Y_obj": _last(self._Y_obj),
                "Y_obj_var": _last(self._Y_obj_var),
                "Y_con": _last(self._Y_con),
                "Y_con_var": _last(self._Y_con_var),
                "Y_trk": _last(self._Y_trk),
                "Y_trk_var": _last(self._Y_trk_var),
            },
        }

        with open(Path(filepath or "experiment.json"), "w") as file:
            json.dump(payload, file, indent=2)

        if verbose:
            self._print_success()

    def to_json(self, filepath: str | Path | None = None, verbose=True):
        """Write the observations of the latest step to JSON, one record per observation.

        Each record carries its position in the run, what produced it, and its values
        named from the problem definition rather than left as bare tensor columns.
        """
        if verbose:
            print("Saving observations to JSON... ", end="")

        q = self._batch_size
        n_observations = self._X.shape[0]

        def _last(tensor):
            """The batch's rows as lists, or one None per observation when the run
            records nothing for that tensor."""
            if tensor is None:
                return [None] * q
            return tensor[-q:].detach().cpu().numpy().tolist()

        def _named(cfgs, values, variances):
            """label -> value for one observation, each with its <label>_var partner."""
            named = {}
            for i, cfg in enumerate(cfgs or []):
                named[cfg.label] = None if values is None else values[i]
                named[f"{cfg.label}_var"] = None if variances is None else variances[i]
            return named

        X = _last(self._X)
        Y_obj, Y_obj_var = _last(self._Y_obj), _last(self._Y_obj_var)
        Y_con, Y_con_var = _last(self._Y_con), _last(self._Y_con_var)
        Y_trk, Y_trk_var = _last(self._Y_trk), _last(self._Y_trk_var)
        par_labels = [cfg.label for cfg in self._objective.par_cfg or []]

        data = []
        for i in range(q):
            n = n_observations - q + i
            data.append({
                "observation_n": n,
                "source": self._source[n],
                "parameters": dict(zip(par_labels, X[i])),
                "objectives": _named(self._objective.obj_cfg, Y_obj[i], Y_obj_var[i]),
                "constraints": _named(self._objective.ineq_Y_con_cfg, Y_con[i], Y_con_var[i]),
                "trackers": _named(self._objective.trk_cfg, Y_trk[i], Y_trk_var[i]),
            })

        payload = {
            "datetime": self._datetime.isoformat(),
            # Which arm proposed the point - this the optimizer does know about
            # itself, unlike whether the measurement came from a real rig or a
            # simulated trial, which nothing here is in a position to say.
            "optimizer": self.optimizer_type,
            "batch_size": q,
            "data": data,
        }
        with open(Path(filepath or "experiment.json"), "w") as file:
            json.dump(payload, file, indent=2)

        if verbose:
            self._print_success()

    @classmethod
    def from_file(cls, filepath: str | Path = None):
        """Load an optimizer instance from a file.
        If no filepath is provided, load the most recent .dat file from the current directory.
        """
        if filepath is None:
            files = glob.glob("*.dat")
            if not files:
                raise FileNotFoundError("No .dat files found in current directory")
            filepath = max(files, key=os.path.getctime)

        with open(filepath, "rb") as f:
            return pickle.load(f)
