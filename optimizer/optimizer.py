import copy
import inspect
import pickle
import warnings
import torch
import datetime
import time
import glob
import os
import botorch
import gpytorch
import pandas as pd
from pathlib import Path
from typing import Union, Type
from gpytorch.kernels import Kernel
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling import SobolQMCNormalSampler, MCSampler
from botorch.utils.multi_objective import is_non_dominated, Hypervolume
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import AcquisitionFunction, MCAcquisitionFunction
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase, MCMultiObjectiveBase
from samplers.samplers import SamplerBase, SobolSampler


class BayesianOptimizer:
    """
    A wrapper around BoTorch for Bayesian Optimization supporting both
    single-objective and multi-objective optimization.
    This class is designed to work with maximization problems only.
    For minimization, objective values must be negated.
    """

    def __init__(
            self,
            device: torch.device,
            dtype: torch.device.type,
            objective: MCSingleObjectiveBase | MCMultiObjectiveBase,
            X: torch.Tensor | None = None,
            Y_obj: torch.Tensor | None = None,
            Y_obj_var: torch.Tensor | None = None,
            Y_con: torch.Tensor | None = None,
            Y_con_var: torch.Tensor | None = None,
            Y_trk: torch.Tensor | None = None,
            Y_trk_var: torch.Tensor | None = None,
            acqf: Type[AcquisitionFunction] | None = None,
            kernel: Kernel | None = None,
            batch_size: int = 1,
            mc_samples: int = 256,
            raw_samples: int = 1024,
            n_acqf_opt_max_iter: int = 250,
            n_acqf_opt_restarts: int = 20,
            n_model_fit_restarts: int = 20,
    ):

        # === Device Attributes ===
        self.device = device
        self.dtype = dtype

        # === State attributes ===
        self._new_X: torch.Tensor | None = None
        self._model: ModelListGP | None = None
        self._mll: SumMarginalLogLikelihood | None = None
        self._ref_point: torch.Tensor | None = None
        self._acquisition_function_list: list[AcquisitionFunction] | None = None
        self._acqf_instance: AcquisitionFunction | None = None
        self._partitioning: torch.Tensor | None = None
        # self._pareto_front: torch.Tensor | None = None

        self._n_initial_samples: int | None = None
        self._feasible_mask: torch.Tensor | None = None

        # For single-objective
        self._best_f: torch.Tensor | None = None
        self._best_f_mask: torch.Tensor | None = None
        self._best_feasible_Y: torch.Tensor | None = None
        self._best_feasible_X: torch.Tensor | None = None
        # For multi-objective
        self._pareto_front_mask: torch.Tensor | None = None
        self._feasible_pareto_front_Y: torch.Tensor | None = None
        self._feasible_pareto_front_X: torch.Tensor | None = None

        # === Experiment Attributes ===
        self._datetime = datetime.datetime.now()
        self.objective = objective
        self._X: torch.Tensor = X
        if self._X is not None:
            self._n_initial_samples = self._X.shape[0]
        self._Y_obj: torch.Tensor = Y_obj
        self._Y_obj_var: torch.Tensor = Y_obj_var
        self._Y_con: torch.Tensor = Y_con
        self._Y_con_var: torch.Tensor = Y_con_var
        self._Y_trk: torch.Tensor = Y_trk
        self._Y_trk_var: torch.Tensor = Y_trk_var

        # === Optimization attributes ===
        self._acqf = acqf
        self._kernel = kernel
        self._sampler = None
        self._n_acqf_opt_max_iter = n_acqf_opt_max_iter  # Number of iterations for acquisition function optimization
        self._n_acqf_opt_restarts = n_acqf_opt_restarts  # The number of initial guesses used to optimize the acquisition function.
        self._n_model_fit_restarts = n_model_fit_restarts  # Max number of model fit attempts.
        self._batch_size = batch_size  # Number of candidates to be generated in parallel in each optimization step
        self._num_mc_samples = mc_samples  # Number of samples drawn from the predictive posterior distribution to estimate the acquisition function
        self._num_raw_samples = raw_samples  # Number of random points sampled in the search space to initialize the optimizer that maximizes the acquisition function

        # === Metrics ===
        self._hypervolume: list[float] = []  # for multi-objective
        self._best_values: list[float] = []  # For single-objective
        self._elapsed_time: list[float] = []

        # === Warnings ===
        self._warnings = []

    """ =========================== """
    """ ===== PICKLING HELPER ===== """
    """ =========================== """

    def __getstate__(self):
        state = self.__dict__.copy()
        # List attributes you want to exclude from pickling
        attrs_to_exclude = [
            '_acquisition_function',
            "_sampler",
            "_acq_func_list"
        ]
        for attr in attrs_to_exclude:
            state.pop(attr, None)  # remove if present
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize excluded attributes if needed
        self._transient = None

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

    """ =========================== """
    """ ===== CUDA PROPERTIES ===== """
    """ =========================== """

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

    """ =================================== """
    """ ===== EXPERIMENTAL PROPERTIES ===== """
    """ =================================== """

    @property
    def datetime(self):
        return self._datetime

    @property
    def objective(self) -> Union[MCObjectiveBase,]:
        return self._objective

    @property
    def X(self) -> torch.Tensor | None:
        return self._X

    @property
    def X_baseline(self):
        return self.X  # This is required for dynamic instantiation of acqf.

    @property
    def Y_obj(self) -> torch.Tensor | None:
        return self._Y_obj

    @property
    def Y_obj_var(self) -> torch.Tensor | None:
        return self._Y_obj_var

    @property
    def Y_con(self) -> torch.Tensor | None:
        return self._Y_con

    @property
    def Y_con_var(self) -> torch.Tensor | None:
        return self._Y_con_var

    @property
    def Y_trk(self) -> torch.Tensor | None:
        return self._Y_trk

    @property
    def Y_trk_var(self) -> torch.Tensor | None:
        return self._Y_trk_var

    @property
    def acqf(self) -> Type[AcquisitionFunction]:
        return self._acqf

    @property
    def sampler(self) -> SamplerBase:
        return self._sampler

    @property
    def n_acqf_opt_iter(self) -> int:
        return self._n_acqf_opt_max_iter

    @property
    def n_acqf_opt_restarts(self) -> int:
        return self._n_acqf_opt_restarts

    @property
    def n_model_fit_restarts(self) -> int:
        return self._n_model_fit_restarts

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def num_mc_samples(self):
        return self._num_mc_samples

    @property
    def num_raw_samples(self):
        return self._num_raw_samples

    @objective.setter
    def objective(self, objective: Union[MCObjectiveBase]):
        if not isinstance(objective, Union[MCObjectiveBase]):
            raise ValueError("Objective is not compatible.")
        self._objective = objective

    @X.setter
    def X(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must a torch.Tensor.")
        if X.shape[-1] != self.objective.dim:
            raise ValueError("X must have the same number of dimensions as objective.")
        if self._X is None:
            self._n_initial_samples = X.shape[0]
        self._X = X.to(self._device, self._dtype)

    @Y_obj.setter
    def Y_obj(self, Y_obj: torch.Tensor):
        if not isinstance(Y_obj, Union[torch.Tensor, None]):
            raise ValueError("Y_obj must be of type torch.Tensor or None.")
        if Y_obj is not None and Y_obj.shape[-1] != self.objective.num_objectives:
            raise ValueError("Y_obj must have the same number of dimensions as objective.")
        self._Y_obj = Y_obj.to(self._device, self._dtype) if Y_obj is not None else None

    @Y_obj_var.setter
    def Y_obj_var(self, Y_obj_var: torch.Tensor | None = None):
        if not isinstance(Y_obj_var, Union[torch.Tensor, None]):
            raise ValueError("Y_obj_var must be of type torch.Tensor or None.")
        if Y_obj_var is not None and Y_obj_var.shape[-1] != self.objective.num_objectives:
            raise ValueError("Y_obj_var must have the same number of dimensions as objective.")
        self._Y_obj_var = Y_obj_var.to(self._device, self._dtype) if Y_obj_var is not None else None

    @Y_con.setter
    def Y_con(self, Y_con: torch.Tensor | None):
        if not isinstance(Y_con, Union[torch.Tensor, None]):
            raise ValueError("Y_con must be of type torch.Tensor or None.")
        if Y_con is not None and Y_con.shape[-1] != self.objective.num_constraints:
            raise ValueError("Y_con must have the same number of ineq_Y_con_cfg as objective.")
        self._Y_con = Y_con.to(self._device, self._dtype) if Y_con is not None else None

    @Y_con_var.setter
    def Y_con_var(self, Y_con_var: torch.Tensor | None = None):
        if not isinstance(Y_con_var, Union[torch.Tensor, None]):
            raise ValueError("Y_con_var must be of type torch.Tensor or None.")
        if Y_con_var is not None and Y_con_var.shape[-1] != self.objective.num_constraints:
            raise ValueError("Y_con_var must have the same number of ineq_Y_con_cfg as objective.")
        self._Y_con_var = Y_con_var.to(self._device, self._dtype) if Y_con_var is not None else None

    @Y_trk.setter
    def Y_trk(self, Y_track: torch.Tensor | None):
        if not isinstance(Y_track, torch.Tensor | None):
            raise ValueError("Y_trk must be of type torch.Tensor or None.")
        self._Y_trk = Y_track

    @Y_trk_var.setter
    def Y_trk_var(self, Y_track_var: torch.Tensor | None):
        if not isinstance(Y_track_var, torch.Tensor | None):
            raise ValueError("Y_trk_var must be of type torch.Tensor or None.")
        self._Y_trk_var = Y_track_var

    @acqf.setter
    def acqf(self, af_type):
        if not (inspect.isclass(af_type) and issubclass(af_type, AcquisitionFunction)):
            raise ValueError(
                f"Acquisition function must be a class inheriting from AcquisitionFunction. "
                f"Got {type(af_type).__name__}: {af_type}"
            )
        self._acqf = af_type

    @sampler.setter
    def sampler(self, sampler):
        if not isinstance(sampler, SamplerBase):
            raise ValueError("SamplerBase type must be of type SamplerType")
        self._sampler = sampler

    @n_acqf_opt_iter.setter
    def n_acqf_opt_iter(self, n_acqf_opt_iter):
        if not isinstance(n_acqf_opt_iter, int):
            raise ValueError("n_acqf_opt_max_iter must be of type int")
        self._n_acqf_opt_max_iter = n_acqf_opt_iter

    @n_acqf_opt_restarts.setter
    def n_acqf_opt_restarts(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_acqf_opt_restarts must be of type int")
        self._n_acqf_opt_restarts = value

    @n_model_fit_restarts.setter
    def n_model_fit_restarts(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_model_fit_restarts must be of type int")
        self._n_model_fit_restarts = value

    @batch_size.setter
    def batch_size(self, batch_size: int):
        if not isinstance(batch_size, int):
            raise ValueError("batch_size must be of type int")
        self._batch_size = batch_size

    @num_mc_samples.setter
    def num_mc_samples(self, mc_samples: int):
        if not isinstance(mc_samples, int):
            raise ValueError("num_mc_samples must be of type int")
        self._num_mc_samples = mc_samples

    @num_raw_samples.setter
    def num_raw_samples(self, raw_samples: int):
        if not isinstance(raw_samples, int):
            raise ValueError("num_raw_samples must be of type int")
        self._num_raw_samples = raw_samples

    """ ============================ """
    """ ===== STATE PROPERTIES ===== """
    """ ============================ """

    @property
    def model(self) -> SingleTaskGP | ModelListGP | None:
        if self._model is None:
            print("A model has not been generated yet.")
        return self._model

    @property
    def mll(self) -> SumMarginalLogLikelihood | None:
        if self._mll is None:
            print("A model has not been generated yet.")
        return self._mll

    @property
    def ref_point(self) -> torch.Tensor | None:
        if self._ref_point is None:
            print("A reference point (in maximization space) has not been computed yet.")
        return self._ref_point

    @property
    def acquisition_function_list(self) -> list[AcquisitionFunction] | None:
        if self._acquisition_function_list is None:
            print("The acquisition function has not been initialized yet.")
        return self._acquisition_function_list

    @property
    def partitioning(self) -> NondominatedPartitioning | None:
        if self._partitioning is None:
            print("A partitioning has not been computed yet.")
        return self._partitioning

    @property
    def pareto_front_mask(self) -> torch.Tensor | None:
        if self._pareto_front_mask is None:
            print("A pareto front has not been computed yet.")
        return self._pareto_front_mask

    @property
    def best_f_mask(self) -> torch.Tensor | None:
        if self._best_f_mask is None:
            print("A best f mask has not been computed yet.")
        return self._best_f_mask

    @property
    def new_X(self) -> torch.Tensor | None:
        if self._new_X is None:
            print("A new_X has not been computed yet.")
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
    def acqf_instance(self) -> AcquisitionFunction | None:
        if self._acqf_instance is None:
            print("An acquisition function has not been instantiated yet.")
        return self._acqf_instance

    @property
    def n_initial_samples(self) -> int:
        return self._n_initial_samples

    @n_initial_samples.setter
    def n_initial_samples(self, n: int):
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n_initial_samples must be a positive integer")
        if self._n_initial_samples is None:
            self._n_initial_samples = n

    @property
    def feasible_mask(self) -> torch.Tensor | None:
        """Boolean mask indicating which observations are feasible."""
        return self._feasible_mask

    @property
    def best_feasible_X(self) -> torch.Tensor | None:
        """X value at best feasible observation (single-objective)."""
        return self._best_feasible_X

    @property
    def best_feasible_Y(self) -> torch.Tensor | None:
        """Y value at best feasible observation (single-objective)."""
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

    @property
    def kernel(self) -> Kernel:
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: Kernel) -> None:
        if not isinstance(kernel, Kernel):
            raise ValueError("kernel_type must be of type Kernel.")
        self._kernel = kernel

    """ ===================== """
    """ ===== Optimizer ===== """
    """ ===================== """

    def optimize(self, verbose=True):
        t0 = time.monotonic()

        # === 1. Data Processing (Historical Analysis) ===
        # First, find out what is feasible in the current dataset
        self._compute_feasible_mask(verbose=verbose)

        # Second, identify the current "Best" (best_f or Pareto Front).
        # This MUST happen before metrics because metrics log these values.
        self._compute_acquisition_function_reference(verbose=verbose)

        # Third, log the progress (Hypervolume or Best Value)
        self._compute_metrics(verbose=verbose)

        # === 2. Surrogate Modeling ===
        # Fit the GPs to the data we just analyzed
        self._initialize_model(verbose=verbose)
        self._fit_model(verbose=verbose)

        # === 3. Acquisition Strategy ===
        # Initialise and optimize the acquisition function
        self._initialize_acquisition_function(verbose=verbose)
        self._optimize_acquisition_function(verbose=verbose)

        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)

        if verbose:
            print(f"Optimization step completed in {t1 - t0:.2f}s")

    def _compute_feasible_mask(self, verbose=True):
        """ Computes feasible mask for all observed data. A point is defined feasible only
        if it satisfies both Input and Output ineq_Y_con_cfg. """
        if verbose:
            print("Computing feasibility mask... ", end="")

        feasible_input_mask = self._compute_feasible_input_mask()
        feasible_output_mask = self._compute_feasible_output_mask()
        self._feasible_mask = feasible_input_mask & feasible_output_mask

        if verbose:
            n_points = self.X.shape[0]
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
        Single-objective: _best_f
        Multi-objective: hypervolume."""
        if self._objective.num_obj == 1:
            self._compute_best_value(verbose=verbose)
        else:
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

    def _initialize_model(self, verbose=True):
        """ Initialize Gaussian Process model(s) for the objectives and ineq_Y_con_cfg.

        This method prepares the training dataset by combining the objective and constraint
        observations (and optionally their variances). Then, it creates an independent
        SingleTaskGP model for each output dimension (including objectives and ineq_Y_con_cfg).
        The SingleTaskGP are finally combined into a ModelListGP to jointly represent the full
        multi-output model.

        Important: The GP model in BoTorch is a pure regression model: it simply fits the data
        it receives. It does not know or care about whether the model is for objectives to
        minimize or maximize.

        Note: by setting an input transform and an outcome transform, input (X) and output data (Y) are
        transformed and untransformed accordingly across the whole optimization pipeline, including
        the optimization of the acquisition function. For example, by setting the outcome transform to
        standardization, the Ys are standardized before the optimization and unstandardized right after.
        However, if a penalty is added in the forward method, this is not standardized properly and a
        pre-factor (or scaling) results in different penalty weights.

        Note: the kernel instance must be deep-copied, otherwise the same kernel is shared across
        all the models. This is critical in multi objective problems. """

        if verbose:
            print("Initializing model... ", end="")

        # Prepare dataset by concatenating the objectives and initialize models - one model
        # for each objective (or observable). Note that a base noise of 1e-4 is the default value
        train_x, train_y, train_y_var = self._prepare_training_dataset()
        models = []
        for i in range(0, train_y.shape[-1]):
            models.append(
                SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y[..., i: i + 1],
                    train_Yvar=(train_y_var[..., i: i + 1] if train_y_var is not None else None),
                    input_transform=Normalize(d=self.objective.dim, bounds=self.objective.bounds),
                    outcome_transform=Standardize(m=1),
                    covar_module=copy.deepcopy(self._kernel),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-4)),
                )
            )

        self._model = ModelListGP(*models)
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        if verbose:
            self._print_success()

    def _prepare_training_dataset(self):
        """ Prepare the training dataset for fitting the surrogate model. """
        train_x = self.X.clone()
        train_y = self._Y_obj.clone()

        # Concatenate ineq_Y_con_cfg if available (not None)
        if self._Y_con is not None:
            train_y_con = self._Y_con.clone()
            train_y = torch.cat((train_y, train_y_con), dim=-1)

        # Define train_y_var
        if self._Y_obj_var is not None and self._Y_con_var is not None:
            train_y_var = self._Y_obj_var.clone()
            train_y_con_var = self._Y_con_var.clone()
            train_var = torch.cat((train_y_var, train_y_con_var), dim=-1)
        else:
            train_var = None

        return train_x, train_y, train_var

    def _fit_model(self, restart_on_error=True, verbose=True):
        if not isinstance(self._model, ModelListGP):
            raise ValueError("Model must be initialised before fitting.")

        max_attempts = self._n_model_fit_restarts + 1

        for attempt in range(1, max_attempts + 1):
            try:
                if verbose:
                    print(f"Fitting model (Attempt {attempt}/{max_attempts})... ", end="")

                botorch.fit_gpytorch_mll(self._mll)

                if verbose:
                    self._print_success()

                return

            except Exception as e:
                # If we aren't allowed to restart, or we've run out of attempts, raise the error
                if not restart_on_error or attempt == max_attempts:
                    raise RuntimeError(f"Fitting failed after {attempt} attempts. Last error: {e}")

                if verbose:
                    print(f"Encountered error: {e}. Reinitializing model...")

                # Re-initialize the model to get new random starting hyperparameters
                self._initialize_model(verbose=False)

    def _compute_acquisition_function_reference(self, verbose=True):
        """Compute the reference value for the acquisition function.
        Single-objective: best value (_best_f)
        Multi-objective: reference point in high dimensional space."""
        if self._objective.num_obj == 1:
            self._compute_best_f(verbose=verbose)
        else:
            self._compute_reference_point(verbose=verbose)
            self._compute_pareto_front(verbose=verbose)

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

        if feasible_X is not None:
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
                self._print_success(msg=f"({self._best_feasible_Y.item():.4f} in max. space)")

        else:
            self._best_feasible_Y = None
            self._best_feasible_X = None
            self._best_f = -float("inf")
            self._best_values.append(None)
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
        Compute the Pareto front including ineq_Y_con_cfg. Note that since
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

    def _initialize_acquisition_function(self, verbose=True):
        """ Initialize an acquisition function instance using the acqf. """
        with warnings.catch_warnings(record=True) as caught:

            sig = inspect.signature(self.acqf.__init__)
            kwargs = {}

            # Initialize sampler if required by the acqf
            if issubclass(self._acqf, MCAcquisitionFunction):
                self._initialize_sampler(verbose=verbose)

            # Initialize partitioning if required by the acqf
            if "partitioning" in sig.parameters and getattr(self, "_partitioning", None) is None:
                self._initialize_partitioning(verbose=verbose)

            if verbose:
                name = self.acqf.__name__
                print(f"Initializing acquisition function of type {name}... ", end="", flush=True)

            for name, param in sig.parameters.items():
                if name in ('self', 'args', 'kwargs'):
                    continue

                if hasattr(self, name):
                    kwargs[name] = getattr(self, name)
                elif hasattr(self.objective, name):
                    kwargs[name] = getattr(self.objective, name)
                elif param.default is not inspect.Parameter.empty:
                    kwargs[name] = param.default

            self._acqf_instance = self.acqf(**kwargs)

        self._warnings.extend(caught)

        if verbose:
            self._print_success()
            self._print_caught_warnings()

        self._warnings = []

    def _initialize_partitioning(self, verbose=True):
        if verbose:
            print("Initializing partitioning... ", end="")

        # Make sure feasibility mask exists
        if getattr(self, "_feasible_mask", None) is None:
            self._feasible_mask = self._compute_feasible_output_mask()

        # Use observed objective values ONLY
        Y_obj = self._Y_obj[self._feasible_mask]  # (n_feas, M)
        if Y_obj.numel() == 0:
            self._partitioning = None
            return

        # Convert to maximization space
        Y = Y_obj.clone()
        Y[..., self.objective.to_minimize] *= -1

        self._partitioning = NondominatedPartitioning(
            ref_point=self._ref_point,
            Y=Y
        )

        if verbose:
            self._print_success()

    def _initialize_sampler(self, verbose=True):
        """Initialize sampler for Monte Carlo acquisition functions."""
        if verbose:
            print("Initializing sampler... ", end="")

        self._sampler = SobolQMCNormalSampler(torch.Size([self._num_mc_samples]))

        if verbose:
            self._print_success()

    def _compute_feasible_XY(self, verbose=False) -> (torch.Tensor, torch.Tensor):
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

    def _compute_infeasible_XY(self, verbose=False) -> (torch.Tensor, torch.Tensor):
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

    def _optimize_acquisition_function(self, verbose=True):

        if verbose:
            print(f"Optimizing acquisition function... ", end="")

        with warnings.catch_warnings(record=True) as caught:

            if isinstance(self._acqf_instance, qLogNParEGO):
                self._new_X, _ = optimize_acqf_list(
                    acq_function_list=self._acquisition_function_list,
                    bounds=self._objective.bounds,
                    num_restarts=self._n_acqf_opt_restarts,
                    raw_samples=self._num_raw_samples,
                    options={"batch_limit": 5, "maxiter": self._n_acqf_opt_max_iter},
                )
            else:
                # If nonlinear inequality **input** ineq_Y_con_cfg are provided, use a custom initial condition
                # generator that selects "num_restarts" points. These points are distributed according to
                # "fraction_of_previous_X" between the optimal points and randomly generated points.
                self._new_X, _ = optimize_acqf(
                    acq_function=self._acqf_instance,
                    bounds=self._objective.bounds,
                    q=self._batch_size,
                    num_restarts=self._n_acqf_opt_restarts,
                    raw_samples=self._num_raw_samples,
                    options={"maxiter": self._n_acqf_opt_max_iter, "disp": False},
                    inequality_constraints=self.objective.lin_ineq_X_con,
                    equality_constraints=self.objective.lin_eq_X_con,
                    nonlinear_inequality_constraints=self.objective.nonlin_ineq_X_con,
                    sequential=True,
                    ic_generator=self._ic_generator
                    if self.objective.nonlin_ineq_X_con is not None
                    else None,
                    **{
                        "fraction_of_previous_X": 0.8,
                        "noise_scale": 0,
                    } if self.objective.nonlin_ineq_X_con is not None
                    else {}
                )

        self._warnings.extend(caught)

        if verbose:
            self._print_success(msg=f"New X: {self._new_X.detach().cpu().numpy()}")
            self._print_caught_warnings()

        self._warnings = []

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

    def _ic_generator(
            self,
            acq_function,  # noqa
            q: int,  # noqa
            num_restarts: int,
            raw_samples: int,
            **kwargs: dict
    ) -> torch.Tensor:
        """
        Generates initial conditions for constrained acquisition optimization.
        Mixes previous points and new samples from a constraint-aware sampler.
        """

        frac_prev = kwargs.get("fraction_of_previous_X", 0.5)
        noise_scale = kwargs.get("noise_scale", 0.0)
        sampler = SobolSampler(device=self._device, dtype=self._dtype, objective=self.objective)

        # 1. Collect previous observations
        X_feas, Y_feas = self._compute_feasible_XY()
        Y = Y_feas.clone()
        X = X_feas.clone()

        if X_feas.shape[0] > 0:
            # Adjust for minimization (BoTorch assumes maximization)
            Y[..., self.objective.to_minimize] *= -1
            n_requested = int(frac_prev * num_restarts)

            # LOGIC SPLIT: Multi-Objective vs Single-Objective
            if self.objective.num_obj > 1:
                # Multi-objective: use Pareto front
                mask = is_non_dominated(Y)
                X_candidates = X[mask]
                n_to_take = min(int(frac_prev * num_restarts), X_candidates.shape[0])
                idx = torch.randperm(X_candidates.shape[0])[:n_to_take]
                prev_points = X_candidates[idx]
            else:
                # Single-objective: use feasible top performing points
                _, indices = torch.sort(Y.squeeze(-1), descending=True)
                n_to_take = min(n_requested, indices.shape[0])
                prev_points = X[indices[:n_to_take]]
                shuffle_idx = torch.randperm(prev_points.shape[0])
                prev_points = prev_points[shuffle_idx]
        else:
            # No feasible points found in history yet
            prev_points = torch.empty(0, self.objective.dim, device=self._device)

        # 2. Generate new points (Exploration)
        n_new = max(0, num_restarts - prev_points.shape[0])
        raw_X = sampler.draw_samples(n=raw_samples)

        if n_new > raw_X.shape[0]:
            raise RuntimeError(
                f"Requested {n_new} new points, but only {raw_X.shape[0]} feasible raw samples available.")
        idx = torch.randperm(raw_X.shape[0])[:n_new]
        new_points = raw_X[idx]

        # Add small noise for exploration
        if noise_scale > 0:
            new_points += noise_scale * torch.randn_like(new_points)

        # 3. Combine previous points and new points
        combined = torch.cat([prev_points, new_points], dim=0)

        # Ensure proper shape for BoTorch: (num_restarts, q, d)
        return combined.unsqueeze(1)  # q=1 for sequential optimization

    def compute_acquisition_function_value_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        acq_val = self._acqf_instance(X)
        if verbose:
            print(f"Acquisition function value at {X.detach().cpu().numpy()}: {acq_val.detach().cpu().numpy()}")

    def compute_posterior_mean_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        posterior = self._model.posterior(X)
        if verbose:
            print(f"Posterior mean at {X.detach().cpu().numpy()}: {posterior.mean.detach().cpu().numpy()}")
        return posterior

    def compute_posterior_variance_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        posterior_var = self._model.posterior(X).variance
        if verbose:
            print(f"Posterior variance at {X.detach().cpu().numpy()}: {posterior_var.detach().cpu().numpy()}")
        return posterior_var

    def update_XY(self, new_X: torch.Tensor, new_Y_obj: torch.Tensor, new_Y_trk: torch.Tensor | None = None,
                  new_Y_obj_var: torch.Tensor | None = None,
                  new_Y_con: torch.Tensor | None = None, new_Y_con_var=None,
                  new_Y_trk_var: torch.Tensor | None = None) -> None:
        self.update_X(new_X)
        self.update_Y_obj(new_Y_obj, new_Y_obj_var)
        self.update_Y_con(new_Y_con, new_Y_con_var)
        self.update_Y_trk(new_Y_trk, new_Y_trk_var)

    def update_X(self, new_X: torch.Tensor):
        if new_X is not None:
            new_X = new_X.to(self._device, self._dtype)
            self.X = torch.cat([self.X, new_X], dim=0)

    def update_Y_obj(self, new_Y_obj: torch.Tensor, new_Y_obj_var: torch.Tensor or None = None):
        if new_Y_obj is not None:
            new_Y_obj = new_Y_obj.to(self._device, self._dtype)
            self._Y_obj = torch.cat([self._Y_obj, new_Y_obj], dim=0)
        if new_Y_obj_var is not None:
            new_Y_obj_var = new_Y_obj_var.to(self._device, self._dtype)
            self._Y_obj_var = torch.cat([self._Y_obj_var, new_Y_obj_var], dim=0)

    def update_Y_con(self, new_Y_con: torch.Tensor, new_Y_con_var: torch.Tensor or None = None):
        if new_Y_con is not None:
            new_Y_con = new_Y_con.to(self._device, self._dtype)
            self._Y_con = torch.cat([self._Y_con, new_Y_con], dim=0)
        if new_Y_con_var is not None:
            new_Y_con_var = new_Y_con_var.to(self._device, self._dtype)
            self._Y_con_var = torch.cat([self._Y_con_var, new_Y_con_var], dim=0)

    def update_Y_trk(self, new_Y_track: torch.Tensor, new_Y_track_var: torch.Tensor or None = None):
        if new_Y_track is not None:
            new_Y_track = new_Y_track.to(self._device, self._dtype)
            self._Y_trk = torch.cat([self._Y_trk, new_Y_track], dim=0)
        if new_Y_track_var is not None:
            new_Y_track_var = new_Y_track_var.to(self._device, self._dtype)
            self._Y_trk_var = torch.cat([self._Y_trk_var, new_Y_track_var], dim=0)

    """ =============== """
    """ ===== I/O ===== """
    """ =============== """

    def to_file(self, filepath: Path or str = None):
        filepath = Path(filepath or "data.dat")
        stem = filepath.stem
        suffix = filepath.suffix
        save_path = Path.cwd() / filepath.parent / f"{stem}_000{suffix}"

        i = 0
        while save_path.exists():
            i += 1
            save_path = Path.cwd() / filepath.parent / f"{stem}_{i:03d}{suffix}"

        with open(save_path, "wb") as file:
            pickle.dump(self, file)  # type: ignore

    def to_csv(self, filepath: str | None = None, verbose=True):
        if verbose:
            print("Saving optimizer to CSV... ", end="")

        data = torch.cat((self.X, self.Y_obj), dim=1)
        cols = []
        cols.extend([x.label for x in self.objective.par_cfg])
        cols.extend([x.label for x in self.objective.obj_cfg])

        if self.Y_obj_var is not None:
            cols.extend([x.label.lower() + " var" for x in self.objective.obj_cfg])
            data = torch.cat((data, self.Y_obj_var), dim=1)
        if self.Y_con is not None:
            cols.extend([x.label.lower() for x in self.objective.ineq_Y_con_cfg])
            data = torch.cat((data, self.Y_con), dim=1)
        if self.Y_con_var is not None:
            cols.extend([x.label.lower() + " var" for x in self.objective.ineq_Y_con_cfg])
            data = torch.cat((data, self.Y_con_var), dim=1)
        if self.Y_trk is not None:
            cols.extend([x.label.lower() for x in self.objective.trk_cfg])
            data = torch.cat((data, self.Y_trk), dim=1)
        if self.Y_trk_var is not None:
            cols.extend([x.label.lower() + " var" for x in self.objective.trk_cfg])
            data = torch.cat((data, self.Y_trk_var), dim=1)
        if self.feasible_mask is not None:
            cols.append("feasible")
            data = torch.cat((data, self.feasible_mask.unsqueeze(-1)), dim=1)
        if self.pareto_front_mask is not None:
            cols.append("pareto front")
            data = torch.cat((data, self.pareto_front_mask), dim=1)
        if self.best_f_mask is not None:
            cols.append("best value")
            data = torch.cat((data, self.best_f_mask), dim=1)

        df = pd.DataFrame(data.detach().cpu().numpy(), columns=cols)

        filepath = Path(filepath or "data.csv")
        stem = filepath.stem
        suffix = filepath.suffix
        save_path = Path.cwd() / filepath.parent / f"{stem}_000{suffix}"

        i = 0
        while save_path.exists():
            i += 1
            save_path = Path.cwd() / filepath.parent / f"{stem}_{i:03d}{suffix}"

        df.to_csv(save_path, index=False)

        if verbose:
            self._print_success()

    @classmethod
    def from_file(cls, filepath: str | Path = None):
        """Load a MOBO instance from a file.
        If no filepath is provided, load the most recent .dat file from the current directory.
        """
        if filepath is None:
            files = glob.glob("*.dat")
            if not files:
                raise FileNotFoundError("No .dat files found in current directory")
            filepath = max(files, key=os.path.getctime)

        with open(filepath, "rb") as f:
            return pickle.load(f)
