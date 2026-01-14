import pickle
import warnings
from typing import Union
import torch
import datetime
import time
import glob
import os
from pathlib import Path
import botorch
import gpytorch
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling import SobolQMCNormalSampler, MCSampler
from botorch.utils.multi_objective import is_non_dominated, Hypervolume, get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import AcquisitionFunction
from gpytorch.constraints import GreaterThan
from bayesian_optimizer.acquisition_function import AcquisitionRuntimeParams, AcquisitionFunctionFactory
from bayesian_optimizer.kernel import KernelFactory
from objectives.base_class import MCObjectiveBase
from samplers.samplers import Sampler
from gpytorch.mlls import SumMarginalLogLikelihood
from utils.bo_types import *


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
            objective: MCObjectiveBase,
            X: torch.Tensor | None = None,
            Y_obj: torch.Tensor | None = None,
            Y_obj_var: torch.Tensor | None = None,
            Y_con: torch.Tensor | None = None,
            Y_con_var: torch.Tensor | None = None,
            Y_track: torch.Tensor | None = None,
            Y_track_var: torch.Tensor | None = None,
            acquisition_function_factory: AcquisitionFunctionFactory | None = None,
            kernel_factory: KernelFactory | None = None,
            sampler_type: SamplerType = SamplerType.Sobol,
            batch_size: int = 1,
            mc_samples: int = 256,
            raw_samples: int = 1024,
            n_acqf_opt_max_iter: int = 250,
            n_acqf_opt_restarts: int = 20,
            n_model_fit_restarts: int = 20,
            ucb_beta: float = 2.0,

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
        self._partitioning: torch.Tensor | None = None
        # self._pareto_front: torch.Tensor | None = None

        self._acquisition_function_instance: AcquisitionFunction | None = None
        self._kernel_instance: Kernel | None = None
        self._sampler_instance: MCSampler | None = None

        self._n_initial_samples: int | None = None
        self._feasible_mask: torch.Tensor | None = None

        # For single-objective
        self._best_f: torch.Tensor | None = None
        self._best_feasible_Y: torch.Tensor | None = None
        self._best_feasible_X: torch.Tensor | None = None
        # For multi-objective
        self._feasible_pareto_front_Y: torch.Tensor | None = None
        self._feasible_pareto_front_X: torch.Tensor | None = None

        # === Experiment Attributes ===
        self._datetime = datetime.datetime.now()
        self.objective = objective
        self.X: torch.Tensor = X
        self.Y_obj: torch.Tensor = Y_obj
        self.Y_obj_var: torch.Tensor = Y_obj_var
        self.Y_con: torch.Tensor = Y_con
        self.Y_con_var: torch.Tensor = Y_con_var
        self.Y_track: torch.Tensor = Y_track
        self.Y_track_var: torch.Tensor = Y_track_var

        # === Optimization attributes ===
        self.acquisition_function_factory = acquisition_function_factory
        self.kernel_factory = kernel_factory
        self.sampler_type = sampler_type
        self.n_acqf_opt_iter = n_acqf_opt_max_iter  # Number of iterations for acquisition function optimization
        self.n_acqf_opt_restarts = n_acqf_opt_restarts  # The number of initial guesses used to optimize the acquisition function.
        self.n_model_fit_restarts = n_model_fit_restarts  # Max number of model fit attempts.
        self.batch_size = batch_size  # Number of candidates to be generated in parallel in each optimization step
        self.num_mc_samples = mc_samples  # Number of samples drawn from the predictive posterior distribution to estimate the acquisition function
        self.num_raw_samples = raw_samples  # Number of random points sampled in the search space to initialize the optimizer that maximizes the acquisition function
        self.ucb_beta = ucb_beta

        # === Metrics ===
        self._hypervolume: list[float] = []  # for multi-objective
        self._best_values: list[float] = []  # For single-objective
        self._elapsed_time: list[float] = []

    """ =========================== """
    """ ===== PICKLING HELPER ===== """
    """ =========================== """

    def __getstate__(self):
        state = self.__dict__.copy()
        # List attributes you want to exclude from pickling
        attrs_to_exclude = [
            '_acquisition_function_instance',
            "_sampler_instance",
            "_acq_func_list"
        ]
        for attr in attrs_to_exclude:
            state.pop(attr, None)  # remove if present
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize excluded attributes if needed
        self._transient = None

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

    @objective.setter
    def objective(self, objective: Union[MCObjectiveBase]):
        if not isinstance(objective, Union[MCObjectiveBase]):
            raise ValueError("Objective is not compatible.")
        self._objective = objective

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must a torch.Tensor.")
        if X.shape[-1] != self.objective.dim:
            raise ValueError("X must have the same number of dimensions as objective.")
        self._X = X.to(self._device, self._dtype)
        self.n_initial_samples = self._X.shape[0]

    @property
    def Y_obj(self) -> torch.Tensor | None:
        return self._Y_obj

    @Y_obj.setter
    def Y_obj(self, Y_obj: torch.Tensor):
        if not isinstance(Y_obj, Union[torch.Tensor, None]):
            raise ValueError("Y_obj must be of type torch.Tensor or None.")
        if Y_obj is not None and Y_obj.shape[-1] != self.objective.num_objectives:
            raise ValueError("Y_obj must have the same number of dimensions as objective.")
        self._Y_obj = Y_obj.to(self._device, self._dtype) if Y_obj is not None else None

    @property
    def Y_obj_var(self) -> torch.Tensor | None:
        return self._Y_obj_var

    @Y_obj_var.setter
    def Y_obj_var(self, Y_obj_var: torch.Tensor | None = None):
        if not isinstance(Y_obj_var, Union[torch.Tensor, None]):
            raise ValueError("Y_obj_var must be of type torch.Tensor or None.")
        if Y_obj_var is not None and Y_obj_var.shape[-1] != self.objective.num_objectives:
            raise ValueError("Y_obj_var must have the same number of dimensions as objective.")
        self._Y_obj_var = Y_obj_var.to(self._device, self._dtype) if Y_obj_var is not None else None

    @property
    def Y_con(self) -> torch.Tensor | None:
        return self._Y_con

    @Y_con.setter
    def Y_con(self, Y_con: torch.Tensor | None):
        if not isinstance(Y_con, Union[torch.Tensor, None]):
            raise ValueError("Y_con must be of type torch.Tensor or None.")
        if Y_con is not None and Y_con.shape[-1] != self.objective.num_constraints:
            raise ValueError("Y_con must have the same number of constraints as objective.")
        self._Y_con = Y_con.to(self._device, self._dtype) if Y_con is not None else None

    @property
    def Y_con_var(self) -> torch.Tensor | None:
        return self._Y_con_var

    @Y_con_var.setter
    def Y_con_var(self, Y_con_var: torch.Tensor | None = None):
        if not isinstance(Y_con_var, Union[torch.Tensor, None]):
            raise ValueError("Y_con_var must be of type torch.Tensor or None.")
        if Y_con_var is not None and Y_con_var.shape[-1] != self.objective.num_constraints:
            raise ValueError("Y_con_var must have the same number of constraints as objective.")
        self._Y_con_var = Y_con_var.to(self._device, self._dtype) if Y_con_var is not None else None

    @property
    def Y_track(self) -> torch.Tensor | None:
        return self._Y_track

    @Y_track.setter
    def Y_track(self, Y_track: torch.Tensor | None):
        if not isinstance(Y_track, torch.Tensor | None):
            raise ValueError("Y_track must be of type torch.Tensor or None.")
        self._Y_track = Y_track

    @property
    def Y_track_var(self) -> torch.Tensor | None:
        return self._Y_track_var

    @Y_track_var.setter
    def Y_track_var(self, Y_track_var: torch.Tensor | None):
        if not isinstance(Y_track_var, torch.Tensor | None):
            raise ValueError("Y_track_var must be of type torch.Tensor or None.")
        self._Y_track_var = Y_track_var

    @property
    def acquisition_function_factory(self) -> AcquisitionFunctionFactory:
        return self._acquisition_function_type

    @acquisition_function_factory.setter
    def acquisition_function_factory(self, af_type):
        if not isinstance(af_type, AcquisitionFunctionFactory):
            raise ValueError("Acquisition function type must be of type AcquisitionFunctionFactory.")
        self._acquisition_function_type = af_type

    @property
    def sampler_type(self) -> SamplerType:
        return self._sampler_type

    @sampler_type.setter
    def sampler_type(self, sampler_type):
        if not isinstance(sampler_type, SamplerType):
            raise ValueError("Sampler type must be of type SamplerType")
        self._sampler_type = sampler_type

    @property
    def n_acqf_opt_iter(self) -> int:
        return self._n_acqf_opt_max_iter

    @n_acqf_opt_iter.setter
    def n_acqf_opt_iter(self, n_acqf_opt_iter):
        if not isinstance(n_acqf_opt_iter, int):
            raise ValueError("n_acqf_opt_max_iter must be of type int")
        self._n_acqf_opt_max_iter = n_acqf_opt_iter

    @property
    def n_acqf_opt_restarts(self) -> int:
        return self._n_acqf_opt_restarts

    @n_acqf_opt_restarts.setter
    def n_acqf_opt_restarts(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_acqf_opt_restarts must be of type int")
        self._n_acqf_opt_restarts = value

    @property
    def n_model_fit_restarts(self) -> int:
        return self._n_model_fit_restarts

    @n_model_fit_restarts.setter
    def n_model_fit_restarts(self, value: int):
        if not isinstance(value, int):
            raise ValueError("n_model_fit_restarts must be of type int")
        self._n_model_fit_restarts = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        if not isinstance(batch_size, int):
            raise ValueError("batch_size must be of type int")
        self._batch_size = batch_size

    @property
    def num_mc_samples(self):
        return self._num_mc_samples

    @num_mc_samples.setter
    def num_mc_samples(self, mc_samples: int):
        if not isinstance(mc_samples, int):
            raise ValueError("num_mc_samples must be of type int")
        self._num_mc_samples = mc_samples

    @property
    def num_raw_samples(self):
        return self._num_raw_samples

    @num_raw_samples.setter
    def num_raw_samples(self, raw_samples: int):
        if not isinstance(raw_samples, int):
            raise ValueError("num_raw_samples must be of type int")
        self._num_raw_samples = raw_samples

    # === STATE properties ===
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
    def pareto_front(self) -> torch.Tensor | None:
        if self._pareto_front is None:
            print("A pareto front has not been computed yet.")
        return self._pareto_front

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
    def acquisition_function_instance(self) -> AcquisitionFunction | None:
        if self._acquisition_function_instance is None:
            print("An acquisition function has not been initialized yet.")
        return self._acquisition_function_instance

    @property
    def sampler_instance(self) -> MCSampler | None:
        if self._sampler_instance is None:
            print("A sampler has not been initialized yet.")
        return self._sampler_instance

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
    def kernel_factory(self) -> KernelFactory:
        return self._kernel_factory

    @kernel_factory.setter
    def kernel_factory(self, kernel_factory: KernelFactory):
        if not isinstance(kernel_factory, KernelFactory):
            raise ValueError("kernel_type must be of type KernelType")
        self._kernel_factory = kernel_factory

    """ ===================== """
    """ ===== Optimizer ===== """
    """ ===================== """

    def optimize(self, verbose=True):

        t0 = time.monotonic()

        # === 1. Compute metrics on current data ===
        self._compute_feasibility_mask(verbose=verbose)
        self._compute_acquisition_function_reference(verbose=verbose)  # best_f or ref_point + pareto
        self._compute_metrics(verbose=verbose)  # HV or best_value → appends to history

        # === 2. Initialize model ===
        self._initialize_kernel(verbose=verbose)
        self._initialize_model(verbose=verbose)
        self._fit_model(verbose=verbose)

        # === 3. Initialize acquisition function ===
        if self.acquisition_function_factory.requires_sampler():
            self._initialize_sampler(verbose=verbose)
        self._initialize_acquisition_function(verbose=verbose)

        # === 3. Optimize ===
        self._optimize_acquisition_function(verbose=verbose)

        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)

        if verbose:
            print(f"Optimization step completed in {t1 - t0:.2f}s")

    def _initialize_kernel(self, verbose=True):
        """ Initialize a kernel (or covariance module) instance using the kernel_factory.
        Note that, by building a fresh covariance module for each model and for each optimization
        iteration, kernels are freshly optimized at each iteration. """

        if verbose:
            print(f"Initializing kernel instance of type {self.kernel_factory.kernel_type.value}... ", end="")

        self._kernel_instance = self._kernel_factory() if self._kernel_factory else None

        if verbose:
            print("✓")

    def _initialize_model(self, verbose=True):
        """ Initialize Gaussian Process model(s) for the objectives and constraints.

        This method prepares the training dataset by combining the objective and constraint
        observations (and optionally their variances). Then it creates one independent
        SingleTaskGP model for each output dimension (each objective or constraint).
        The SingleTaskGP are finally combined into a ModelListGP to jointly represent the full
        multi-output model.

        Important: The GP model in BoTorch is a pure regression model: it simply fits the data
        it receives. It does not know or care about whether the model is for objectives to
        minimize or maximize. As such, the model must always and only receive the true, unnegated
        objective values as training data. In other words, the model is always fit to the true data.

        Note: by setting an input transform and an outcome transform, input (X) and output data (Y) are
        transformed and untransformed accordingly across the whole optimization pipeline, including
        the optimization of the acquisition function. For example, by setting an outcome transform to
        standardization, the Ys are standardized before the optimization and unstandardized right after.
        However, if a penalty is added in the forward method, this is not standardized properly and a
        pre-factor (or scaling) results in different penalty weights."""

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
                    covar_module=self._kernel_instance,
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-4)),
                )
            )

        self._model = ModelListGP(*models)
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        if verbose:
            print("✓")

    def _prepare_training_dataset(self):
        """ Prepare the training dataset for fitting the surrogate model.

        This method formats the inputs and outputs for model training:
            - Concatenates any available constraints to the outputs.
            - Concatenates variances for both objectives and constraints if available, else None."""

        train_x = self._X.clone()
        train_y = self._Y_obj.clone()

        # Concatenate constraints if available (not None)
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

    def _initialize_sampler(self, verbose=True):
        """Initialize sampler for Monte Carlo acquisition functions."""

        if verbose:
            print("Initializing sampler... ", end="")

        # Skip for analytical acquisition functions
        if self.acquisition_function_factory.requires_sampler():
            if self._sampler_type.name == SamplerType.Sobol.name:
                self._sampler_instance = SobolQMCNormalSampler(
                    torch.Size([self._num_mc_samples])
                )
            else:
                raise ValueError("Only Sobol Sampler is currently supported.")

        if verbose:
            print("✓")

    def _compute_acquisition_function_reference(self, verbose=True):
        """Compute reference values needed for acquisition function initialization.
        Single-objective: computes best value (_best_f)
        Multi-objective: computes reference point in high dimensional space."""

        if self._objective.num_objectives == 1:
            self._compute_best_Y(verbose=verbose)
        else:
            self._compute_reference_point(verbose=verbose)
            self._compute_pareto_front(verbose=verbose)

    def _compute_best_Y(self, verbose=True):
        """Compute the current best observed value (for single-objective).
        This method assumes maximization, therefore, the Y must be cast into a
        maximization problem before computing the best observation."""

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        if verbose:
            print("Computing best feasible... ", end="")

        feasible_X, feasible_Y = self.compute_feasible_XY(verbose=False)

        if feasible_X is not None:
            feasible_Y_max = feasible_Y.clone()
            feasible_Y_max[..., self._objective.obj_to_minimize] *= -1
            best_idx = feasible_Y_max.squeeze(-1).argmax()

            self._best_feasible_X = feasible_X[best_idx]
            self._best_feasible_Y = feasible_Y[best_idx]
            self._best_f = feasible_Y_max[best_idx]

            # best_value = feasible_Y[best_idx].squeeze().item()
            # self._best_values.append(best_value)

            if verbose:
                print(f"✓ {self._best_feasible_Y.item():.4f} in max. space.")

        else:
            self._best_feasible_Y = None
            self._best_feasible_X = None
            self._best_f = -float("inf")
            self._best_values.append(None)

            if verbose:
                print("✗ (no feasible points)")

    def _compute_reference_point(self, verbose=True):
        """
        Compute and set the reference point in the maximization space.
        Note that the reference point in the original space must be
        provided explicitly by the objective ("self._objective.ref_point").
        """

        if verbose:
            print("Defining reference point... ", end="")

        self._ref_point = self.objective.ref_point.clone().to(self._device, self._dtype)
        self._ref_point[..., self._objective.obj_to_minimize] *= -1

        if verbose:
            print(f"✓ {self._ref_point.detach().cpu().numpy()} in max. space.")

    def _compute_pareto_front(self, verbose=True):
        """
        Compute the Pareto front including constraints. Note that as
        "is_non_dominated" assumes maximization, the Y must be cast into a
        maximization problem before computing the pareto front.
        """

        if verbose:
            print("Finding Pareto front... ", end="")

        if self._objective.num_objectives == 1:
            raise ValueError("Pareto front cannot be computed for single-objective problems.")

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        Y_obj_maximized = self._Y_obj.clone()
        Y_obj_maximized[..., self._objective.obj_to_minimize] *= -1

        # Compute feasible Pareto front
        if self._feasible_mask.any():
            feasible_pareto_mask = torch.zeros_like(self._feasible_mask)
            feasible_pareto_mask[self._feasible_mask] = is_non_dominated(
                Y_obj_maximized[self._feasible_mask]
            )

            self._feasible_pareto_front_X = self._X[feasible_pareto_mask]
            self._feasible_pareto_front_Y = self._Y_obj[feasible_pareto_mask]

        else:

            self._feasible_pareto_front_X = None
            self._feasible_pareto_front_Y = None

        if verbose:
            print(f"✓")

    def _initialize_acquisition_function(self, verbose=True):
        """ Initialize an acquisition function instance using the acquisition_function_factory. """

        if verbose:
            print(
                f"Initializing acquisition function of type {self.acquisition_function_factory.acquisition_function_type.value}... ",
                end="")

        params = AcquisitionRuntimeParams(
            model=self._model,
            maximize=not self._objective.obj_to_minimize[0],
            best_f=self._best_f,
            X_baseline=self._X,
            sampler=self._sampler_instance,
            objective=self._objective,
            ref_point=self._ref_point,
            partitioning=self._partitioning,
            constraints=self._objective.output_constraints if hasattr(self._objective, 'output_constraints') else None,
        )
        self._acquisition_function_instance = self.acquisition_function_factory(params)

        if verbose:
            print("✓")

    def _initialize_partitioning(self):
        # Compute posterior mean of objectives
        prediction = self._model.posterior(self._X).mean

        # Filter feasible points if constraints exist
        if self._objective.output_constraints is None:
            feasible_Y = prediction
        else:
            feasible_maks = torch.stack([c(prediction) <= 0 for c in self._objective.output_constraints]).all(dim=0)
            feasible_Y = prediction[feasible_maks]

        # Cast feasible points in maximization space
        Y = feasible_Y.clone()
        Y[..., self.objective.obj_to_minimize] *= -1

        # Build partitioning with feasible objectives only
        self._partitioning = NondominatedPartitioning(
            ref_point=self._ref_point,
            Y=Y,
        )

    def _fit_model(self, restart_on_error=True, verbose=True):
        if not isinstance(self._model, ModelListGP):
            raise ValueError("Model must be initialized before fitting.")

        if verbose:
            print("Fitting model... ", end="", flush=True)

        restart_count = 0
        while restart_count <= self._n_model_fit_restarts:
            try:
                botorch.fit_gpytorch_mll(self._mll)

                if self._mll.training:
                    raise RuntimeError("Model fitting failed (still in training mode)")

                # Validate each model in the list
                for i, model in enumerate(self._model.models):
                    self._validate_model_fit(model, i, verbose=verbose)

                if verbose:
                    print("Fitting model... ✓")
                break

            except Exception as e:
                restart_count += 1

                if restart_on_error and restart_count <= self._n_model_fit_restarts:
                    if verbose:
                        print("Fitting model... ✗")
                        print(f"  Error: {e}")
                        print(f"  Reinitializing and retrying... "
                              f"(Attempt {restart_count}/{self._n_model_fit_restarts})")

                    self._initialize_kernel(verbose=False)
                    self._initialize_model(verbose=False)
                    self._randomize_hyperparameters()

                else:
                    raise RuntimeError(
                        f"Model fitting failed after {self._n_model_fit_restarts} attempts. "
                        f"Last error: {e}"
                    )
        return None

    def _randomize_hyperparameters_bak(self):
        """Randomize hyperparameters for a fresh optimization start."""
        for m in self._model.models:
            # Sample in log space for better coverage
            # Lengthscale: log-uniform in [0.1, 10] relative to default
            log_ls = torch.empty_like(m.covar_module.base_kernel.lengthscale).uniform_(-1, 1)
            m.covar_module.base_kernel.lengthscale = 10 ** log_ls

            # Outputscale: log-uniform in [0.1, 10]
            log_os = torch.empty(1, device=self._device).uniform_(-1, 1)
            m.covar_module.outputscale = 10 ** log_os

            # Noise: log-uniform in [1e-4, 1e-1]
            log_noise = torch.empty(1, device=self._device).uniform_(-4, -1)
            m.likelihood.noise = 10 ** log_noise

    # TODO
    def _randomize_hyperparameters(self):
        """Randomize hyperparameters with log-uniform distribution.

        Workflow should be:

        raw = module.raw_outputscale
        c = module.constraint_for_parameter_name("raw_outputscale")

        # sample in transformed space (positive)
        target = torch.empty_like(module.outputscale).uniform_(0.1, 10.0)

        # write into raw space
        raw.data = c.inverse_transform(target)


        """
        for model in self._model.models:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    with torch.no_grad():
                        old_val = param.clone()
                        log_val = torch.empty_like(param).uniform_(-1, 1)
                        param.copy_(10 ** log_val)
                        print(f"  Randomized {name}: {old_val.item():.4e} → {param.item():.4e}")

    def _validate_model_fit(self, model, model_idx, verbose=False):
        """ Validate hyperparameters of a fitted GP model. Note: this validation method
        operates on a single GP model. Returns list of issues (empty if fit looks healthy). """
        noise = self._extract_model_noise(model, verbose=verbose)
        self._validate_noise(noise)
        transformed_params = self._extract_transformed_params(covar_module=model.covar_module, verbose=verbose)
        self._validate_transformed_params(params=transformed_params, noise=noise)

    @staticmethod
    def _extract_transformed_params(covar_module, verbose=False):

        params = {}
        # for name, param in covar_module.named_parameters():
        #     # these are ALWAYS raw parameters
        #     if not param.requires_grad:
        #         continue
        #     params[name] = param

        for name, raw_param in covar_module.named_parameters():
            constraint = covar_module.constraint_for_parameter_name(name)
            if constraint is not None:
                value = constraint.transform(raw_param)
                params[name.replace("raw_", "")] = value
                if verbose:
                    print(f"{name.replace("raw_", "")}: {value.item():.2e}")

        return params

    @staticmethod
    def _print_kernel_params(params):
        print(f"\n{'Parameter':<20} {'Module':<20} {'Value'}")
        print("-" * 55)
        for p in params:
            val = p["value"]
            val_str = f"{val:.4e}" if isinstance(val, float) else str(val)
            print(f"  {p['name']:<18} {p['module_name']:<20} {val_str}")

    @staticmethod
    def _param_to_value(param):
        tensor = param.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.view(-1).tolist()

    @staticmethod
    def _extract_model_noise(model, verbose=False):
        noise = model.likelihood.noise
        if verbose:
            print(f"\nnoise: {noise.item():.2e}")
        return noise

    @staticmethod
    def _group_params_by_type(kernel_params: dict):
        """ Group hyperparameters by type for targeted validation.
        Returns dict with keys:
        - variance: These parameters control the output magnitude of the kernel — how much the GP function
        can deviate from the mean. For example, the outputscale of the ScaleKernel is the variance of the
        GP, since k(x, x) = sigma_f^2 = Var(f(x)).
        - lengthscale:
        - period
        - mixture
        - other
        """
        groups = {
            "variance": {},  # outputscale, variance, constant
            "lengthscale": {},
            "period": {},
            "mixture": {},  # spectral mixture params
            "other": {},
        }

        for key, val in kernel_params.items():
            k = key.lower()
            if "lengthscale" in k:
                groups["lengthscale"][key] = val
            elif "outputscale" in k or "variance" in k:
                groups["variance"][key] = val
            elif "period" in k:
                groups["period"][key] = val
            elif "mixture" in k:
                groups["mixture"][key] = val
            else:
                groups["other"][key] = val

        return groups

    @staticmethod
    def _validate_noise(noise: torch.Tensor):
        if not torch.isfinite(noise).all():
            raise Warning(f"Noise is not finite: {noise}")

    def _validate_transformed_params(self, params: dict, noise: torch.Tensor):
        """Hard check: ensure hyperparameters and noise are finite."""
        # Validate NaN/Infinite
        self._validate_transformed_params_finiteness(params)

        # Validate parameters by group
        groups = self._group_params_by_type(params)
        self._validate_transformed_params_variance(groups["variance"], noise)
        self._validate_transformed_params_lengthscale(groups["lengthscale"])
        self._validate_transformed_params_period(groups["period"])
        self._validate_transformed_params_mixture(groups["mixture"])

    @staticmethod
    def _validate_transformed_params_finiteness(params: dict):
        for key, val in params.items():
            if not torch.isfinite(val).all():
                raise Warning(f"Non-finite hyperparameter {key} detected: {val}.")
        return

    @staticmethod
    def _validate_transformed_params_variance(params: dict, noise):
        """ Validate variance/outputscale parameters. Specifically, checks whether the
        model outputscale is not smaller than the noise.
        Core principle: signal variance should exceed noise variance,
        otherwise the GP is noise-dominated and predictions collapse to mean.
        With standardized Y (std=1), outputscale ≈ 1.0 is expected. """
        for key, val in params.items():
            if val < noise:
                raise Warning(f"{key} ({val.item():.2e}) < noise ({noise.item():.2e})")
        return warnings

    @staticmethod
    def _validate_transformed_params_lengthscale(params: dict):
        """ Validate lenghtscale/domain range. Specifically, checks whether the
         lengthscale is larger than 100x the corresponding domain size. This method
         assumes that the input domain is normalized between 0 and 1. """
        for key, val in params.items():
            if val > 100:
                warnings.append(f"{key} ({val.item():.2e}) > 100 → no correlation")
            elif val > 10:
                warnings.append(f"{key} ({val.item():.2e}) > 10 → weak correlation")
            elif val < 0.01:
                warnings.append(f"{key} ({val.item():.2e}) < 0.01 → possible overfitting")
            elif val < 0.0001:
                warnings.append(f"{key} ({val.item():.2e}) < 0.0001 → overfitting")

        return warnings

    @staticmethod
    def _validate_transformed_params_period(params: dict):
        """ Validate period parameters. With standardized inputs in [0, 1]:
        - period = 1.0 means one full cycle in domain
        - Too small → high frequency, overfitting
        - Too large → effectively non-periodic. """
        for key, val in params.items():
            if val > 10:
                warnings.append(f"{key} ({val.item():.2e}) > 10 → effectively non-periodic")
            elif val > 5:
                warnings.append(f"{key} ({val.item():.2e}) > 5 → weak periodicity")
            elif val < 0.05:
                warnings.append(f"{key} ({val.item():.2e}) < 0.05 → likely overfitting")
            elif val < 0.1:
                warnings.append(f"{key} ({val.item():.2e}) < 0.1 → possible overfitting")

        return warnings

    @staticmethod
    def _validate_transformed_params_mixture(params):
        """
        Validate spectral mixture parameters.

        Checks:
        - Degenerate weights (all near zero)
        - Single component dominance (wasted complexity)
        """
        # Check on the mixture in not implemented yet.
        return

    def _optimize_acquisition_function(self, verbose=True):

        if verbose:
            print(f"Optimizing acquisition function... ", end="")

        if self._acquisition_function_type == AcquisitionFunctionType.qNParEGO:
            self._new_X, _ = optimize_acqf_list(
                acq_function_list=self._acquisition_function_list,
                bounds=self._objective.bounds,
                num_restarts=self._n_acqf_opt_restarts,
                raw_samples=self._num_raw_samples,
                options={"batch_limit": 5, "maxiter": self._n_acqf_opt_max_iter},
            )
        else:
            # If nonlinear inequality input constraints are provided, use a custom initial condition
            # generator that selects "num_restarts" points. These points are distributed according to
            # "fraction_of_previous_X" between the current pareto front and randomly generated points.
            self._new_X, _ = optimize_acqf(
                acq_function=self._acquisition_function_instance,
                bounds=self._objective.bounds,
                q=self._batch_size,
                num_restarts=self._n_acqf_opt_restarts,
                raw_samples=self._num_raw_samples,
                options={"maxiter": self._n_acqf_opt_max_iter, "disp": False},
                sequential=True,
                equality_constraints=self._objective.linear_equality_input_constraints,
                inequality_constraints=self._objective.linear_inequality_input_constraints,
                nonlinear_inequality_constraints=self._objective.nonlinear_inequality_input_constraints,
                ic_generator=self._ic_generator
                if self._objective.nonlinear_inequality_input_constraints is not None
                else None,
                **{
                    "fraction_of_previous_X": 0.8,
                    "noise_scale": 0,
                } if self.objective.nonlinear_inequality_input_constraints is not None
                else {}
            )

        if verbose:
            print(f"✓ (New X: {self._new_X.detach().cpu().numpy()}")

    def is_converged(self, patience=10, tol=1e-3, verbose=True):

        if verbose:
            print("Checking convergence... ", end="")

        # Select the relevant metric
        if self._objective.num_objectives == 1:
            metrics_list = self._best_values
        else:
            metrics_list = self._hypervolume

        # Need enough history to evaluate
        if not metrics_list or len(metrics_list) < patience:
            if verbose:
                print("✗")
            return False

        # Take the last `patience` values
        metrics = metrics_list[-patience:]
        # Compute improvements between consecutive steps
        improvements = [metrics[i + 1] - metrics[i] for i in range(len(metrics) - 1)]
        # Converged if all improvements are smaller than tolerance
        if verbose:
            print("✓")
        return all(abs(impr) < tol for impr in improvements)

    def _ic_generator(
            self,
            acq_function,  # noqa
            bounds: torch.Tensor,
            q: int,  # noqa
            num_restarts: int,
            raw_samples: int,
            fixed_features: dict[int, float] | None = None,  # noqa
            inequality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None,  # noqa
            equality_constraints: list[tuple[torch.Tensor, torch.Tensor, float]] | None = None,  # noqa
            **kwargs: dict
    ) -> torch.Tensor:
        """
        Generates initial conditions for constrained acquisition optimization.
        Mixes previous points and new samples from a constraint-aware sampler.
        """

        frac_prev = kwargs.get("fraction_of_previous_X", 0.5)
        noise_scale = kwargs.get("noise_scale", 0.0)

        # Instantiate a constraint-aware sampler
        sampler = Sampler(
            device=self._device,
            dtype=self.dtype,
            sampler_type=SamplerType.Sobol,
            bounds=bounds,
            n_dimensions=self.objective.dim,
            normalize=False,
            linear_equality_constraints=self.objective.linear_equality_input_constraints,
            linear_inequality_constraints=self.objective.linear_inequality_input_constraints,
            nonlinear_inequality_constraints=self.objective.nonlinear_inequality_input_constraints,
        )

        # Determine number of points from previous Pareto front
        if self._X is not None and self._Y_obj is not None:
            Y_obj_for_pareto = self._Y_obj.clone()
            Y_obj_for_pareto[..., self._objective.obj_to_minimize] *= -1
            pareto_mask = is_non_dominated(Y_obj_for_pareto)
            X_pareto = self._X[pareto_mask]
            n_pareto = int(frac_prev * num_restarts)
            if X_pareto.shape[0] > 0:
                # Randomly select n_pareto points
                indices = torch.randperm(X_pareto.shape[0])[:n_pareto]
                prev_points = X_pareto[indices]
            else:
                prev_points = torch.empty(0, bounds.shape[1], device=self._device)
                n_pareto = 0
        else:
            prev_points = torch.empty(0, bounds.shape[1], device=self._device)
            n_pareto = 0

        # Determine number of previous and new points
        n_new = num_restarts - n_pareto

        # Draw raw samples for new points
        raw_X = sampler.draw_samples(n=raw_samples)
        if n_new > raw_X.shape[0]:
            raise RuntimeError(
                f"Requested {n_new} new points, but only {raw_X.shape[0]} feasible raw samples available.")
        idx = torch.randperm(raw_X.shape[0])[:n_new]
        new_points = raw_X[idx]

        # Add small noise for exploration
        if noise_scale > 0:
            new_points += noise_scale * torch.randn_like(new_points)

        # Combine previous points and new points
        combined = torch.cat([prev_points, new_points], dim=0)

        # Ensure proper shape for BoTorch: (num_restarts, q, d)
        return combined.unsqueeze(1)  # q=1 for sequential optimization

    def compute_acquisition_function_value_at_X(self, X: torch.Tensor, verbose=True):
        if not isinstance(X, torch.Tensor):
            raise ValueError("X must be of type torch.Tensor.")
        acq_val = self._acquisition_function_instance(X)
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

    def _compute_feasibility_mask(self, verbose=True):
        """ Compute feasibility mask on the original, non maximized, Y.
        If the objective is unconstrained, all observations are feasible.
        Otherwise, concatenate objectives and constraints along the last
        dimension, then compute the feasibility mask: a point is feasible
        only if all constraints are ≤ 0.
        Handles arbitrary batch shapes: (..., n_points, n_outputs) -> (..., n_points) """

        if verbose:
            print("Computing feasibility mask... ", end="")

        n_points = self._Y_obj.shape[0]

        if self._objective.output_constraints is None:
            # No constraints — all points feasible
            self._feasible_mask = torch.ones(n_points, dtype=torch.bool, device=self._device)
        else:
            # Concatenate objectives and constraints, check all constraints ≤ 0
            Y_full = torch.cat([self._Y_obj, self._Y_con], dim=-1)
            self._feasible_mask = torch.stack(
                [c(Y_full) <= 0 for c in self._objective.output_constraints]
            ).all(dim=0).squeeze()  # (n_constraints, n_points) -> (n_points,)

        if verbose:
            n_feasible = self._feasible_mask.sum().item()
            print(f"✓ ({n_feasible}/{n_points} feasible)")

    def compute_feasible_XY(self, verbose=False):
        """ Computes feasible X and Y. This method assumes maximization,
        therefore, the Y must be cast into a maximization problem. """

        if verbose:
            print("Computing feasible X and Y... ", end="")

        if self._objective.num_objectives != 1:
            raise ValueError("Only single objective is currently supported.")

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        if self._feasible_mask.any():
            feasible_X = self._X[self._feasible_mask].clone()
            feasible_Y = self._Y_obj[self._feasible_mask].clone()
        else:
            feasible_X = None
            feasible_Y = None

        if verbose:
            print("✓")

        return feasible_X, feasible_Y

    def compute_infeasible_XY(self, verbose=False):
        """ Computes infeasibile X and Y. This method assumes maximization,
        therefore, the Y must be cast into a maximization problem. """

        if verbose:
            print("Computing infeasible X and Y... ", end="")

        if self._objective.num_objectives != 1:
            raise ValueError("Only single objective is currently supported.")

        if self._feasible_mask is None:
            raise ValueError(
                "Feasibility mask is missing. "
                "Call compute_feasibility_mask() first."
            )

        infeasible_mask = torch.logical_not(self._feasible_mask)
        if infeasible_mask.any():
            infeasible_X = self._X[infeasible_mask].clone()
            infeasible_Y = self._Y_obj[infeasible_mask].clone()
        else:
            infeasible_X = None
            infeasible_Y = None

        if verbose:
            print("✓")

        return infeasible_X, infeasible_Y

    def _compute_metrics(self, verbose=True):
        """Compute and store the appropriate metric based on problem type."""
        if self._objective.num_objectives == 1:
            self._compute_best_value(verbose=verbose)
        else:
            self._compute_hypervolume(verbose=verbose)

    def _compute_hypervolume(self, verbose=True):
        """ Compute the hypervolume metric. It assumes maximization. """

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
        feasible_pareto_front_Y_maximized[..., self._objective.obj_to_minimize] *= -1
        hv = Hypervolume(self._ref_point).compute(feasible_pareto_front_Y_maximized)
        self._hypervolume.append(hv)

        if verbose:
            print("✓")
            print(f"Hypervolume = {self._hypervolume[-1]:>4.2f}")

    def _compute_best_value(self, verbose=True):
        """ Track the best observed value as metric for single objective problems."""

        if verbose:
            print("Computing best value... ", end="")

        if self._best_feasible_Y is not None:
            best_value = self._best_feasible_Y.squeeze().item()  # Original space
        else:
            best_value = float('inf') if self._objective.obj_to_minimize[0] else float('-inf')

        self._best_values.append(best_value)

        if verbose:
            print(f"✓ (best = {self._best_feasible_Y.item():.4f})")

    def update_XY(self, new_X: torch.Tensor, new_Y_obj: torch.Tensor, new_Y_track: torch.Tensor | None = None,
                  new_Y_obj_var: torch.Tensor | None = None,
                  new_Y_con: torch.Tensor | None = None, new_Y_con_var=None,
                  new_Y_track_var: torch.Tensor | None = None) -> None:
        self.update_X(new_X)
        self.update_Y_obj(new_Y_obj, new_Y_obj_var)
        self.update_Y_con(new_Y_con, new_Y_con_var)
        self.update_Y_track(new_Y_track, new_Y_track_var)

    def update_X(self, new_X: torch.Tensor):
        if new_X is not None:
            new_X = new_X.to(self._device, self._dtype)
            self._X = torch.cat([self._X, new_X], dim=0)

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

    def update_Y_track(self, new_Y_track: torch.Tensor, new_Y_track_var: torch.Tensor or None = None):
        if new_Y_track is not None:
            new_Y_track = new_Y_track.to(self._device, self._dtype)
            self._Y_track = torch.cat([self._Y_track, new_Y_track], dim=0)
        if new_Y_track_var is not None:
            new_Y_track_var = new_Y_track_var.to(self._device, self._dtype)
            self._Y_track_var = torch.cat([self._Y_track_var, new_Y_track_var], dim=0)

    """ =============== """
    """ ===== I/O ===== """
    """ =============== """

    def to_file(self, output_path: Path or str = None):
        if output_path is None:
            output_path = Path.cwd() / "bayesian_optimizer.dat"
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as file:
            pickle.dump(self, file)  # type: ignore
        return output_path

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

    """ ===================== """
    """ ===== DEBUGGERS ===== """
    """ ===================== """

    def print_hyperparameters(model):
        print(f"Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.4f}")
        print(f"Outputscale: {model.covar_module.outputscale.item():.4f}")
        print(f"Noise: {model.likelihood.noise.item():.6f}")
