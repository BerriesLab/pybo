import json
import pickle
from enum import Enum
from datetime import datetime

import warnings
import time
import glob
import os
from pathlib import Path

import botorch
import gpytorch
import numpy as np

from botorch.exceptions import (
    BadInitialCandidatesWarning,
    InputDataWarning,
    OptimizationWarning,
)
from botorch.exceptions.warnings import NumericsWarning
from botorch.models import ModelList
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective import is_non_dominated, Hypervolume, get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition import (
    qNoisyExpectedImprovement, GenericMCObjective
)
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.utils.transforms import normalize
from botorch.utils.sampling import sample_simplex

from gpytorch.constraints import GreaterThan
from torch import Tensor

from acquisition_functions.qNEHVI import qExplorationWeightedNEHVI, qDiversityWeightedNEHVI

from objectives.base_class import MCMultiOutputBase
from utils import AcquisitionFunctionType, SamplerType
from utils.cuda import get_device, get_supported_dtype
from utils.validators import *

from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood


class Mobo:
    """
    A wrapper around BoTorch for (Multi) Objective Bayesian Optimization.
    Similarly to BoTorch, this class is designed to work with maximization
    problems only.
    """

    def __init__(
            self,
            experiment_name: str,
            device: torch.device,
            dtype: torch.device.type,
            objective: MCMultiOutputBase,
            X: torch.Tensor | None = None,
            Yobj: torch.Tensor | None = None,
            Yobj_var: torch.Tensor | None = None,
            Ycon: torch.Tensor | None = None,
            Ycon_var: torch.Tensor | None = None,
            acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.qNEHVI,
            sampler_type: SamplerType = SamplerType.Sobol,
            batch_size: int = 1,
            mc_samples: int = 256,
            raw_samples: int = 512,
            n_acqf_opt_iter: int = 250,
            n_acqf_opt_restarts: int = 1,
            n_model_fit_restarts: int = 10,
    ):

        # === Device Attributes ===
        self.device = device
        self.dtype = dtype

        # === Experiment Attributes ===
        self.experiment_name = experiment_name
        self._datetime = datetime.datetime.now()
        self.objective = objective
        self.X: torch.Tensor = X
        self.Yobj: torch.Tensor = Yobj
        self.Yobj_var: torch.Tensor = Yobj_var
        self.Ycon: torch.Tensor = Ycon
        self.Ycon_var: torch.Tensor = Ycon_var

        # === Optimization attributes ===
        self.acquisition_function_type = acquisition_function_type
        self.sampler_type = sampler_type
        self.n_acqf_opt_iter = n_acqf_opt_iter  # Number of iterations for acquisition function optimization
        self.n_acqf_opt_restarts = n_acqf_opt_restarts  # Number of acquisition function optimization restarts
        self.n_model_fit_restarts = n_model_fit_restarts  # Max number of model fit attempts.
        self.batch_size = batch_size  # Number of candidates to be generated in parallel in each optimization step
        self.num_mc_samples = mc_samples  # Number of samples drawn from the predictive posterior distribution to estimate the acquisition function
        self.num_raw_samples = raw_samples  # Number of random points sampled in the search space to initialize the optimizer that maximizes the acquisition function

        # === Metrics ===
        self.hypervolume: list[float] = []
        self.elapsed_time: list[float] = []

    # === Pickling helper ===
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

    # === Experiment Attributes ===
    @property
    def experiment_name(self):
        return self._experiment_name

    @experiment_name.setter
    def experiment_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        self._experiment_name = name

    @property
    def datetime(self):
        return self._datetime

    @property
    def objective(self) -> Union[MCMultiOutputBase,]:
        return self._objective

    @objective.setter
    def objective(self, objective: Union[MCMultiOutputBase]):
        if not isinstance(objective, Union[MCMultiOutputBase]):
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

    @property
    def Yobj(self) -> torch.Tensor | None:
        return self._Yobj

    @Yobj.setter
    def Yobj(self, Yobj: torch.Tensor):
        if not isinstance(Yobj, Union[torch.Tensor, None]):
            raise ValueError("Yobj must be of type torch.Tensor or None.")
        if Yobj is not None and Yobj.shape[-1] != self.objective.num_objectives:
            raise ValueError("Yobj must have the same number of dimensions as objective.")
        self._Yobj = Yobj.to(self._device, self._dtype) if Yobj is not None else None

    @property
    def Yobj_var(self) -> torch.Tensor | None:
        return self._Yobj_var

    @Yobj_var.setter
    def Yobj_var(self, Yobj_var: torch.Tensor | None = None):
        if not isinstance(Yobj_var, Union[torch.Tensor, None]):
            raise ValueError("Yobj_var must be of type torch.Tensor or None.")
        if Yobj_var is not None and Yobj_var.shape[-1] != self.objective.num_objectives:
            raise ValueError("Yobj_var must have the same number of dimensions as objective.")
        self._Yobj_var = Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None

    @property
    def Ycon(self) -> torch.Tensor | None:
        return self._Ycon

    @Ycon.setter
    def Ycon(self, Ycon: torch.Tensor | None):
        if not isinstance(Ycon, Union[torch.Tensor, None]):
            raise ValueError("Ycon must be of type torch.Tensor or None.")
        if Ycon is not None and Ycon.shape[-1] != self.objective.num_constraints:
            raise ValueError("Ycon must have the same number of constraints as objective.")
        self._Ycon = Ycon.to(self._device, self._dtype) if Ycon is not None else None

    @property
    def Ycon_var(self) -> torch.Tensor | None:
        return self._Ycon_var

    @Ycon_var.setter
    def Ycon_var(self, Ycon_var: torch.Tensor | None = None):
        if not isinstance(Ycon_var, Union[torch.Tensor, None]):
            raise ValueError("Ycon_var must be of type torch.Tensor or None.")
        if Ycon_var is not None and Ycon_var.shape[-1] != self.objective.num_constraints:
            raise ValueError("Ycon_var must have the same number of constraints as objective.")
        self._Ycon_var = Ycon_var.to(self._device, self._dtype) if Ycon_var is not None else None

    @property
    def acquisition_function_type(self) -> AcquisitionFunctionType:
        return self._acquisition_function_type

    @acquisition_function_type.setter
    def acquisition_function_type(self, af_type):
        if not isinstance(af_type, AcquisitionFunctionType):
            raise ValueError("Acquisition function type must be of type AcquisitionFunctionType")
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
        return self._n_acqf_opt_iter

    @n_acqf_opt_iter.setter
    def n_acqf_opt_iter(self, n_acqf_opt_iter):
        if not isinstance(n_acqf_opt_iter, int):
            raise ValueError("n_acqf_opt_iter must be of type int")
        self._n_acqf_opt_iter = n_acqf_opt_iter

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
        validate_mc_samples(mc_samples)
        self._num_mc_samples = mc_samples

    @property
    def num_raw_samples(self):
        return self._num_raw_samples

    @num_raw_samples.setter
    def num_raw_samples(self, raw_samples: int):
        validate_raw_samples(raw_samples)
        self._num_raw_samples = raw_samples

    # === STATE properties ===
    @property
    def model(self):
        return self._model

    @property
    def mll(self):
        return self._mll

    @property
    def pareto_front(self):
        return self._pareto_front

    @property
    def new_X(self) -> torch.Tensor:
        if getattr(self, "_new_X", None) is None:
            print("A new_X has not been computed yet.")
            return None
        return self._new_X
    # @property
    # def pareto_front_mask(self):
    #     return self._pareto_front_mask
    #
    # @property
    # def feasible_observations_mask(self):
    #     return self._feasible_observations_mask
    #

    @property
    def hypervolume(self):
        return self._hypervolume

    @hypervolume.setter
    def hypervolume(self, hv: list[float]):
        self._hypervolume = hv

    @property
    def elapsed_time(self):
        return self._elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, t: list[float]):
        self._elapsed_time = t

    """ Optimizer """

    def initialize_model(self, verbose=True):
        """ Initialize Gaussian Process model(s) for the objectives and constraints.

        This method prepares the training dataset by combining the objective and constraint
        observations (and optionally their variances). Then it creates one independent
        SingleTaskGP model for each output dimension (each objective or constraint).
        These models are combined into a ModelListGP to jointly represent the full
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
        # for each objective (or observable)
        train_x, train_y, train_y_var = self.prepare_training_dataset()
        models = []
        for i in range(0, train_y.shape[-1]):
            models.append(
                SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y[..., i: i + 1],
                    train_Yvar=(train_y_var[..., i: i + 1] if train_y_var is not None else None),
                    input_transform=Normalize(d=self.objective.dim, bounds=self.objective.bounds),
                    outcome_transform=Standardize(m=1),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6)),
                )
            )
        self.__setattr__("_model", ModelListGP(*models))
        self.__setattr__("_mll", SumMarginalLogLikelihood(self._model.likelihood, self._model))

        # self._model = ModelListGP(*models)
        # self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        if verbose:
            print("✓")

    def prepare_training_dataset(self):
        """ Prepare the training dataset for fitting the surrogate model.

        This method formats the inputs and outputs for model training:
            - Concatenates any available constraints to the outputs.
            - Concatenates variances for both objectives and constraints if available, else None."""

        train_x = self._X.clone()
        train_y = self._Yobj.clone()

        # Concatenate constraints if available (not None)
        if self._Ycon is not None:
            train_y_con = self._Ycon.clone()
            train_y = torch.cat((train_y, train_y_con), dim=-1)

        # Define train_y_var
        if self._Yobj_var is not None and self._Ycon_var is not None:
            train_y_var = self._Yobj_var.clone()
            train_y_con_var = self._Ycon_var.clone()
            train_var = torch.cat((train_y_var, train_y_con_var), dim=-1)
        else:
            train_var = None

        return train_x, train_y, train_var

    def initialize_sampler(self, verbose=True):
        if verbose:
            print("Initializing sampler... ", end="")

        if self._sampler_type.name == SamplerType.Sobol.name:
            self._sampler_instance = SobolQMCNormalSampler(
                torch.Size([self._num_mc_samples])
            )
        else:
            raise ValueError("Only Sobol Sampler is currently supported.")

        if verbose:
            print("✓")

    def compute_reference_point(self, verbose=True):
        """
        Compute and set the reference point in the maximization space.

        The reference point is a key component in multi-objective optimization,
        representing a point in objective space that is dominated by all observed
        solutions. It is used to compute the hypervolume improvement metric.

        This method sets the reference point only if the reference point is None
        (typically at the first iteration). Note that the reference point must provide
        explicitly by the objective ("self._objective.ref_point").
        """

        # The reference point is calculated only for the first iteration step
        if hasattr(self, "_ref_point"):
            return

        if verbose:
            print("Defining reference point... ", end="")

        ref_point = self.objective.ref_point.clone().to(self._device, self._dtype)
        self.__setattr__("_ref_point", ref_point)
        self._ref_point[..., self._objective.obj_to_minimize] *= -1

        if verbose:
            print("✓")
            print(f"Reference point (in maximization space): {self._ref_point.detach().cpu().numpy()}")

    def initialize_acquisition_function(self, verbose=True):
        if verbose:
            print("Initializing acquisition function... ", end="")

        if self._acquisition_function_type == AcquisitionFunctionType.qEHVI:
            self.initialize_partitioning()
            self._acquisition_function_instance = qExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._objective.output_constraints,
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogEHVI:
            self.initialize_partitioning()
            self._acquisition_function_instance = qLogExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._objective.output_constraints,
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qNEHVI:
            self._acquisition_function_instance = qNoisyExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                X_baseline=self._X,
                sampler=self._sampler_instance,
                prune_baseline=True,
                objective=self._objective,
                constraints=self._objective.output_constraints,
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogNEHVI:
            self._acquisition_function_instance = (
                qLogNoisyExpectedHypervolumeImprovement(
                    model=self._model,
                    ref_point=self._ref_point,
                    X_baseline=self._X,
                    prune_baseline=True,
                    sampler=self._sampler_instance,
                    objective=self._objective,
                    constraints=self._objective.output_constraints,
                )
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qEWNEHVI:
            self._acquisition_function_instance = (
                qExplorationWeightedNEHVI(
                    model=self._model,
                    ref_point=self._ref_point,
                    X_baseline=self._X,
                    prune_baseline=True,
                    sampler=self._sampler_instance,
                    objective=self._objective,
                    constraints=self._objective.output_constraints,
                    exploration_weight=1.0,
                )
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qDWNEHVI:
            self._acquisition_function_instance = (
                qDiversityWeightedNEHVI(
                    model=self._model,
                    ref_point=self._ref_point,
                    X_baseline=self._X,
                    prune_baseline=True,
                    sampler=self._sampler_instance,
                    objective=self._objective,
                    constraints=self._objective.output_constraints,
                    min_dist_radius=1.0,
                    distance_penalty_weight=1.0,
                )
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qNParEGO:
            with torch.no_grad():
                pred = self._model.posterior(self._X).mean
            self.__setattr__(__name="_acquisition_function_list", __value=[])
            for _ in range(self._batch_size):
                weights = sample_simplex(self.objective.num_objectives, device=self._device, dtype=self._dtype).squeeze()
                objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
                acq_func = qNoisyExpectedImprovement(
                    model=self._model,
                    objective=objective,
                    X_baseline=normalize(self._X, self.objective.bounds),
                    sampler=self._sampler_instance,
                    prune_baseline=True,
                )
                self._acquisition_function_list.append(acq_func)

        else:
            raise ValueError(
                f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}."
            )

        if verbose:
            print("✓")

    def initialize_partitioning(self):
        with torch.no_grad():
            warnings.warn("Partitioning is not taking into account output constraints, if any", OptimizationWarning)
            # TODO: when using qEHVI, use non_dominated_partitioning and pass only feasible Ys
            prediction = self._model.posterior(normalize(self._X, self._objective.bounds)).mean
            partitioning = NondominatedPartitioning(ref_point=self._ref_point, Y=prediction)
            self.__setattr__(__name="_partitioning", __value=partitioning)

    def fit_model(self, restart_on_error=True, verbose=True):
        if not isinstance(self._model, ModelListGP):
            raise ValueError("Model must be initialized before fitting.")

        if verbose:
            print("Fitting model... ", end="")

        restart_count = 0
        while True:
            try:
                botorch.fit_gpytorch_mll(self._mll)
                if verbose:
                    print("✓")
                break  # Exit the inner loop on success

            except Exception as e:
                if restart_on_error and restart_count < self._n_model_fit_restarts:
                    print("x")
                    print(
                        f"Restarting fitting... (Attempt {restart_count + 1}/{self._n_model_fit_restarts})"
                    )
                    restart_count += 1
                else:
                    raise e  # Raise if not restarting or max restarts reached
        return None

    def optimize_acquisition_function(self, verbose=True):
        if verbose:
            print(f"Optimizing acquisition function... ", end="")

        if self._acquisition_function_type == AcquisitionFunctionType.qNParEGO:
            candidates, _ = optimize_acqf_list(
                acq_function_list=self._acquisition_function_list,
                bounds=self._objective.bounds,
                num_restarts=self._n_acqf_opt_restarts,
                raw_samples=self._num_raw_samples,
                options={"batch_limit": 5, "maxiter": self._n_acqf_opt_iter},
            )
        else:
            candidates, _ = optimize_acqf(
                acq_function=self._acquisition_function_instance,
                bounds=self._objective.bounds,
                q=self._batch_size,
                num_restarts=self._n_acqf_opt_restarts,
                raw_samples=self._num_raw_samples,
                options={"maxiter": self._n_acqf_opt_iter, "disp": True},
                sequential=True,
                equality_constraints=self._objective.linear_equality_input_constraints,
                inequality_constraints=self._objective.linear_inequality_input_constraints,
                nonlinear_inequality_constraints=self._objective.nonlinear_inequality_input_constraints,
            )
        self.__setattr__("_new_X", candidates)

        if verbose:
            print("✓")

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

    def compute_pareto_front(self, verbose=True):
        """
        Compute the Pareto front including constraints. It uses
        "is_non_dominated", which assumes maximization.
        """

        if verbose:
            print("Finding Pareto front... ", end="")

        # Determine which Yobj to use for Pareto computation
        # If any of the objectives is negated, flip their sign to work in maximization space. This is
        # necessary as is_non_dominated method works by default in maximization mode.
        Yobj_for_pareto = self._Yobj.clone()
        Yobj_for_pareto[..., self._objective.obj_to_minimize] *= -1

        # Check if the problem is unconstrained (no constraints defined).
        # If unconstrained, all observations are feasible. Otherwise,
        # concatenate objectives and constraints along the last dimension,
        # then compute the feasibility mask: a point is feasible only if all
        # constraints ≤ 0
        if self._Ycon is None or self._output_constraints is None:
            feasible_mask = torch.ones(self._Yobj.shape[0], dtype=torch.bool, device=self._device)
        else:
            Y_full = torch.cat([self._Yobj, self._Ycon], dim=-1)
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in self._output_constraints]).all(dim=0)

        # Store the feasibility mask
        # self._feasible_observations_mask = feasible_mask

        # Initialize Pareto front mask with all False (no points marked yet)
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)

        # If there are any feasible points, compute Pareto front within
        # the dataset of feasible points only.
        if feasible_mask.any():
            pareto_mask[feasible_mask] = is_non_dominated(Yobj_for_pareto[feasible_mask])

        # Store the Pareto front mask
        # self._pareto_front_mask = pareto_mask

        # Extract objective values corresponding to the Pareto front points
        pareto_front = self._Yobj[pareto_mask]
        self.__setattr__("_pareto_front", pareto_front)
        # self._pareto_front = self._Yobj[pareto_mask]

        if verbose:
            print("✓.")

    def compute_hypervolume(self, verbose=True):
        """
        Compute the hypervolume. It assumes maximization.
        """
        if verbose:
            print("Computing hypervolume... ", end="")

        if self._pareto_front.shape[0] == 0:
            hv = torch.nan
            if verbose:
                print("✗ Cannot compute hypervolume. No Pareto front found.")
            self._hypervolume.append(hv)
            return

        # Negate only the dimensions that are minimization objectives
        pareto_front = self._pareto_front.clone()
        pareto_front[..., self._objective.obj_to_minimize] *= -1
        hv = Hypervolume(self._ref_point).compute(pareto_front)
        self._hypervolume.append(hv)

        if verbose:
            print("✓")
            print(f"Hypervolume = {self._hypervolume[-1]:>4.2f}")

    def optimize(self, verbose=True):

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=InputDataWarning)
        warnings.filterwarnings("ignore", category=NumericsWarning)
        warnings.filterwarnings("ignore", category=OptimizationWarning)

        t0 = time.monotonic()
        self.initialize_model(verbose=verbose)
        self.compute_reference_point(verbose=verbose)
        self.initialize_sampler(verbose=verbose)
        self.fit_model(verbose=verbose)
        self.initialize_acquisition_function(verbose=verbose)
        self.optimize_acquisition_function(verbose=verbose)
        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)
        if verbose:
            print(f"Calculation Time = {t1 - t0:>4.2f} s")

    def update_XY(
            self,
            new_X: torch.Tensor,
            new_Yobj: torch.Tensor,
            new_Yobj_var: torch.Tensor or None = None,
            new_Ycon: torch.Tensor or None = None,
            new_Ycon_var=None,
    ) -> None:

        if new_X is not None:
            new_X = new_X.to(self._device, self._dtype)
            self._X = torch.cat([self._X, new_X], dim=0)
        if new_Yobj is not None:
            new_Yobj = new_Yobj.to(self._device, self._dtype)
            self._Yobj = torch.cat([self._Yobj, new_Yobj], dim=0)
        if new_Yobj_var is not None:
            new_Yobj_var = new_Yobj_var.to(self._device, self._dtype)
            self._Yobj_var = torch.cat([self._Yobj_var, new_Yobj_var], dim=0)
        if new_Ycon is not None:
            new_Ycon = new_Ycon.to(self._device, self._dtype)
            self._Ycon = torch.cat([self._Ycon, new_Ycon], dim=0)
        if new_Ycon_var is not None:
            new_Ycon_var = new_Ycon_var.to(self._device, self._dtype)
            self._Ycon_var = torch.cat([self._Ycon_var, new_Ycon_var], dim=0)

    """ I/O """

    def to_file(self, output_path: Path = None):
        if output_path is None:
            output_path = Path.cwd() / "mobo.dat"
        # Ensure the parent directory exists
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as file:
            pickle.dump(self, file)
        return output_path

    def save_dataset_to_csv(self, output_path: Path = None):

        if output_path is None:
            output_path = Path.cwd() / "dataset.csv"

        XY = torch.cat([self._X, self._Yobj], dim=-1)
        if self._Yobj_var is not None:
            XY = torch.cat([XY, self._Yobj_var], dim=-1)
        if self._Ycon is not None:
            XY = torch.cat([XY, self._Ycon], dim=-1)
            if self._Ycon_var is not None:
                XY = torch.cat([XY, self._Ycon_var], dim=-1)

        XY = XY.detach().cpu().numpy()
        np.savetxt(output_path, XY, delimiter=",", comments="")

    def to_json(self, output_path: Path = None):
        """
         Serializes the serializable attributes of the Experiment instance to a JSON string.

         This method iterates through the instance's attributes (those starting with '_')
         and converts them to a JSON-compatible format. It handles:
         - datetime objects: converted to ISO 8601 strings.
         - torch.Tensor objects: converted to Python lists.
         - torch.device objects: converted to string representations (e.g., 'cpu', 'cuda:0').
         - torch.dtype objects: converted to their string name (e.g., 'float64').
         - Enum objects: converted to their string value.
         - Basic Python types (int, float, str, bool, None, lists of basic types):
           serialized directly.
         - Complex objects (like Callables, ModelListGP, ExactMarginalLogLikelihood instances)
           and lists containing them are skipped as they are not directly JSON serializable.

         Returns:
             str: A JSON string representation of the serializable attributes,
                  formatted with an indent of 4 for readability.
         """

        if output_path is None:
            output_path = Path.cwd() / "model.json"

        serializable_data = {}

        def serialize_value(value):
            """Helper function to recursively serialize individual values."""
            if isinstance(value, datetime.datetime):
                return value.isoformat()
            elif isinstance(value, torch.Tensor):
                return value.tolist()
            elif isinstance(value, torch.device):
                return str(value)
            elif isinstance(value, torch.dtype):
                return str(value).split('.')[-1]
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, (int, float, str, bool)) or value is None:
                return value
            elif isinstance(value, list):
                if all(isinstance(item, (int, float, str, bool, type(None))) for item in value):
                    return value
                else:
                    return None  # Skip complex lists
            elif isinstance(value, dict):
                if all(isinstance(k, str) for k in value.keys()):
                    serialized_dict = {}
                    for k, v in value.items():
                        serialized_item = serialize_value(v)
                        if serialized_item is not None:
                            serialized_dict[k] = serialized_item
                    return serialized_dict
                else:
                    return None  # Skip dicts with non-string keys
            else:
                return None  # Skip unhandled types

        # Iterate over all instance attributes
        for key, value in self.__dict__.items():
            json_key = key.lstrip('_')
            serialized_value = serialize_value(value)
            if serialized_value is not None:
                serializable_data[json_key] = serialized_value
            # else: skip

        # Save to disk
        with open(output_path, "w") as file:
            json.dump(serializable_data, file, indent=4)

    def load_dataset_from_csv(
            self,
            input_space_dim: int | None = None,
            objective_space_dim: int | None = None,
            constraint_space_dim: int | None = None,
            objective_variance: bool = False,
            constraint_variance: bool = False,
            filepath: str or None = None,
            skiprows: int = 0,
            skipcols: int = 0,
    ):
        """Assumes that the dataset is saved in the CSV format and columns are ordered as follows:
        X ¦ Yobj ¦ Yobj_var ¦ Ycon ¦ Ycon_var."""

        if input_space_dim is None:
            try:
                # Get input dimensions from existing X tensor if available
                input_space_dim = self._X.shape[-1]
            except (AttributeError, RuntimeError, TypeError):
                # X tensor isn't properly initialized or doesn't exist
                raise ValueError(
                    "Input space dimension must be provided explicitly as a parameter "
                    "when X tensor is not initialized. Could not infer dimension from self._X."
                )

        if objective_space_dim is None:
            try:
                # Get objective dimensions from existing Yobj tensor if available
                objective_space_dim = self._Yobj.shape[-1]
            except (AttributeError, RuntimeError, TypeError):
                # Yobj tensor not properly initialized or doesn't exist
                raise ValueError(
                    "Objective space dimension must be provided explicitly as a parameter "
                    "when Yobj tensor is not initialized. Could not infer dimension from self._Yobj."
                )

        if constraint_space_dim is None:
            try:
                constraints = self.get_output_constraints()
                if constraints is not None and self._Ycon is not None:
                    # The Problem is constrained and Ycon tensor exists
                    constraint_space_dim = self._Ycon.shape[-1]
                else:
                    # The Problem is unconstrained or Ycon tensor doesn't exist
                    constraint_space_dim = 0
            except (AttributeError, RuntimeError, TypeError):
                raise ValueError(
                    "Constraint space dimension must be provided explicitly as a parameter "
                    "since constraint tensor (Ycon) could not be determined automatically."
                )

        if filepath is None:
            csv_files = list(Path("..").glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the current directory")
            filepath = max(csv_files, key=lambda x: x.stat().st_mtime)

        xy = np.loadtxt(filepath, delimiter=",", skiprows=skiprows)

        i = skipcols + 0
        j = skipcols + input_space_dim
        self._X = torch.tensor(xy[..., i:j])

        if objective_space_dim > 0:
            i = j
            j += objective_space_dim
            self._Yobj = torch.tensor(xy[..., i:j])

            if objective_variance:
                i = j
                j += objective_space_dim
                self._Yobj_var = torch.tensor(xy[..., i:j])
            else:
                self._Yobj_var = None
        else:
            self._Yobj = None
            self._Yobj_var = None

        if constraint_space_dim > 0:
            i = j
            j += constraint_space_dim
            self._Ycon = torch.tensor(xy[..., i:j])

            if constraint_variance:
                i = j
                j += constraint_space_dim
                self._Ycon_var = torch.tensor(xy[..., i:j])
            else:
                self._Ycon_var = None
        else:
            self._Ycon = None
            self._Ycon_var = None

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
