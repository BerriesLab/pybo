import pickle
import torch
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
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective import is_non_dominated, Hypervolume
from botorch.utils.multi_objective.box_decompositions import (
    FastNondominatedPartitioning
)
from botorch.utils.transforms import normalize
from gpytorch.constraints import GreaterThan

from pybo.utils.cuda import get_device, get_supported_dtype
from pybo.utils.types import TorchDeviceType
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from pybo.utils.validators import *


class Mobo:
    """ A wrapper around BoTorch for Multi Objective Bayesian Optimization. Note that, similarly to BoTorch,
    this class is designed to work with maximization problems only. For maximization problems, the objective function
    must be negated.

    Note:
        - The model is always fit to the true data.
        - The acquisition function always maximizes.
        - The objective transformation (custom class) is the only place where negation for minimization happens."""

    def __init__(
            self,
            experiment_name: str,
            device: torch.device,
            dtype: torch.device.type,
            X: torch.Tensor | None = None,
            Yobj: torch.Tensor | None = None,
            Yobj_var: torch.Tensor | None = None,
            Ycon: torch.Tensor | None = None,
            Ycon_var: torch.Tensor | None = None,
            bounds: torch.Tensor | None = None,
            objective: Callable | None = None,
            output_constraints: list[Callable] | None = None,
            input_constraints: list[Callable] | None = None,
            acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.qEHVI,
            sampler_type: SamplerType = SamplerType.Sobol,
            batch_size: int = 1,
            mc_samples: int = 1024,
            raw_samples: int = 512,
            n_acqf_opt_iter: int = 500,
            max_n_acqf_opt_restarts: int = 10,
            max_attempts: int = 100,
    ):

        # Validate input arguments
        validate_experiment_name(experiment_name)
        validate_X(X)
        validate_Yobj(Yobj)
        validate_Yobj_var(Yobj_var)
        validate_Ycon(Ycon)
        validate_Ycon_var(Ycon_var)
        validate_bounds(bounds)
        validate_objective(objective)
        validate_constraints(output_constraints)
        validate_constraints(input_constraints)
        validate_acquisition_function(acquisition_function_type)
        validate_sampler_type(sampler_type)
        validate_batch_size(batch_size)
        validate_mc_samples(mc_samples)
        validate_raw_samples(raw_samples)
        validate_n_acqf_opt_iter(n_acqf_opt_iter)
        validate_max_n_acqf_opt_restarts(max_n_acqf_opt_restarts)

        # Experiment Name Attributes
        self._experiment_name = experiment_name  # A name used when saving results to file
        self._datetime = datetime.datetime.now()  # A datetime stamp used when saving results to file

        # Device Attributes
        self._device = device if device is not None else get_device()
        self._dtype = dtype if dtype is not None else get_supported_dtype(self._device)

        # Problem Attributes
        self._X: torch.Tensor = X.to(self._device, self._dtype) if X is not None else None  # Input variables
        self._Yobj: torch.Tensor = Yobj.to(self._device, self._dtype) if Yobj is not None else None
        self._Ycon: torch.Tensor = Ycon.to(self._device, self._dtype) if Ycon is not None else None
        self._Yobj_var: torch.Tensor = Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None
        self._Ycon_var: torch.Tensor = Ycon_var.to(self._device, self._dtype) if Ycon_var is not None else None

        # Observed output_constraints variables
        self._bounds: torch.Tensor = bounds.to(self._device, self._dtype)  # A '2 x d' tensor of lower and upper bounds
        self._acquisition_function_type = acquisition_function_type  # Type of acquisition function used for optimization
        self._acquisition_function_instance = None  # Instance of the acquisition function - instantiated within the optimization loop
        self._sampler_type = sampler_type  # Type of sampler used for initialization and acquisition function optimization
        self._sampler_instance = None  # Instance of the sampler - instantiated within the optimization loop
        self._objective = objective  # The ground truth (multi)objective function
        self._output_constraints = output_constraints  # The functional output_constraints
        self._input_constraints = input_constraints  # The functional input_constraints
        self._n_acqf_opt_iter = n_acqf_opt_iter  # Number of iterations for acquisition function optimization
        self._max_n_acqf_opt_restarts = max_n_acqf_opt_restarts  # Max number of restarts for acquisition function optimization
        self._max_attempts = max_attempts  # Max number of optimization attempts if new X does not satisfy input constraints
        self._batch_size = batch_size  # Number of candidates to be generated in parallel in each optimization step
        self._num_mc_samples = mc_samples  # Number of samples for initialization and acquisition function optimization
        self._num_raw_samples = raw_samples  # Number of samples for acquisition function optimization
        self._pareto_front_mask = None
        self._feasible_observations_mask = None

        # State Attributes
        self._model: ModelListGP | None = None
        self._mlls: list[ExactMarginalLogLikelihood] | list = []
        self._pareto_front: torch.Tensor | None = None
        self._pareto_mask: torch.Tensor | None = None
        self._ref_point: torch.Tensor | None = None
        self._new_X: torch.Tensor | None = None

        # Metrics
        self._hypervolume: float | None = None
        self._elapsed_time: float | None = None

    """ Setters and getters """

    def set_experiment_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        self._experiment_name = name

    def get_experiment_name(self):
        return self._experiment_name

    def set_datetime(self, date_time: datetime.datetime):
        validate_datetime(date_time)
        self._datetime = date_time

    def get_datetime(self):
        return self._datetime

    def set_X(self, X: torch.Tensor):
        validate_X(X)
        self._X = X.to(self._device, self._dtype)

    def get_X(self):
        return self._X

    def set_Yobj(self, Yobj: torch.Tensor):
        validate_Yobj(Yobj)
        self._Yobj = Yobj.to(self._device, self._dtype)

    def get_Yobj(self) -> torch.Tensor | None:
        return self._Yobj

    def set_Yobj_var(self, Yobj_var: torch.Tensor | None = None):
        validate_Yobj_var(Yobj_var)
        self._Yobj_var = (
            Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None
        )

    def get_Yobj_var(self):
        return self._Yobj_var

    def set_Ycon(self, Ycon: torch.Tensor | None):
        validate_Ycon(Ycon)
        self._Ycon = Ycon.to(self._device, self._dtype) if Ycon is not None else None

    def get_Ycon(self):
        return self._Ycon

    def set_Ycon_var(self, Ycon_var: torch.Tensor | None = None):
        validate_Yobj_var(Ycon_var)
        self._Ycon_var = (
            Ycon_var.to(self._device, self._dtype) if Ycon_var is not None else None
        )

    def get_Ycon_var(self):
        return self._Ycon_var

    def get_new_X(self):
        return self._new_X

    def set_bounds(self, bounds: torch.Tensor):
        validate_bounds(bounds)
        self._bounds = bounds.to(self._device, self._dtype)

    def get_bounds(self):
        return self._bounds

    def get_device(self):
        return self._device

    def set_device(self, device: TorchDeviceType):
        self._device = torch.device(device)

    def set_dtype(self):
        self._dtype = get_supported_dtype(self._device)

    def get_dtype(self):
        return self._dtype

    def get_model(self):
        return self._model

    def set_acquisition_function(self, acquisition_function_type: AcquisitionFunctionType):
        validate_acquisition_function(acquisition_function_type)
        self._acquisition_function_type = acquisition_function_type

    def get_acquisition_function(self):
        return self._acquisition_function_type

    def set_sampler_type(self, sampler_type: SamplerType):
        validate_sampler_type(sampler_type)
        self._sampler_type = sampler_type

    def set_batch_size(self, batch_size: int):
        """Set the number of candidates to be generated in each optimization step."""
        validate_batch_size(batch_size)
        self._batch_size = batch_size

    def get_batch_size(self):
        return self._batch_size

    def set_mc_samples(self, MC_samples: int):
        validate_mc_samples(MC_samples)
        self._num_mc_samples = MC_samples

    def get_mc_samples(self):
        return self._num_mc_samples

    def set_raw_samples(self, raw_samples: int):
        validate_raw_samples(raw_samples)
        self._num_raw_samples = raw_samples

    def get_raw_samples(self):
        return self._num_raw_samples

    def set_objective(self, objective: Callable or None):
        validate_objective(objective)
        self._objective = objective

    def get_objective(self):
        return self._objective

    def get_pareto(self):
        return self._pareto_front

    def get_pareto_front_mask(self):
        return self._pareto_front_mask

    def get_feasible_observations_mask(self):
        return self._feasible_observations_mask

    def set_output_constraints(self, constraints: list[Callable] or None = None):
        """Set non-linear output_constraints on the output domain (Y)."""
        validate_constraints(constraints)
        self._output_constraints = constraints

    def get_output_constraints(self):
        return self._output_constraints

    def set_input_constraints(self, constraints: list[Callable] or None = None):
        """Set non-linear input_constraints on the input domain (X)."""
        validate_constraints(constraints)
        self._input_constraints = constraints

    def get_input_constraints(self):
        return self._input_constraints

    def add_constraint(self, constraint: Callable):
        validate_constraints([constraint,])
        self._output_constraints.append(constraint)

    def get_hypervolume(self):
        return self._hypervolume

    def get_ref_point(self):
        return self._ref_point

    def get_elapsed_time(self):
        return self._elapsed_time

    """ Optimizer """

    def initialize_model(self, verbose=True):
        """ Initialize Gaussian Process models for the objectives and constraints.

        This method prepares the training dataset by combining the objective and constraint
        observations (and optionally their variances). Then it creates one independent
        SingleTaskGP model for each output dimension (each objective or constraint).
        These models are combined into a ModelListGP to jointly represent the full
        multi-output model.

        Important: The GP model in BoTorch is a pure regression model: it simply fits the data
        it receives. It does not know or care about whether the model is for objectives to
        minimize or maximize. As such, the model must always and only receive the true, unnegated
        objective values as training data. In other words, the model is always fit to the true data."""

        if verbose:
            print("Initializing model... ", end="")

        # Prepare dataset by concatenating the objectives
        train_x, train_y, train_y_var = self.prepare_training_dataset()

        # Initialize models - one model for each objective (or observable)
        models = []
        for i in range(0, train_y.shape[-1]):
            models.append(
                SingleTaskGP(
                    train_X=train_x,
                    train_Y=train_y[..., i: i + 1],
                    train_Yvar=(train_y_var[..., i: i + 1] if train_y_var is not None else None),
                    input_transform=Normalize(d=self._X.shape[-1], bounds=self._bounds),
                    outcome_transform=Standardize(m=1),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6)),
                )
            )
        self._model = ModelListGP(*models)
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

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

    def compute_reference_point(self, verbose=True, buffer=0.1):
        """
        Compute and set the reference point for hypervolume calculations.

        The reference point is a key component in multi-objective optimization,
        representing a point in objective space that is dominated by all observed
        solutions. It is used to compute the hypervolume improvement metric.

        This method sets the reference point only once (typically at the first iteration).
        If the objective already provides a reference point (`self._objective.ref_point`),
        that point is used directly. Otherwise, the reference point is automatically
        computed based on observed objective values, with a small buffer added to ensure
        it lies beyond the worst observed point.

        The method correctly handles whether the problem is a maximization or
        minimization problem by using the `negate` flag in the objective:
        - For maximization (negate=False), the reference point is set slightly worse
          than the minimum observed objective values.
        - For minimization (negate=True), the reference point is set slightly worse
          than the maximum negated objective values (i.e., the worst true objective)."""
        
        # The reference point is calculated only for the first iteration step
        if isinstance(self._ref_point, torch.Tensor):
            return

        if verbose:
            print("Defining reference point... ", end="")

        # The reference point must provided within the objective class.
        ref_point = getattr(self._objective, "ref_point", None)
        if ref_point is None:
            raise ValueError("Reference point must be defined.")
        self._ref_point = ref_point.clone().to(self._device, self._dtype)
        self._ref_point[..., self._objective.negate] *= -1

        if verbose:
            print("✓")
            print(f"Reference point (in maximization space): {self._ref_point.detach().cpu().numpy()}")

    def initialize_acquisition_function(self, verbose=True):
        if verbose:
            print("Initializing acquisition function... ", end="")

        if self._acquisition_function_type == AcquisitionFunctionType.qEHVI:
            raise NotImplementedError("qEHVI is not yet implemented.")
            # self._acquisition_function_instance = qExpectedHypervolumeImprovement(
            #     model=self._model,
            #     ref_point=self._ref_point,
            #     partitioning=self._partitioning,
            #     sampler=self._sampler_instance,
            #     objective=self._objective,
            #     constraints=self._output_constraints,
            # )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogEHVI:
            raise NotImplementedError("qLohEHVI is not yet implemented.")
            # self._acquisition_function_instance = qLogExpectedHypervolumeImprovement(
            #     model=self._model,
            #     ref_point=self._ref_point,
            #     partitioning=self._partitioning,
            #     sampler=self._sampler_instance,
            #     objective=self._objective,
            #     constraints=self._output_constraints,
            # )

        elif self._acquisition_function_type == AcquisitionFunctionType.qNEHVI:
            self._acquisition_function_instance = qNoisyExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                X_baseline=normalize(self._X, self._bounds),
                sampler=self._sampler_instance,
                prune_baseline=True,
                objective=self._objective,
                constraints=self._output_constraints,
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogNEHVI:
            self._acquisition_function_instance = (
                qLogNoisyExpectedHypervolumeImprovement(
                    model=self._model,
                    ref_point=self._ref_point,
                    X_baseline=normalize(self._X, self._bounds),
                    prune_baseline=True,
                    sampler=self._sampler_instance,
                    objective=self._objective,
                    constraints=self._output_constraints,
                )
            )

        else:
            raise ValueError(
                f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}."
            )

        if verbose:
            print("✓")

    def initialize_partitioning(self, verbose=True):
        raise NotImplementedError("Partitioning is not yet implemented.")
        # if verbose:
        #      print("Initializing partitioning function... ", end="")
        #
        # self._partitioning = FastNondominatedPartitioning(
        #     ref_point=self._ref_point, Y=self._Yobj
        # )
        #
        # if verbose:
        #     print(" Done.")

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
                if restart_on_error and restart_count < self._max_n_acqf_opt_restarts:
                    print("x")
                    print(
                        f"Restarting fitting... (Attempt {restart_count + 1}/{self._max_n_acqf_opt_restarts})"
                    )
                    restart_count += 1
                else:
                    raise e  # Raise if not restarting or max restarts reached
        return None

    def optimize_acquisition_function_loop(self, verbose=True):
        if self._input_constraints:
            for attempt in range(1, self._max_attempts + 1):
                if verbose:
                    if attempt > 1:
                        print("The new X does not satisfy the input constraints\n")
                    print(f"Attempt {attempt}/{self._max_attempts}. Optimizing acquisition function... ", end="")
                self.optimize_acquisition_function()
                if verbose:
                    print("✓")

                if all(torch.all(c(self._new_X) < 0) for c in self._input_constraints):
                    break
            else:
                raise ValueError(f"Could not find a new X that satisfies all input constraints after {self._max_attempts} attempts.")
        else:
            if verbose:
                print("Optimizing acquisition function... ", end="")
            self.optimize_acquisition_function()
            if verbose:
                print("✓")

    def optimize_acquisition_function(self):
        self._new_X, _ = optimize_acqf(
            acq_function=self._acquisition_function_instance,
            bounds=self._bounds,
            q=self._batch_size,
            num_restarts=self._max_n_acqf_opt_restarts,
            raw_samples=self._num_raw_samples,
            options={"maxiter": self._n_acqf_opt_iter, "disp": True},
            sequential=True,
        )

    def compute_acquisition_function_value_at_X(self,X: torch.Tensor, verbose=True):
        acq_val = self._acquisition_function_instance(X)
        if verbose:
            print(f"Acquisition function value at {X.detach().cpu().numpy()}: {acq_val.detach().cpu().numpy()}")

    def compute_posterior_mean_at_X(self, X: torch.Tensor, verbose=True):
        posterior = self._model.posterior(X)
        if verbose:
            print(f"Posterior mean at {X.detach().cpu().numpy()}: {posterior.mean.detach().cpu().numpy()}")

    def compute_pareto_front(self, verbose=True):
        """
        Compute the Pareto front considering constraints and negation.

        This method filters the objective observations by constraints and identifies
        non-dominated points (Pareto optimal). It handles the sign flip
        if the problem is a minimization problem (negate=True), by performing
        non-dominated checks in maximization space and then returning the Pareto
        front in the original objective space.

        Side Effects:
        - Sets self._par_mask: boolean mask indicating Pareto-optimal points.
        - Sets self._con_mask: boolean mask indicating feasible points w.r.t. constraints.
        - Sets self._pareto_front: tensor of Pareto-optimal objective points.
        """

        if verbose:
            print("Finding Pareto front... ", end="")

        # Determine which Yobj to use for Pareto computation
        # If any of the objectives is negated, flip their sign to work in maximization space. This is
        # necessary as is_non_dominated method works by default in maximization mode.
        Yobj_for_pareto = self._Yobj.clone()
        Yobj_for_pareto[..., self._objective.negate] *= -1

        # Check if the problem is unconstrained (no constraints defined)
        if self._Ycon is None or self._output_constraints is None:
            # If unconstrained, all observations are feasible
            feasible_mask = torch.ones(self._Yobj.shape[0], dtype=torch.bool, device=self._device)
        else:
            # For constrained problems, concatenate objectives and constraints along last dimension,
            # then compute feasibility mask: a point is feasible only if all constraints ≤ 0
            Y_full = torch.cat([self._Yobj, self._Ycon], dim=-1)
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in self._output_constraints]).all(dim=0)

        # Store the feasibility mask
        self._feasible_observations_mask = feasible_mask

        # Initialize Pareto front mask with all False (no points marked yet)
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)

        # If there are any feasible points, compute Pareto front among them
        if feasible_mask.any():
            # Find non-dominated points (Pareto optimal) within feasible points only
            pareto_mask[feasible_mask] = is_non_dominated(Yobj_for_pareto[feasible_mask])

        # Store the Pareto front mask
        self._pareto_front_mask = pareto_mask

        # Extract objective values corresponding to the Pareto front points
        self._pareto_front = self._Yobj[pareto_mask]

        if verbose:
            print("✓.")

    def compute_hypervolume(self, verbose=True):
        if verbose:
            print("Computing hypervolume... ", end="")

        if self._pareto_front.shape[0] == 0:
            hv = torch.nan
            if verbose:
                print("✗ Cannot compute hypervolume. No Pareto front found.")
            self._hypervolume = hv
            return

        # Negate only the dimensions that are minimization objectives
        pareto_front = self._pareto_front.clone()
        pareto_front[..., self._objective.negate] *= -1
        # ref_point = self._objective.ref_point.clone()
        # ref_point[..., self._objective.negate] *= -1
        hv = Hypervolume(self._ref_point).compute(pareto_front)
        self._hypervolume = hv

        if verbose:
            print("✓")
            print(f"Hypervolume = {self._hypervolume:>4.2f}")

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
        if self._acquisition_function_type.value in AcquisitionFunctionType.require_partitioning():
            self.initialize_partitioning(verbose=verbose)
        self.initialize_acquisition_function(verbose=verbose)
        self.optimize_acquisition_function_loop(verbose=verbose)
        t1 = time.monotonic()
        self._elapsed_time = t1 - t0
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
            self._X = torch.cat([self._X, new_X], dim=0)
        if new_Yobj is not None:
            self._Yobj = torch.cat([self._Yobj, new_Yobj], dim=0)
        if new_Yobj_var is not None:
            self._Yobj_var = torch.cat([self._Yobj_var, new_Yobj_var], dim=0)
        if new_Ycon is not None:
            self._Ycon = torch.cat([self._Ycon, new_Ycon], dim=0)
        if new_Ycon_var is not None:
            self._Ycon_var = torch.cat([self._Ycon_var, new_Ycon_var], dim=0)

    """ I/O """

    def to_file(self, output_path=None):
        if output_path is None:
            output_path = Path.cwd() / "mobo.dat"  # compose_model_filename(iteration_number=self._iteration_number)
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
