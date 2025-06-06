import pickle
import warnings
import time
import glob
import os

import botorch
import gpytorch
from botorch.exceptions import BadInitialCandidatesWarning, InputDataWarning, OptimizationWarning
from botorch.exceptions.warnings import NumericsWarning
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.multi_objective import is_non_dominated, Hypervolume
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.transforms import normalize
from gpytorch.constraints import GreaterThan

from pybo.utils.cuda import get_device, get_supported_dtype
from pybo.utils.types import TorchDeviceType
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement, \
    qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from pybo.validators import *


class Mobo:

    def __init__(
            self,
            experiment_name: str,
            X: torch.Tensor | None = None,
            Yobj: torch.Tensor | None = None,
            Yobj_var: torch.Tensor | None = None,
            Ycon: torch.Tensor | None = None,
            Ycon_var: torch.Tensor | None = None,
            bounds: torch.Tensor | None = None,
            objective: Callable | None = None,
            optimization_problem_type: OptimizationProblemType = OptimizationProblemType.Maximization,
            true_objective: Callable = None,
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
            device: torch.device = None,
            dtype: torch.device.type = None
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
        validate_true_objective(true_objective)
        validate_optimization_problem(optimization_problem_type)
        validate_constraints(output_constraints)
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
        if device is None:
            self._device = get_device()  # The device used for computation (e.g., GPU, CPU or MPS)
        else:
            self._device = device
        if dtype is None:
            self._dtype = get_supported_dtype(self._device)  # The data type used for computation - Inferred from device
        else:
            self._dtype = dtype

        # Problem Attributes
        self._X: torch.Tensor = X.to(self._device, self._dtype) if X is not None else None  # Input variables
        self._Yobj: torch.Tensor = Yobj.to(self._device, self._dtype) if Yobj is not None else None  # Objective observables
        self._Ycon: torch.Tensor = Ycon.to(self._device, self._dtype) if Ycon is not None else None  # Constrained observables
        self._Yobj_var: torch.Tensor = Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None  # Observed variance
        self._Ycon_var: torch.Tensor = Ycon_var.to(self._device, self._dtype) if Ycon_var is not None else None  # Observed output_constraints variables
        self._bounds: torch.Tensor = bounds.to(self._device, self._dtype)  # A '2 x d' tensor of lower and upper bounds for each column of 'X'
        self._optimization_problem_type = optimization_problem_type  # Type of optimization problem (minimization or maximization)
        self._acquisition_function_type = acquisition_function_type  # Type of acquisition function used for optimization
        self._acquisition_function_instance = None  # Instance of the acquisition function - instantiated within the optimization loop
        self._sampler_type = sampler_type  # Type of sampler used for initialization and acquisition function optimization
        self._sampler_instance = None  # Instance of the sampler - instantiated within the optimization loop
        self._true_objective = true_objective  # The ground truth (multi)objective function
        self._objective = objective  # The (multi)objective function to be optimized
        self._output_constraints = output_constraints  # The functional output_constraints
        self._input_constraints = input_constraints  # The functional input_constraints
        self._n_acqf_opt_iter = n_acqf_opt_iter  # Number of iterations for acquisition function optimization
        self._max_n_acqf_opt_restarts = max_n_acqf_opt_restarts  # Max number of restarts for acquisition function optimization
        self._max_attempts = max_attempts  # Max number of optimization attempts if new X does not satisfy input
        # constraints
        self._batch_size = batch_size  # Number of candidates to be generated in parallel in each optimization step
        self._mc_samples = mc_samples  # Number of samples for initialization and acquisition function optimization
        self._raw_samples = raw_samples  # Number of samples for acquisition function optimization
        self._par_mask = None
        self._con_mask = None

        # State Attributes
        self._model: ModelListGP | None = None
        self._mlls: list[ExactMarginalLogLikelihood] | list = []
        self._pareto_front: torch.Tensor | None = None
        self._pareto_mask: torch.Tensor | None = None
        self._ref_point: torch.Tensor | None = None
        self._new_X: torch.Tensor | None = None

        # Metrics
        self._iteration_number = 0  # Number of iterations - Incremental value
        self._hypervolume = []  # Hypervolume spanned by the pareto front for each iteration step
        self._elapsed_time = []  # Time elapsed for each iteration step
        self._allocated_memory = []  # Memory allocated for each iteration step

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
        self._Yobj_var = Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None

    def get_Yobj_var(self):
        return self._Yobj_var

    def set_Ycon(self, Ycon: torch.Tensor | None):
        validate_Ycon(Ycon)
        self._Ycon = Ycon.to(self._device, self._dtype) if Ycon is not None else None

    def get_Ycon(self):
        return self._Ycon

    def set_Ycon_var(self, Ycon_var: torch.Tensor | None = None):
        validate_Yobj_var(Ycon_var)
        self._Ycon_var = Ycon_var.to(self._device, self._dtype) if Ycon_var is not None else None

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

    def set_optimization_problem(self, optimization_problem_type: OptimizationProblemType):
        """ Set the optimization problem as maximization or minimization. Note that
        BoTorch supports natively only maximization problems, therefore, this setting is
        used to negate the objective function and choose appropriate values for the
        reference point."""
        validate_optimization_problem(optimization_problem_type)
        self._optimization_problem_type = optimization_problem_type

    def get_optimization_problem_type(self):
        return self._optimization_problem_type

    def set_sampler_type(self, sampler_type: SamplerType):
        validate_sampler_type(sampler_type)
        self._sampler_type = sampler_type

    def set_batch_size(self, batch_size: int):
        """ Set the number of candidates to be generated in each optimization step."""
        validate_batch_size(batch_size)
        self._batch_size = batch_size

    def get_batch_size(self):
        return self._batch_size

    def set_mc_samples(self, MC_samples: int):
        validate_mc_samples(MC_samples)
        self._mc_samples = MC_samples

    def get_mc_samples(self):
        return self._mc_samples

    def set_raw_samples(self, raw_samples: int):
        validate_raw_samples(raw_samples)
        self._raw_samples = raw_samples

    def get_raw_samples(self):
        return self._raw_samples

    def set_true_objective(self, true_objective: MultiObjectiveTestProblem):
        validate_true_objective(true_objective)
        self._true_objective = true_objective

    def get_true_objective(self):
        return self._true_objective

    def set_objective(self, objective: Callable or None):
        validate_objective(objective)
        self._objective = objective

    def get_objective(self):
        return self._objective

    def get_pareto(self):
        return self._pareto_front

    def get_par_mask(self):
        return self._par_mask

    def get_con_mask(self):
        return self._con_mask

    def set_output_constraints(self, constraints: list[Callable] or None = None):
        """ Set non-linear output_constraints on the output domain (Y). """
        validate_constraints(constraints)
        self._output_constraints = constraints

    def get_output_constraints(self):
        return self._output_constraints

    def add_constraint(self, constraint: Callable):
        validate_constraints([constraint, ])
        self._output_constraints.append(constraint)

    def get_hypervolume(self):
        return self._hypervolume

    def get_iteration_number(self):
        return self._iteration_number

    def get_ref_point(self):
        return self._ref_point

    def get_elapsed_time(self):
        return self._elapsed_time

    def get_allocated_memory(self):
        return self._allocated_memory

    """ Optimizer """

    def _initialize_model(self, verbose=True):
        """ Define models for objectives and output_constraints. """

        if verbose:
            print("Initializing model...", end="")
        
        # Prepare dataset by concatenating the objectives
        train_x, train_y, train_y_var = self._prepare_training_dataset()

        # Initialize models
        models = []
        for i in range(0, train_y.shape[-1]):
            models.append(
                SingleTaskGP(
                    train_x,
                    train_y[..., i: i + 1],
                    train_y_var[..., i: i + 1] if train_y_var is not None else None,
                    input_transform=Normalize(d=self._X.shape[-1], bounds=self._bounds),
                    outcome_transform=Standardize(m=1),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
                )
            )
        self._model = ModelListGP(*models)
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        if verbose:
            print(" Done.")

    def _prepare_training_dataset(self):
        """Prepare training data by combining objectives and output_constraints."""

        train_x = self._X

        # Combine objectives and output_constraints if they exist
        train_y = (
            torch.cat((self._Yobj, self._Ycon), dim=-1)
            if self._Ycon is not None
            else self._Yobj
        )

        # Combine variances if variances exist
        train_y_var = None
        if self._Yobj_var is not None and self._Ycon_var is not None:
            train_y_var = torch.cat((self._Yobj_var, self._Ycon_var), dim=-1)
        elif self._Yobj_var is not None:
            train_y_var = self._Yobj_var

        return train_x, train_y, train_y_var

    def _initialize_sampler(self, verbose=True, ):
        if verbose:
            print("Initializing sampler...", end="")

        if self._sampler_type.name == SamplerType.Sobol.name:
            self._sampler_instance = SobolQMCNormalSampler(torch.Size([self._mc_samples]))
        else:
            raise ValueError("Only Sobol Sampler is currently supported.")

        if verbose:
            print(" Done.")

    def _initialize_acquisition_function(self, verbose=True, ):
        if verbose:
            print("Initializing acquisition function...", end="")

        if self._acquisition_function_type == AcquisitionFunctionType.qEHVI:
            self._acquisition_function_instance = qExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._output_constraints
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogEHVI:
            self._acquisition_function_instance = qLogExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._output_constraints,
            )

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
            self._acquisition_function_instance = qLogNoisyExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                X_baseline=normalize(self._X, self._bounds),
                prune_baseline=True,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._output_constraints,
            )

        else:
            raise ValueError(f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}.")

        if verbose:
            print(" Done.")

    def _initialize_partitioning(self, verbose=True, ):
        if verbose:
            print("Initializing partitioning function...", end="")

        self._partitioning = FastNondominatedPartitioning(ref_point=self._ref_point, Y=self._Yobj)

        if verbose:
            print(" Done.")

    def _compute_reference_point(self, verbose=True, buffer=0.1):

        # The reference point is calculated only for the first iteration step
        if self._iteration_number > 0:
            return

        if verbose:
            print("Defining reference point...", end="")

        # If the true objective is provided, then use its reference point if exists
        if self._true_objective is not None and hasattr(self._true_objective, "_ref_point"):
            self._ref_point = self._true_objective.ref_point.to(self._device, self._dtype)

        # Otherwise, compute a reference point
        else:
            # Handle maximization problems
            if self._optimization_problem_type == OptimizationProblemType.Maximization:
                worst = torch.min(self._Yobj, dim=0).values
                self._ref_point = (worst - buffer * torch.abs(worst)).to(self._device, self._dtype)

            # Handle minimization problems
            elif self._optimization_problem_type == OptimizationProblemType.Minimization:
                worst = torch.max(self._Yobj, dim=0).values
                self._ref_point = (worst + buffer * torch.abs(worst)).to(self._device, self._dtype)

        if verbose:
            print(f" {self._ref_point}... Done")

    def _fit_model(self, restart_on_error=True, verbose=True, ):
        if not isinstance(self._model, ModelListGP):
            raise ValueError("Model must be initialized before fitting.")

        if verbose:
            print("Fitting model...", end="")

        restart_count = 0
        while True:
            try:
                botorch.fit_gpytorch_mll(self._mll)
                if verbose:
                    print(" Done.")
                break  # Exit the inner loop on success

            except Exception as e:
                if restart_on_error and restart_count < self._max_n_acqf_opt_restarts:
                    print(f"Restarting fitting... (Attempt {restart_count + 1}/{self._max_n_acqf_opt_restarts})")
                    restart_count += 1
                else:
                    raise e  # Raise if not restarting or max restarts reached
        return None

    def _optimize_acquisition_function(self, verbose=True, ):
        if verbose:
            print("Optimizing acquisition function...", end="")

        self._new_X, _ = optimize_acqf(
            acq_function=self._acquisition_function_instance,
            bounds=self._bounds,
            q=self._batch_size,
            num_restarts=self._max_n_acqf_opt_restarts,
            raw_samples=self._raw_samples,
            options={"maxiter": self._n_acqf_opt_iter, "disp": True},
            sequential=True,
        )

        if verbose:
            print(" Done.")
            print(f"New X: {self._new_X}")

    def _compute_hypervolume(self, verbose=True, ):
        if verbose:
            print("Computing hypervolume...", end="")

        if self._pareto_front.shape[0] == 0:
            volume = 0
        else:
            if self._optimization_problem_type == OptimizationProblemType.Maximization:
                hv = Hypervolume(self._ref_point)
                volume = hv.compute(self._pareto_front)
            else:
                # The minus sign is needed to account for the fact that BoTorch computes the hypervolume conventionally
                # for a maximization problem, so it expects to find the reference to the lower-left side (in 2D) of the
                # pareto front.
                hv = Hypervolume(-self._ref_point)
                volume = hv.compute(-self._pareto_front)
        self._hypervolume.append(volume)

        if verbose:
            print(" Done.")
            print(f"Hypervolume = {self._hypervolume[-1]:>4.2f}")

    def _compute_pareto_front(self, verbose=True):
        """The pareto front is computed by considering only the objective observations for which the corresponding constraint
        observations satisfy the constraint functions. This is done by filtering out the acceptable Yobj observations."""

        if verbose:
            print("Finding Pareto front...", end="")

        if self._Ycon is None or self._output_constraints is None:
            # If the problem is unconstrained, then all observations can be Pareto-optimal

            self._par_mask = is_non_dominated(self._Yobj, maximize=self._optimization_problem_type.value)
            self._con_mask = torch.ones_like(self._par_mask, dtype=torch.bool).to(self._device, self._dtype)
            mask = self._par_mask

        else:
            # If the problem is constrained, then only the observations satisfying the output_constraints
            # can be Pareto-optimal. Therefore, first find the acceptable observations, i.e., those observations
            # satisfy the output_constraints and build a logical mask. # Then, find the acceptable observations that
            # are non-dominated and build a logical mask.

            Y = torch.cat([self._Yobj, self._Ycon], dim=-1)
            constraint_vals = [c(Y) for c in self._output_constraints]
            self._con_mask = torch.stack([(cv <= 0) for cv in constraint_vals]).all(dim=0)

            feasible_obj = self._Yobj[self._con_mask]
            par_mask_feasible = is_non_dominated(feasible_obj, maximize=self._optimization_problem_type.value)
            mask = torch.zeros_like(self._con_mask)
            mask[self._con_mask] = par_mask_feasible
            self._par_mask = mask

        self._pareto_front = self._Yobj[mask]

        if verbose:
            print(" Done.")

    def optimize(self, verbose=True):

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=InputDataWarning)
        warnings.filterwarnings("ignore", category=NumericsWarning)
        warnings.filterwarnings("ignore", category=OptimizationWarning)

        t0 = time.monotonic()
        self._initialize_model(verbose=verbose)
        self._compute_reference_point(verbose=verbose)
        self._initialize_sampler(verbose=verbose)
        self._fit_model(verbose=verbose)

        if self._acquisition_function_type.value in AcquisitionFunctionType.require_partitioning():
            self._initialize_partitioning(verbose=verbose)

        self._initialize_acquisition_function(verbose=verbose)

        if self._input_constraints:
            for attempt in range(1, self._max_attempts + 1):
                if verbose:
                    if attempt > 1:
                        print("The new X does not satisfy the input constraints\n")
                    print(f"Attempt {attempt}/{self._max_attempts}")

                self._optimize_acquisition_function(verbose=verbose)

                if all(torch.all(c(self._new_X) < 0) for c in self._input_constraints):
                    break
            else:
                raise ValueError(f"Could not find a new X that satisfies all input constraints after {self._max_attempts} attempts.")
        else:
            self._optimize_acquisition_function(verbose=verbose)

        self._compute_pareto_front(verbose=verbose)
        self._compute_hypervolume(verbose=verbose)
        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)
        print(f"Calculation Time = {t1 - t0:>4.2f}")
        self._iteration_number += 1
        if self._device.type == "cuda":
            self._allocated_memory.append(torch.cuda.memory_allocated() / 1024 ** 2)

    def update_XY(
            self,
            new_X: torch.Tensor,
            new_Yobj: torch.Tensor,
            new_Yobj_var: torch.Tensor or None = None,
            new_Ycon: torch.Tensor or None = None,
            new_Ycon_var=None
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

    def to_file(self):
        filepath = compose_model_filename(iteration_number=self._iteration_number)
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        return filepath

    def save_dataset_to_csv(self):

        XY = torch.cat([self._X, self._Yobj], dim=-1)
        if self._Yobj_var is not None:
            XY = torch.cat([XY, self._Yobj_var], dim=-1)
        if self._Ycon is not None:
            XY = torch.cat([XY, self._Ycon], dim=-1)
            if self._Ycon_var is not None:
                XY = torch.cat([XY, self._Ycon_var], dim=-1)

        XY = XY.detach().cpu().numpy()
        filepath = compose_dataset_filename(self._iteration_number)
        np.savetxt(filepath, XY, delimiter=",", comments="")

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
        """ Assumes that the dataset is saved in the CSV format and columns are ordered as follows:
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
            csv_files = list(Path('.').glob('*.csv'))
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
    def from_file(cls, filepath: str = None):
        """Load a MOBO instance from a file.
        If no filepath is provided, load the most recent .dat file from the current directory."""
        if filepath is None:
            files = glob.glob('*.dat')
            if not files:
                raise FileNotFoundError("No .dat files found in current directory")
            filepath = max(files, key=os.path.getctime)

        with open(filepath, 'rb') as f:
            return pickle.load(f)
