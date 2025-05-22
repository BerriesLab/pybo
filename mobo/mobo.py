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

from utils.cuda import get_device, get_supported_dtype
from utils.types import TorchDeviceType
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement, \
    qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from mobo.validators import *
from utils.io import *


class Mobo:

    def __init__(
            self,
            experiment_name: str,
            X: torch.Tensor,
            Yobj: torch.Tensor,
            Yobj_var: torch.Tensor | None = None,
            Ycon: torch.Tensor | None = None,
            Ycon_var: torch.Tensor | None = None,
            bounds: torch.Tensor | None = None,
            objective: Callable | None = None,
            optimization_problem_type: OptimizationProblemType = OptimizationProblemType.Maximization,
            true_objective=None,
            constraints: list[Callable] | None = None,
            acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.qEHVI,
            sampler_type: SamplerType = SamplerType.Sobol,
            batch_size: int = 1,
            mc_samples: int = 1024,
            raw_samples: int = 512,
            n_acqf_opt_iter: int = 500,  # Number of iterations for acquisition function optimization
            max_n_acqf_opt_restarts: int = 10,  # Max number of restarts for acquisition function optimization
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
        validate_constraints(constraints)
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
        self._device = get_device()  # The device used for computation (e.g., GPU, CPU or MPS)
        self._dtype = get_supported_dtype(self._device)  # The data type used for computation - Inferred from device

        # Problem Attributes
        self._X: torch.Tensor = X.to(self._device, self._dtype)  # Input variables
        self._Yobj: torch.Tensor = Yobj.to(self._device, self._dtype)  # Objective observables
        self._Ycon: torch.Tensor = Ycon.to(self._device, self._dtype) if Ycon is not None else None  # Constrained observables
        self._Yobj_var: torch.Tensor = Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None  # Observed variance
        self._Ycon_var: torch.Tensor = Ycon_var.to(self._device, self._dtype) if Ycon_var is not None else None  # Observed constraints variables
        self._bounds: torch.Tensor = bounds.to(self._device, self._dtype)  # Input domain bounds
        self._optimization_problem_type = optimization_problem_type  # Type of optimization problem (minimization or maximization)
        self._acquisition_function_type = acquisition_function_type  # Type of acquisition function used for optimization
        self._acquisition_function_instance = None  # Instance of the acquisition function - instantiated within the optimization loop
        self._sampler_type = sampler_type  # Type of sampler used for initialization and acquisition function optimization
        self._sampler_instance = None  # Instance of the sampler - instantiated within the optimization loop
        self._true_objective = true_objective  # The ground truth (multi)objective function
        self._objective = objective  # The (multi)objective function to be optimized
        self._constraints = constraints  # The functional constraints
        self._n_acqf_opt_iter = n_acqf_opt_iter  # Number of iterations for acquisition function optimization
        self._max_n_acqf_opt_restarts = max_n_acqf_opt_restarts  # Max number of restarts for acquisition function optimization
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
        self._device = torch.device(device.value)

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

    def set_constraints(self, constraints: list[Callable] or None = None):
        validate_constraints(constraints)
        self._constraints = constraints

    def get_constraints(self):
        return self._constraints

    def add_constraint(self, constraint: Callable):
        validate_constraints([constraint, ])
        self._constraints.append(constraint)

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
        """ Define models for objectives and constraints. Note that the model is trained on
        normalized input features, while the outputs are not normalized. """

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
        """Prepare training data by combining objectives and constraints."""

        train_x = self._X

        # Combine objectives and constraints if they exist
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
                constraints=self._constraints
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogEHVI:
            self._acquisition_function_instance = qLogExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._constraints,
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qNEHVI:
            self._acquisition_function_instance = qNoisyExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                X_baseline=normalize(self._X, self._bounds),
                sampler=self._sampler_instance,
                prune_baseline=True,
                objective=self._objective,
                constraints=self._constraints,
            )

        elif self._acquisition_function_type == AcquisitionFunctionType.qLogNEHVI:
            self._acquisition_function_instance = qLogNoisyExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                X_baseline=normalize(self._X, self._bounds),
                prune_baseline=True,
                sampler=self._sampler_instance,
                objective=self._objective,
                constraints=self._constraints,
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

    def _compute_reference_point(self, verbose=True, buffer=0.25):

        # The reference point is calculated only for the first iteration step
        if self._iteration_number > 0:
            return

        if verbose:
            print("Defining reference point...", end="")

        # If the true objective is provided, then use its reference point if provided
        if self._true_objective is not None and hasattr(self._true_objective, "_ref_point"):
            self._ref_point = -torch.tensor(self._true_objective._ref_point).to(self._device)

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
            print(" Done.")

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

    def _compute_hypervolume(self, verbose=True, ):
        if verbose:
            print("Computing hypervolume...", end="")

        if self._pareto_front.shape[0] == 0:
            volume = 0
        else:
            hv = Hypervolume(self._ref_point)
            volume = hv.compute(self._pareto_front)
        self._hypervolume.append(volume)

        if verbose:
            print(" Done.")

    def _compute_pareto_front(self, verbose=True):
        """The pareto front is computed by considering only the objective observations for which the corresponding constraint
        observations satisfy the constraint functions. This is done by filtering out the acceptable Yobj observations."""

        if verbose:
            print("Finding Pareto front...", end="")

        if self._Ycon is None:
            # If the problem is unconstrained, then all observations can be Pareto-optimal

            self._par_mask = is_non_dominated(self._Yobj, maximize=self._optimization_problem_type.value)
            self._con_mask = torch.ones_like(self._par_mask, dtype=torch.bool).to(self._device, self._dtype)
            mask = self._par_mask

        else:
            # If the problem is constrained, then only the observations satisfying the constraints
            # can be Pareto-optimal. Therefore, first find the acceptable observations, i.e., those observations
            # satisfying the constraints and build a logical mask. # Then, find the acceptable observations that
            # are non-dominated and build a logical mask.

            Y = torch.cat([self._Yobj, self._Ycon], dim=-1)
            constraint_vals = [c(Y) for c in self._constraints]
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

        self._initialize_model(verbose=verbose)
        self._compute_reference_point(verbose=verbose)
        self._initialize_sampler(verbose=verbose)

        t0 = time.monotonic()
        self._fit_model(verbose=verbose)
        # self._initialize_partitioning(verbose=verbose)
        self._initialize_acquisition_function(verbose=verbose)
        self._optimize_acquisition_function(verbose=verbose)
        self._compute_pareto_front(verbose=verbose)
        self._compute_hypervolume(verbose=verbose)
        self._iteration_number += 1
        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)
        if self._device.type == "cuda":
            self._allocated_memory.append(torch.cuda.memory_allocated() / 1024 ** 2)

        if verbose:
            print(f"Calculation Time = {t1 - t0:>4.2f}\n"
                  f"Hypervolume = {self._hypervolume[-1]:>4.2f}\n"
                  f"New X: {self._new_X}")

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

    def load_dataset_from_csv(self, filepath: str or None = None, skipcol=0):
        if filepath is None:
            csv_files = list(Path('.').glob('*.csv'))
            if not csv_files:
                raise FileNotFoundError("No CSV files found in the current directory")
            filepath = max(csv_files, key=lambda x: x.stat().st_mtime)

        d = self._X.shape[-1]
        m = self._Yobj.shape[-1]
        c = self._Ycon.shape[-1] if self._Ycon is not None else 0

        xy = np.loadtxt(filepath, delimiter=",")
        X = xy[..., 0: d].copy()
        Yobj = xy[..., d: d + m].copy()
        Ycon = xy[..., d + m: d + m + c].copy()
        Yobj_var = xy[..., d + m + c: d + m + c + m].copy()
        Ycon_var = xy[..., d + m + c + m: d + m + c + m + c].copy()

        self.set_X(torch.Tensor(X))
        self.set_Yobj(torch.Tensor(Yobj))
        self.set_Yobj_var(torch.Tensor(Yobj_var))
        self.set_Ycon(torch.Tensor(Ycon))
        self.set_Ycon_var(torch.Tensor(Ycon_var))

    @classmethod
    def from_file(cls, filepath: str = None):
        """Load a MOBO instance from a file.
        If no filepath provided, loads the most recent .dat file from the current directory."""
        if filepath is None:
            files = glob.glob('*.dat')
            if not files:
                raise FileNotFoundError("No .dat files found in current directory")
            filepath = max(files, key=os.path.getctime)

        with open(filepath, 'rb') as f:
            return pickle.load(f)
