import pickle
import warnings
import time
from collections.abc import Callable

import gpytorch.likelihoods
import botorch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.normal import NormalMCSampler

from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning, DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from utils.cuda import get_device, get_supported_dtype
from utils.types import SamplerType, OptimizationProblemType, TorchDeviceType
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement, \
    qLogExpectedHypervolumeImprovement, IdentityMCMultiOutputObjective
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from bayesian_optimization_for_gpu.validators import *
from utils.io import *


class Mobo:

    def __init__(
            self,
            experiment_name: str,
            n_objectives: int,
            n_constraints: int = 0
    ):

        validate_experiment_name(experiment_name)
        validate_n_objectives(n_objectives)
        validate_n_constraints(n_constraints)

        self._experiment_name = experiment_name
        self._n_objectives = n_objectives
        self._n_constraints= n_constraints
        self._datetime = datetime.datetime.now()
        self._bounds: torch.Tensor | None = None  # of shape "2 x D"
        self._acquisition_function_type: AcquisitionFunctionType | None = None
        self._sampler_type = None
        self._true_objective: MultiObjectiveTestProblem | None = None
        self._objective = None
        self._X: torch.Tensor | None = None
        self._Yobj: torch.Tensor | None = None
        self._Ycon: torch.Tensor | None = None
        self._Yobj_var: torch.Tensor | None = None
        self._Ycon_var: torch.Tensor | None = None

        self._n_acqf_opt_iter: int = 500
        self._max_n_acqf_opt_restarts: int = 10
        self._batch_size: int = 1  # Number of candidates to be generated in parallel in each optimization step
        self._MC_samples: int = 1024  # Number of samples for initialization and acquisition function optimization
        self._raw_samples: int = 256  # Number of samples for acquisition function optimization

        # Device Attributes, filled at object instantiation
        self._device = get_device()
        self._dtype = get_supported_dtype(self._device)

        # Optimization State Attributes
        self._acquisition_function_instance: botorch.acquisition.AcquisitionFunction | None = None
        self._sampler_instance: NormalMCSampler | None = None
        self._model_obj: ModelListGP | None = None
        self._model_con: ModelListGP | None = None
        self._mlls: list[ExactMarginalLogLikelihood] = []
        self._partitioning: NondominatedPartitioning | None = None
        self._pareto_front = None
        self._optimization_problem_type = None
        self._constraints = None
        self._ref_point = None
        self._new_X = None  # The new X location
        self._hypervolume = []

    """ Setters and getters """

    def set_experiment_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        self._experiment_name = name

    def get_experiment_name(self):
        return self._experiment_name

    def set_n_objectives(self, n_objectives: int):
        validate_n_objectives(n_objectives)
        self._n_objectives = n_objectives

    def get_n_objectives(self):
        return self._n_objectives

    def set_n_constraints(self, n_constraints: int):
        validate_n_constraints(n_constraints)
        self._n_constraints = n_constraints

    def get_n_constraints(self):
        return self._n_constraints

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
        validate_Y(Yobj)
        self._Yobj = Yobj.to(self._device, self._dtype)

    def get_Yobj(self) -> torch.Tensor | None:
        return self._Yobj

    def set_Yobj_var(self, Yobj_var: torch.Tensor | None = None):
        validate_Y_var(Yobj_var)
        self._Yobj_var = Yobj_var.to(self._device, self._dtype) if Yobj_var is not None else None

    def get_Yobj_var(self):
        return self._Yobj_var

    def set_Ycon(self, Ycon: torch.Tensor):
        validate_Y(Ycon)
        self._Ycon = Ycon.to(self._device, self._dtype)

    def get_Ycon(self):
        return self._Ycon

    def set_Ycon_var(self, Ycon_var: torch.Tensor | None = None):
        validate_Y_var(Ycon_var)
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

    def get_model(self):
        return self._model_obj

    def set_acquisition_function(self, acquisition_function_type: AcquisitionFunctionType):
        self._acquisition_function_type = acquisition_function_type

    def get_acquisition_function(self):
        return self._acquisition_function_type

    def set_optimization_problem(self, optimization_problem_type: OptimizationProblemType):
        self._optimization_problem_type = optimization_problem_type

    def get_optimization_problem_type(self):
        return self._optimization_problem_type

    def set_sampler_type(self, sampler):
        self._sampler_type = sampler

    def set_batch_size(self, batch_size:int):
        """ Set the number of candidates to be generated in each optimization step."""
        self._batch_size = batch_size

    def get_batch_size(self):
        return self._batch_size

    def set_MC_samples(self, MC_samples:int):
        self._MC_samples = MC_samples

    def get_MC_samples(self):
        return self._MC_samples

    def set_raw_samples(self, raw_samples:int):
        self._raw_samples = raw_samples

    def get_raw_samples(self):
        return self._raw_samples

    def set_true_objective(self, f0: MultiObjectiveTestProblem):
        self._true_objective = f0

    def get_true_objective(self):
        return self._true_objective

    def set_objective(self, objective: Callable):
        self._objective = objective

    def get_objective(self):
        return self._objective

    def get_pareto(self):
        return self._pareto

    def set_constraints(self, constraints: list[Callable]):
        self._constraints = constraints

    def get_constraints(self):
        return self._constraints

    def add_constraint(self, constraint: Callable):
        if self._constraints is None:
            self._constraints = []
        self._constraints.append(constraint)

    """ Optimizer """

    def _initialize_model(self, verbose=True):
        """ Build separate models for the objective and constraint functions."""

        if verbose:
            print("Initializing model...", end="")

        models_obj = []
        for i in range(0, self._Yobj.shape[-1]):
            models_obj.append(
                SingleTaskGP(
                    self._X,
                    self._Yobj[..., i: i + 1],
                    self._Yobj_var[..., i: i + 1] if self._Yobj_var is not None else None,
                    input_transform=Normalize(d=self._X.shape[-1]),
                    outcome_transform=Standardize(m=1),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-4))
                ).to(device=self._device, dtype=self._dtype)
            )

        models_con = []
        for i in range(0, self._Ycon.shape[-1]):
            models_con.append(
                SingleTaskGP(
                    self._X,
                    self._Ycon[..., i: i + 1],
                    self._Ycon_var[..., i: i + 1] if self._Ycon_var is not None else None,
                    input_transform=Normalize(d=self._X.shape[-1]),
                    outcome_transform=Standardize(m=1),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
                ).to(device=self._device, dtype=self._dtype)
            )

        # Aggregate the models into a ModelListGP
        self._model_obj = ModelListGP(*models_obj)
        self._model_con = ModelListGP(*models_con)
        self._model = ModelListGP(*(models_obj + models_con))
        # Extract the Marginal Likelihood for each model
        self._mll_obj = SumMarginalLogLikelihood(self._model_obj.likelihood, self._model_obj)
        self._mll_con = SumMarginalLogLikelihood(self._model_con.likelihood, self._model_con)
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

        if verbose:
            print(" Done.")

    def _initialize_sampler(self, verbose=True,):
        if verbose:
            print("Initializing sampler...", end="")

        if self._sampler_type.name == SamplerType.Sobol.name:
            self._sampler_instance = SobolQMCNormalSampler(torch.Size([self._MC_samples]))

        if verbose:
            print(" Done.")

    def _initialize_acquisition_function(self, verbose=True,):
        if verbose:
            print("Initializing acquisition function...", end="")

        if self._acquisition_function_type.name == AcquisitionFunctionType.qEHVI.name:
            self._acquisition_function_instance = qExpectedHypervolumeImprovement(
                model=self._model_obj,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                constraints=self._constraints
            )

        elif self._acquisition_function_type.name == AcquisitionFunctionType.qLogEHVI.name:
            self._acquisition_function_instance = qLogExpectedHypervolumeImprovement(
                model=self._model_obj,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
                objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
                constraints=self._constraints,
            )

        elif self._acquisition_function_type.name == AcquisitionFunctionType.qNEHVI:
            if self._Yobj_var is None:
                raise ValueError("qNEHVI requires the observation noise variance (Yvar).")
            self._acquisition_function_instance = qNoisyExpectedHypervolumeImprovement(
                model=self._model_obj,
                ref_point=self._ref_point,
                X_baseline=normalize(self._X, self._bounds),
                prune_baseline=True,
                sampler=self._sampler_instance,
            )

        else:
            raise ValueError(f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}.")

        if verbose:
            print(" Done.")

    def _initialize_partitioning(self, verbose=True,):
        if verbose:
            print("Initializing partitioning function...", end="")

        self._partitioning = FastNondominatedPartitioning(ref_point=self._ref_point, Y=self._Yobj)

        if verbose:
            print(" Done.")

    def _define_reference_point(self, verbose=True, buffer=0.1):
        if verbose:
            print("Defining reference point...", end="")

        if self._optimization_problem_type == OptimizationProblemType.Maximization:
            worst = torch.min(self._Yobj, dim=0).values
            self._ref_point = (worst - buffer * torch.abs(worst)).to(self._device)
        elif self._optimization_problem_type == OptimizationProblemType.Minimization:
            worst = torch.max(self._Yobj, dim=0).values
            self._ref_point = (worst + buffer * torch.abs(worst)).to(self._device)

        if verbose:
            print(" Done.")

    def _fit_model(self, restart_on_error=True, verbose=True,):
        if not isinstance(self._model_obj, ModelListGP):
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

    def _optimize_acquisition_function(self, verbose=True,):
        if verbose:
            print("Optimizing acquisition function...", end="")

        self._new_X, _ = optimize_acqf(
            acq_function=self._acquisition_function_instance,
            bounds=self._bounds,
            q=self._batch_size,
            num_restarts=self._max_n_acqf_opt_restarts,
            raw_samples=self._raw_samples,  # n x q x dim
            options={"maxiter": self._n_acqf_opt_iter, "disp": True},
            sequential=True,
        )
        self._new_X = unnormalize(self._new_X.detach(), self._bounds)

        if verbose:
            print(" Done.")

    def _compute_hypervolume(self, verbose=True,):
        if verbose:
            print("Computing hypervolume...", end="")

        bd = DominatedPartitioning(ref_point=self._ref_point, Y=self._Yobj)
        volume = bd.compute_hypervolume().item()
        self._hypervolume.append(volume)

        if verbose:
            print(" Done.")

    def _find_pareto(self, verbose=True):
        if verbose:
            print("Finding Pareto front...", end="")

        if self._optimization_problem_type == OptimizationProblemType.Maximization:
            self._pareto = is_non_dominated(self._Yobj, maximize=True)
        elif self._optimization_problem_type == OptimizationProblemType.Minimization:
            self._pareto = is_non_dominated(self._Yobj, maximize=False)

        if verbose:
            print(" Done.")

    def optimize(self, verbose=True):

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self._initialize_model(verbose=verbose)
        self._define_reference_point(verbose=verbose)
        self._initialize_partitioning(verbose=verbose)
        self._initialize_sampler(verbose=verbose)

        t0 = time.monotonic()
        self._fit_model(verbose=verbose)
        self._initialize_acquisition_function(verbose=verbose)
        self._optimize_acquisition_function(verbose=verbose)
        self._find_pareto(verbose=verbose)
        self._compute_hypervolume(verbose=verbose)
        t1 = time.monotonic()

        if verbose:
            print(f"Calculation Time = {t1 - t0:>4.2f}\n"
                  f"New X: {self._new_X}\n\n")

    """ I/O """
    
    def to_file(self):
        filepath = compose_model_filename()
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
        return filepath

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
