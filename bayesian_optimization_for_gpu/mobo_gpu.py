import numpy as np
import torch
import botorch

from botorch.acquisition import AcquisitionFunction
from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.test_functions.multi_objective import BraninCurrin
from pyparsing import Empty
from bayesian_optimization_for_gpu.utils import get_device, get_supported_dtype
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from utils import *
from enums import AcquisitionFunction
from validators import *
from datetime import datetime


class Mobo:

    def __init__(self,
                 experiment_name: str,
                 bounds: list[tuple[float, float]],
                 acquisition_function: botorch.acquisition.multi_objective.MultiObjectiveMCAcquisitionFunction,
                 f0: MultiObjectiveTestProblem | None,
                 kernel: str,
                 X: torch.Tensor,  # (n samples x N dimensions)
                 Y: torch.Tensor,  # (n samples x M objectives)
                 Yvar: torch.Tensor | None = None,  # (n samples x M objectives
                 n_acqf_opt_iter: int = 100,
                 max_n_fit_restart: int = 20,
                 ):

        # Validate passed arguments
        validate_experiment_name(experiment_name)
        validate_bounds(bounds)
        validate_X(X)
        validate_Y(Y)
        validate_Yvar(Yvar)

        # Attributes required at initialization
        self._experiment_name = experiment_name
        self._datetime = datetime.now()
        self._bounds = bounds
        self._acquisition_function = acquisition_function.value
        self._f0 = f0
        self._kernel = kernel.value
        self._X = X
        self._Y = Y
        self._Yvar = Yvar
        self._n_acqf_opt_iter = n_acqf_opt_iter
        self._max_n_fit_restart = max_n_fit_restart

        # Device Attributes, filled at object instantiation
        self._device = get_device()
        self._dtype = get_supported_dtype()

        # Optimization State Attributes
        self._domain: np.ndarray | None = None
        self._models: list[SingleTaskGP] = []
        self._mlls: list[ExactMarginalLogLikelihood] = []
        self._partitioning: NondominatedPartitioning = None
        self._pareto_front = None
        self._pareto_front_idx = None
        self._ref_point = None
        self._new_X = None  # The new X location

        # Plotting Attributes
        # self._fig = None
        # self._ax = None
        # self._mu: list[np.ndarray] = [np.zeros(1) for _ in range(n_objectives)]
        # self._sigma: list[np.ndarray] = [np.zeros(1) for _ in range(n_objectives)]

    """ Setters and getters """

    def set_experiment_name(self, name: str):
        if not isinstance(name, str):
            raise ValueError("Experiment name must be a string.")
        self._experiment_name = name

    def get_experiment_name(self):
        return self._experiment_name

    def set_X(self, X: torch.Tensor | None = None):
        if not isinstance(X, torch.Tensor) or X is not None:
            raise ValueError("X must be None or a torch.Tensor.")
        self._X = X

    def get_X(self):
        return self._X

    def set_Y(self, Y: torch.Tensor | None = None):
        if not isinstance(Y, torch.Tensor) or Y is not None:
            raise ValueError("Y must be None or a torch.Tensor.")
        self._Y = Y

    def get_Y(self):
        return self._Y

    def set_Yvar(self, Yvar: torch.Tensor | None = None):
        if not isinstance(Yvar, torch.Tensor) or Yvar is not None:
            raise ValueError("Yvar must be None or a torch.Tensor.")
        self._Yvar = Yvar

    def get_Yvar(self):
        return self._Yvar

    def get_new_X(self):
        return self._new_X

    # TODO: fix setters and getters

    def set_bounds(self, bounds):
        self._validate_bounds(bounds)
        self._bounds = bounds

    def get_bounds(self):
        return self._bounds

    def get_number_of_objective_function_calls(self):
        return self._n_iter

    """ Optimizer """

    def _initialize_models(self):
        """Define the GP model."""
        # We create a list of models, one for each objective.
        # We standardize the output to improve GP training.
        for i in range(self._Y.shape[-1]):
            self._models.append(
                SingleTaskGP(
                    self._X,
                    self._Y[..., i: i + 1],
                    self._Yvar[..., i: i + 1] if self._Yvar is not None else None,
                    input_transform=Normalize(d=self._X.shape[-1]),
                    outcome_transform=Standardize(m=1),
                )
            )

    def _fit_models(self, restart_on_error=True):
        if not self._models:
            raise ValueError("Models must be initialized before fitting.")
        # Use ExactMarginalLogLikelihood to train the hyperparameters of the GP.
        restart_count = 0
        for i in range(self._Y.shape[-1]):
            mll = ExactMarginalLogLikelihood(self._models[i].likelihood, self._models[i])
            # TODO: if it fails at every re-start, enters into infinite loop - add breakin logic
            # Fit the models – Include restarting logic
            try:
                botorch.fit_gpytorch_mll(mll)
                return None
            except Exception as e:
                print(f"Error fitting model: {e}")
                if restart_on_error and restart_count < self._max_n_fit_restart:
                    print(f"Restarting fitting... (Attempt {restart_count + 1}/{self._max_n_fit_restart})")
                    restart_count += 1
                    return self._fit_models(restart_on_error=True)
                else:
                    raise e  # re-raise if not restarting or max restarts reached
        return None

    def _partitioning(self):
        #TODO: add partitioning

    def _define_acquisition_function(self):
            # Use qEHVI for batch optimization.  For single-point optimization, use qEHVI.
            # The ref_point is crucial for EHVI.  It defines the lower bound of the region
            # in objective space that we want to improve upon.  It should be set such that
            # it is weakly dominated by all points in the Pareto front.
            self._acquisition_function = qNoisyExpectedHypervolumeImprovement(
                model=self._models,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                # Define an objective that specifies how the model outputs should be transformed
                # to obtain the objective values.
                # TODO: understand this transformation
                objective=IdentityMCMultiOutputObjective,  # No transformation in this case.
            )


    def _define_reference_point(self):
        # Define the reference point.  This is critical for EHVI.  Choose a point
        # that is weakly dominated by all Pareto optimal solutions.  A conservative
        # choice is often slightly worse than the worst observed values for each objective.
        self._ref_point = torch.min(self._Y, dim=0).values - 1  # make it a bit worse

    def _calculate_pareto_front(self):
        self._partitioning = NondominatedPartitioning(ref_point=self._ref_point, Y=self._Y)

    def optimize(self, q=1, restart_on_error=True):
        self._initialize_models()
        self._fit_models()
        self._define_reference_point()
        self._define_acquisition_function()

        self._calculate_pareto_front()
        self._define_acquisition_function()

    def _optimize_acquisition_function(self, acq_func, bounds, n_opt_iter, acqf_n_opt_iter, restart_on_error=True):

        # Optimize the acquisition function to find the next point(s) to evaluate.
        try:
            self._new_X, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=q,  # Number of candidates to generate
                num_restarts=10,  # Increased restarts
                raw_samples=512,  # Increased raw samples
                options={"local_optimizer_options": {"maxiter": acqf_n_opt_iter}},  # Increased iterations
            )
        except BadInitialCandidatesError:
            print(
                "Optimization of acquisition function failed due to BadInitialCandidatesError. "
                "Restarting optimization loop from the beginning..."
            )
            return optimize_problem(
                num_iterations=n_opt_iter,
                q=q,
                acqf_opt_iters=acqf_n_opt_iter,
                restart_on_error=restart_on_error,
            )  # Restart

        return train_x, train_obj, models, bounds
