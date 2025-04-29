import datetime
import pickle
import warnings
import time

import gpytorch.likelihoods
import numpy as np
import torch
import botorch
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling import SobolQMCNormalSampler
from botorch.sampling.normal import NormalMCSampler

from botorch.test_functions.base import MultiObjectiveTestProblem
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning, DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.transforms import unnormalize, normalize
from utils.cuda import get_device, get_supported_dtype
from utils.types import AcquisitionFunctionType, SamplerType
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement, \
    qLogExpectedHypervolumeImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from validators import *


class Mobo:

    def __init__(self, experiment_name: str):

        validate_experiment_name(experiment_name)

        self._experiment_name = experiment_name
        self._datetime = datetime.datetime.now()
        self._bounds: torch.Tensor | None = None
        self._acquisition_function_type: AcquisitionFunctionType | None = None
        self._sampler_type = None
        self._f0: MultiObjectiveTestProblem | None = None
        self._X: torch.Tensor | None = None
        self._Y: torch.Tensor | None = None
        self._Yvar: torch.Tensor | None = None
        self._n_acqf_opt_iter: int = 100
        self._max_n_model_fit_restart: int = 20
        self._max_n_acqf_opt_restarts: int = 20
        self._batch_size: int = 4
        self._MC_samples: int = 128

        # Device Attributes, filled at object instantiation
        self._device = get_device()
        self._dtype = get_supported_dtype(self._device)

        # Optimization State Attributes
        self._acquisition_function_instance: botorch.acquisition.AcquisitionFunction | None = None
        self._sampler_instance: NormalMCSampler | None = None
        self._model: ModelListGP | None = None
        self._mlls: list[ExactMarginalLogLikelihood] = []
        self._partitioning: NondominatedPartitioning = None
        self._pareto_front = None
        self._pareto_front_idx = None
        self._ref_point = None
        self._new_X = None  # The new X location
        self._hypervolume = []

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

    def set_X(self, X: torch.Tensor):
        validate_X(X)
        self._X = X.to(self._device, self._dtype)

    def get_X(self):
        return self._X

    def set_Y(self, Y: torch.Tensor):
        validate_Y(Y)
        self._Y = Y.to(self._device, self._dtype)

    def get_Y(self):
        return self._Y

    def set_Yvar(self, Yvar: torch.Tensor | None = None):
        validate_Yvar(Yvar)
        self._Yvar = Yvar.to(self._device, self._dtype) if Yvar is not None else None

    def get_Yvar(self):
        return self._Yvar

    def get_new_X(self):
        return self._new_X

    def set_bounds(self, bounds: torch.Tensor):
        validate_bounds(bounds)
        self._bounds = bounds.to(self._device, self._dtype)

    def get_bounds(self):
        return self._bounds

    def set_acquisition_function(self, acquisition_function: AcquisitionFunctionType):
        self._acquisition_function_type = acquisition_function

    def get_acquisition_function(self):
        return self._acquisition_function_type

    def set_sampler(self, sampler):
        self._sampler_type = sampler

    def set_batch_size(self, batch_size:int):
        """ Set the number of candidates to be generated in each optimization step."""
        self._batch_size = batch_size

    def get_batch_size(self):
        return self._batch_size

    def set_f0(self, f0: MultiObjectiveTestProblem):
        self._f0 = f0

    """ Optimizer """

    def _initialize_model(self):
        models = []
        for i in range(self._Y.shape[-1]):
            models.append(
                SingleTaskGP(
                    self._X,
                    self._Y[..., i: i + 1],
                    self._Yvar[..., i: i + 1] if self._Yvar is not None else None,
                    input_transform=Normalize(d=self._X.shape[-1]),
                    outcome_transform=Standardize(m=1),
                    likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-6))
                ).to(device=self._device, dtype=self._dtype)
            )
        # Aggregate the models into a ModelListGP
        self._model = ModelListGP(*models)
        # Extract the Marginal Likelihood for each model
        self._mll = SumMarginalLogLikelihood(self._model.likelihood, self._model)

    def _initialize_sampler(self):
        if self._sampler_type.name == SamplerType.Sobol.name:
            self._sampler_instance = SobolQMCNormalSampler(torch.Size([self._MC_samples]))

    def _initialize_acquisition_function(self):
        if self._acquisition_function_type.name == AcquisitionFunctionType.qEHVI.name:
            self._acquisition_function_instance = qExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
            )

        elif self._acquisition_function_type.name == AcquisitionFunctionType.qLogEHVI.name:
            self._acquisition_function_instance = qLogExpectedHypervolumeImprovement(
                model=self._model,
                ref_point=self._ref_point,
                partitioning=self._partitioning,
                sampler=self._sampler_instance,
            )

        elif self._acquisition_function_type.name == AcquisitionFunctionType.qNEHVI:
            if self._Yvar is None:
                raise ValueError("qNEHVI requires the observation noise variance (Yvar).")
            self._acquisition_function_instance(
                model=self._model,
                ref_point=self._ref_point,
                X_baseline=normalize(self._X, self._bounds),
                prune_baseline=True,
                sampler=self._sampler_instance,
            )

        else:
            raise ValueError(f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}.")

    def _initialize_partitioning(self):
        self._partitioning = FastNondominatedPartitioning(ref_point=self._ref_point, Y=self._Y)

    def _define_reference_point(self):
        self._ref_point = (torch.max(self._Y, dim=0).values + 1).to(device=self._device, dtype=self._dtype)

    def _fit_model(self, restart_on_error=True):
        if not isinstance(self._model, ModelListGP):
            raise ValueError("Model must be initialized before fitting.")

        restart_count = 0
        while True:
            try:
                botorch.fit_gpytorch_mll(self._mll)
                break  # Exit the inner loop on success
            except Exception as e:
                if restart_on_error and restart_count < self._max_n_model_fit_restart:
                    print(f"Restarting fitting... (Attempt {restart_count + 1}/{self._max_n_model_fit_restart})")
                    restart_count += 1
                else:
                    raise e  # Raise if not restarting or max restarts reached
        return None

    def _optimize_acquisition_function(self):
        self._new_X, _ = optimize_acqf(
            acq_function=self._acquisition_function_instance,
            bounds=self._bounds,
            q=self._batch_size,
            num_restarts=50,  # Number of times the optimizer is restarted if it fails to converge
            raw_samples=512,  # Number of samples for initialization
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        self._new_X = unnormalize(self._new_X.detach(), self._bounds)

    def _compute_hypervolume(self):
        bd = DominatedPartitioning(ref_point=self._ref_point, Y=self._Y)
        volume = bd.compute_hypervolume().item()
        self._hypervolume.append(volume)

    def optimize(self, verbose=True):

        warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        self._initialize_model()
        # self._initialize_partitioning_function_for_non_dominated_hypervolume()
        self._define_reference_point()
        # self._compute_dominated_hypervolume()
        self._initialize_partitioning()
        self._initialize_sampler()

        t0 = time.monotonic()
        self._fit_model()
        self._initialize_acquisition_function()
        self._optimize_acquisition_function()
        self._compute_hypervolume()

        t1 = time.monotonic()

        if verbose:
            print(f"Acq. Function: {self._acquisition_function_type.name}\n"
                  f"Hypervolume: {self._hypervolume[-1]:>4.2f}\n"
                  f"New X: {self._new_X}\n"
                  f"Calculation Time = {t1 - t0:>4.2f}.", end="",)
        else:
            print(".", end="")

    # def _initialize_partitioning_function_for_non_dominated_hypervolume(self):
    #     # partition non-dominated space into disjoint rectangles
    #     with torch.no_grad():
    #         pred = self._model.posterior(normalize(self._X, self._bounds)).mean
    #     self._partitioning = FastNondominatedPartitioning(ref_point=self._ref_point, Y=pred)





    """ I/O """

    def to_file(self):
        pickle.dump(self, open(f"./models/{self._experiment_name}.dat", "wb"))

    @classmethod
    def from_file(cls, filepath: str):
        """Load a MOBO instance from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
