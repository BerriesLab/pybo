from __future__ import annotations

import inspect
from abc import ABC
from dataclasses import fields, make_dataclass
from typing import TYPE_CHECKING, Type, Any

from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qLogExpectedImprovement,
    qNoisyExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.multi_objective import (
    qExpectedHypervolumeImprovement,
    qLogExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement,
)

if TYPE_CHECKING:
    from bayesian_optimizer.optimizer import BayesianOptimizer


# =========================
# Base builder
# =========================

class AcquisitionFunctionBuilderBase(ABC):
    """ Base class that automatically generates a RuntimeParams dataclass
    by inspecting the constructor of the target acquisition function class."""
    _acq_class: Type = None

    def __init__(self, require_sampler: bool, is_analytical: bool, is_log: bool):
        self._require_sampler = require_sampler
        self._is_analytical = is_analytical
        self._is_log = is_log
        self._runtime_params = None

        self._generate_runtime_params_class()

    def _generate_runtime_params_class(self):
        """Extracts the parameters from the constructor of the target class."""
        if self._acq_class is None:
            return

        sig = inspect.signature(self._acq_class.__init__)
        params = []

        for name, param in sig.parameters.items():
            if name in ('self', 'args', 'kwargs'):
                continue

            # Default to None if no default value is provided in the signature
            default_value = param.default if param.default is not inspect.Parameter.empty else None
            params.append((name, Any, default_value))

        # Dynamically create the dataclass
        self.RuntimeParams = make_dataclass(f"{self._acq_class.__name__}Params", params)

    def build_runtime_params_from_bo(self, bo: BayesianOptimizer):
        """Populates the dynamic dataclass with attributes from the Bayesian Optimizer."""
        self._runtime_params = self.RuntimeParams()

        for f in fields(self._runtime_params):
            if hasattr(bo, f.name):
                val = getattr(bo, f.name)
                setattr(self._runtime_params, f.name, val)

    def build_acquisition_function_instance(self):
        """ Generic implementation that instantiates the acquisition function
        using the stored runtime parameters. """
        if self._runtime_params is None:
            raise ValueError("Runtime parameters have not been built yet.")

        params_dict = {f.name: getattr(self._runtime_params, f.name)
                       for f in fields(self._runtime_params)}
        return self._acq_class(**params_dict)

    @property
    def runtime_params(self):
        return self._runtime_params

    @property
    def require_sampler(self) -> bool:
        return self._require_sampler

    @property
    def is_analytical(self) -> bool:
        return self._is_analytical

    @property
    def is_log(self) -> bool:
        return self._is_log


# ==========================================================
# Single-objective: Analytical (require_sampler=False)
# ==========================================================

class EIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = ExpectedImprovement

    def __init__(self):
        super().__init__(require_sampler=False, is_analytical=True, is_log=False)


class LogEIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = LogExpectedImprovement

    def __init__(self):
        super().__init__(require_sampler=False, is_analytical=True, is_log=True)


class PIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = ProbabilityOfImprovement

    def __init__(self):
        super().__init__(require_sampler=False, is_analytical=True, is_log=False)


class UCBBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = UpperConfidenceBound

    def __init__(self):
        super().__init__(require_sampler=False, is_analytical=True, is_log=False)


# ==========================================================
# Single-objective: Monte Carlo (require_sampler=True)
# ==========================================================

class qEIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qExpectedImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=False)


class qLogEIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qLogExpectedImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=True)


class qPIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qProbabilityOfImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=False)


class qUCBBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qUpperConfidenceBound

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=False)


class qNEIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qNoisyExpectedImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=False)


class qLogNEIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qLogNoisyExpectedImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=True)


# ==========================================================
# Multi-objective: Monte Carlo (require_sampler=True)
# ==========================================================

class qEHVIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qExpectedHypervolumeImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=False)


class qLogEHVIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qLogExpectedHypervolumeImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=True)


class qNEHVIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qNoisyExpectedHypervolumeImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=False)


class qLogNEHVIBuilder(AcquisitionFunctionBuilderBase):
    _acq_class = qLogNoisyExpectedHypervolumeImprovement

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False, is_log=True)
