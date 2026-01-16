from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional, List
import torch
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
from botorch.acquisition.objective import PosteriorTransform
from bayesian_optimizer.optimizer import BayesianOptimizer


# =========================
# Base builder
# =========================

class AcquisitionFunctionBuilderBase(ABC):
    """Base class for acquisition function builders."""

    def __init__(
            self,
            require_sampler: bool,
            is_analytical: bool,
            runtime_params: Optional[Any] = None
    ):
        self._runtime_params = runtime_params
        self._require_sampler = require_sampler
        self._is_analytical = is_analytical

    @abstractmethod
    def build_acquisition_function_instance(self):
        ...

    def build_runtime_params(self, params: Any):
        self._runtime_params = params

    def build_runtime_params_from_bo(self, bo: BayesianOptimizer):
        """ Build runtime params from a BayesianOptimizer instance"""
        for field in fields(self._runtime_params):
            setattr(self._runtime_params, field.name, getattr(bo, field.name))


# =========================
# Single-objective: Analytical
# =========================

class EIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class RuntimeParams:
        model: Any = None
        best_f: float | torch.Tensor = None
        posterior_transform: PosteriorTransform = None
        maximize: bool = True

    def __init__(self, runtime_params: Optional[RuntimeParams] = None):
        super().__init__(require_sampler=False, is_analytical=True, runtime_params=runtime_params)

    def build_acquisition_function_instance(self):
        return ExpectedImprovement(
            model=self._runtime_params.model,
            best_f=self._runtime_params.best_f,
            posterior_transform=self._runtime_params.posterior_transform,
            maximize=self._runtime_params.maximize,
        )


class LogEIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class RuntimeParams:
        model: Any = None
        best_f: float | torch.Tensor = None
        posterior_transform: PosteriorTransform = None
        maximize: bool = True

    def __init__(self, runtime_params: Optional[RuntimeParams] = None):
        super().__init__(require_sampler=False, is_analytical=True, runtime_params=runtime_params)

    def build_acquisition_function_instance(self):
        return LogExpectedImprovement(
            model=self._runtime_params.model,
            best_f=self._runtime_params.best_f,
            posterior_transform=self._runtime_params.posterior_transform,
            maximize=self._runtime_params.maximize,
        )


class PIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class RuntimeParams:
        model: Any = None
        best_f: float | torch.Tensor = None
        posterior_transform: PosteriorTransform = None
        maximize: bool = True

    def __init__(self, runtime_params: Optional[RuntimeParams] = None):
        super().__init__(require_sampler=False, is_analytical=True, runtime_params=runtime_params)

    def build_acquisition_function_instance(self):
        return ProbabilityOfImprovement(
            model=self._runtime_params.model,
            best_f=self._runtime_params.best_f,
            posterior_transform=self._runtime_params.posterior_transform,
            maximize=self._runtime_params.maximize,
        )


class UCBBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class RuntimeParams:
        model: Any = None
        beta: float | torch.Tensor = None
        posterior_transform: PosteriorTransform = None
        maximize: bool = True

    def __init__(self, runtime_params: Optional[RuntimeParams] = None):
        super().__init__(require_sampler=False, is_analytical=True, runtime_params=runtime_params)

    def build_acquisition_function_instance(self):
        return UpperConfidenceBound(
            model=self._runtime_params.model,
            beta=self._runtime_params.beta,
            posterior_transform=self._runtime_params.posterior_transform,
            maximize=self._runtime_params.maximize,
        )


# =========================
# Single-objective: Monte Carlo
# =========================

class qEIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        best_f: float | torch.Tensor
        sampler: Any
        objective: Optional[Any] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qExpectedImprovement(
            model=self.runtime_params.model,
            best_f=self.runtime_params.best_f,
            sampler=self.runtime_params.sampler,
            objective=self.runtime_params.objective,
        )


class qLogEIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        best_f: float | torch.Tensor
        sampler: Any
        objective: Optional[Any] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qLogExpectedImprovement(
            model=self.runtime_params.model,
            best_f=self.runtime_params.best_f,
            sampler=self.runtime_params.sampler,
            objective=self.runtime_params.objective,
        )


class qPIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        best_f: float | torch.Tensor
        sampler: Any
        objective: Optional[Any] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qProbabilityOfImprovement(
            model=self.runtime_params.model,
            best_f=self.runtime_params.best_f,
            sampler=self.runtime_params.sampler,
            objective=self.runtime_params.objective,
        )


class qUCBBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        beta: float
        sampler: Any
        objective: Optional[Any] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qUpperConfidenceBound(
            model=self.runtime_params.model,
            beta=self.runtime_params.beta,
            sampler=self.runtime_params.sampler,
            objective=self.runtime_params.objective,
        )


class qNEIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        X_baseline: torch.Tensor
        sampler: Any
        prune_baseline: bool = True
        objective: Optional[Any] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qNoisyExpectedImprovement(
            model=self.runtime_params.model,
            X_baseline=self.runtime_params.X_baseline,
            sampler=self.runtime_params.sampler,
            prune_baseline=self.runtime_params.prune_baseline,
            objective=self.runtime_params.objective,
        )


class qLogNEIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        X_baseline: torch.Tensor
        sampler: Any
        prune_baseline: bool = True
        objective: Optional[Any] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qLogNoisyExpectedImprovement(
            model=self.runtime_params.model,
            X_baseline=self.runtime_params.X_baseline,
            sampler=self.runtime_params.sampler,
            prune_baseline=self.runtime_params.prune_baseline,
            objective=self.runtime_params.objective,
        )


# =========================
# Multi-objective: Monte Carlo
# =========================

class qEHVIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        ref_point: torch.Tensor
        partitioning: Any
        sampler: Any
        objective: Optional[Any] = None
        constraints: Optional[List[Callable]] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qExpectedHypervolumeImprovement(
            model=self.runtime_params.model,
            ref_point=self.runtime_params.ref_point,
            partitioning=self.runtime_params.partitioning,
            sampler=self.runtime_params.sampler,
            objective=self.runtime_params.objective,
            constraints=self.runtime_params.constraints,
        )


class qLogEHVIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        ref_point: torch.Tensor
        partitioning: Any
        sampler: Any
        objective: Optional[Any] = None
        constraints: Optional[List[Callable]] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qLogExpectedHypervolumeImprovement(
            model=self.runtime_params.model,
            ref_point=self.runtime_params.ref_point,
            partitioning=self.runtime_params.partitioning,
            sampler=self.runtime_params.sampler,
            objective=self.runtime_params.objective,
            constraints=self.runtime_params.constraints,
        )


class qNEHVIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        ref_point: torch.Tensor
        X_baseline: torch.Tensor
        sampler: Any
        prune_baseline: bool = True
        objective: Optional[Any] = None
        constraints: Optional[List[Callable]] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qNoisyExpectedHypervolumeImprovement(
            model=self.runtime_params.model,
            ref_point=self.runtime_params.ref_point,
            X_baseline=self.runtime_params.X_baseline,
            sampler=self.runtime_params.sampler,
            prune_baseline=self.runtime_params.prune_baseline,
            objective=self.runtime_params.objective,
            constraints=self.runtime_params.constraints,
        )


class qLogNEHVIBuilder(AcquisitionFunctionBuilderBase):
    @dataclass
    class Params:
        model: Any
        ref_point: torch.Tensor
        X_baseline: torch.Tensor
        sampler: Any
        prune_baseline: bool = True
        objective: Optional[Any] = None
        constraints: Optional[List[Callable]] = None

    def __init__(self):
        super().__init__(require_sampler=True, is_analytical=False)

    def build(self, runtime_params: Optional[Params] = None):
        super().build(runtime_params)
        return qLogNoisyExpectedHypervolumeImprovement(
            model=self.runtime_params.model,
            ref_point=self.runtime_params.ref_point,
            X_baseline=self.runtime_params.X_baseline,
            sampler=self.runtime_params.sampler,
            prune_baseline=self.runtime_params.prune_baseline,
            objective=self.runtime_params.objective,
            constraints=self.runtime_params.constraints,
        )
