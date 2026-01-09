from dataclasses import dataclass
from typing import Optional, List, Callable
from utils.bo_types import AcquisitionFunctionType
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


# === Acquisition Function Configs ===

@dataclass
class EIConfig:
    """Config for ExpectedImprovement variants."""
    pass


@dataclass
class PIConfig:
    """Config for ProbabilityOfImprovement variants."""
    pass


@dataclass
class UCBConfig:
    """Config for UpperConfidenceBound variants."""
    beta: float = 0.1


@dataclass
class NEIConfig:
    """Config for NoisyExpectedImprovement variants."""
    prune_baseline: bool = True


@dataclass
class EHVIConfig:
    """Config for ExpectedHypervolumeImprovement variants."""
    pass


@dataclass
class NEHVIConfig:
    """Config for NoisyExpectedHypervolumeImprovement variants."""
    prune_baseline: bool = True


@dataclass
class EWNEHVIConfig:
    """Config for ExplorationWeightedNEHVI."""
    prune_baseline: bool = True
    exploration_weight: float = 1.0


@dataclass
class DWNEHVIConfig:
    """Config for DiversityWeightedNEHVI."""
    prune_baseline: bool = True
    min_dist_radius: float = 1.0
    distance_penalty_weight: float = 1.0


@dataclass
class AcquisitionRuntimeParams:
    """Runtime parameters passed from optimizer."""
    model: any
    maximize: bool = False
    best_f: Optional[float] = None
    X_baseline: Optional[torch.Tensor] = None
    sampler: Optional[any] = None
    objective: Optional[any] = None
    ref_point: Optional[torch.Tensor] = None
    partitioning: Optional[any] = None
    constraints: Optional[List[Callable]] = None


class AcquisitionFunctionFactory:
    def __init__(self, acqf_type: AcquisitionFunctionType, config=None):
        self.acquisition_function_type = acqf_type
        self.config = config

    def requires_sampler(self) -> bool:
        return self.acquisition_function_type.requires_sampler()

    def __call__(self, params: AcquisitionRuntimeParams):

        # === Single-Objective: Analytical ===
        if self.acquisition_function_type == AcquisitionFunctionType.EI:
            return ExpectedImprovement(
                model=params.model,
                best_f=params.best_f,
                maximize=params.maximize,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.LogEI:
            return LogExpectedImprovement(
                model=params.model,
                best_f=params.best_f,
                maximize=params.maximize,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.PI:
            return ProbabilityOfImprovement(
                model=params.model,
                best_f=params.best_f,
                maximize=params.maximize,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.UCB:
            cfg = self.config or UCBConfig()
            return UpperConfidenceBound(
                model=params.model,
                beta=cfg.beta,
                maximize=params.maximize,
            )

        # === Single-Objective: Monte Carlo ===
        elif self.acquisition_function_type == AcquisitionFunctionType.qEI:
            return qExpectedImprovement(
                model=params.model,
                best_f=params.best_f,
                sampler=params.sampler,
                objective=params.objective,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qLogEI:
            return qLogExpectedImprovement(
                model=params.model,
                best_f=params.best_f,
                sampler=params.sampler,
                objective=params.objective,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qNEI:
            cfg = self.config or NEIConfig()
            return qNoisyExpectedImprovement(
                model=params.model,
                X_baseline=params.X_baseline,
                sampler=params.sampler,
                prune_baseline=cfg.prune_baseline,
                objective=params.objective,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qLogNEI:
            cfg = self.config or NEIConfig()
            return qLogNoisyExpectedImprovement(
                model=params.model,
                X_baseline=params.X_baseline,
                sampler=params.sampler,
                prune_baseline=cfg.prune_baseline,
                objective=params.objective,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qPI:
            return qProbabilityOfImprovement(
                model=params.model,
                best_f=params.best_f,
                sampler=params.sampler,
                objective=params.objective,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qUCB:
            cfg = self.config or UCBConfig()
            return qUpperConfidenceBound(
                model=params.model,
                beta=cfg.beta,
                sampler=params.sampler,
                objective=params.objective,
            )

        # === Multi-Objective: Monte Carlo ===
        elif self.acquisition_function_type == AcquisitionFunctionType.qEHVI:
            return qExpectedHypervolumeImprovement(
                model=params.model,
                ref_point=params.ref_point,
                partitioning=params.partitioning,
                sampler=params.sampler,
                objective=params.objective,
                constraints=params.constraints,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qLogEHVI:
            return qLogExpectedHypervolumeImprovement(
                model=params.model,
                ref_point=params.ref_point,
                partitioning=params.partitioning,
                sampler=params.sampler,
                objective=params.objective,
                constraints=params.constraints,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qNEHVI:
            cfg = self.config or NEHVIConfig()
            return qNoisyExpectedHypervolumeImprovement(
                model=params.model,
                ref_point=params.ref_point,
                X_baseline=params.X_baseline,
                sampler=params.sampler,
                prune_baseline=cfg.prune_baseline,
                objective=params.objective,
                constraints=params.constraints,
            )

        elif self.acquisition_function_type == AcquisitionFunctionType.qLogNEHVI:
            cfg = self.config or NEHVIConfig()
            return qLogNoisyExpectedHypervolumeImprovement(
                model=params.model,
                ref_point=params.ref_point,
                X_baseline=params.X_baseline,
                sampler=params.sampler,
                prune_baseline=cfg.prune_baseline,
                objective=params.objective,
                constraints=params.constraints,
            )

        # Note: qEWNEHVI, qDWNEHVI, qNParEGO would need their own imports and handling

        else:
            raise ValueError(f"Unsupported acquisition function type: {self.acquisition_function_type}")
