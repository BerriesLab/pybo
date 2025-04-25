import abc

import torch
import botorch
import gpytorch
from typing import Callable, Optional, Union

from bayesian_optimization_for_cpu.enums import Kernel
from bayesian_optimization_for_gpu.enums import AcquisitionFunction


def validate_experiment_name(name: str):
    if not isinstance(name, str):
        raise ValueError("Experiment name must be a string.")


def validate_X(X: torch.Tensor):
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch.tensor.")


def validate_Y(Y: torch.Tensor):
    if not isinstance(Y, torch.Tensor):
        raise ValueError("Y must be a torch.tensor.")


def validate_Yvar(Yvar: torch.Tensor):
    if Yvar is not None and not isinstance(Yvar, torch.Tensor):
        raise ValueError("Yvar must be None or a torch.tensor.")


def validate_bounds(bounds: list[tuple[float, float]]):
    if not isinstance(bounds, list):
        raise ValueError("bounds must be a list.")
    for bound in bounds:
        if not isinstance(bound, tuple):
            raise ValueError("Each bound must be a tuple.")
        if len(bound) != 2:
            raise ValueError("Each bound must be a tuple of length 2.")
        if bound[0] >= bound[1]:
            raise ValueError("Each bound must be a tuple of (lower, upper) where lower < upper.")
