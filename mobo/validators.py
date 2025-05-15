import datetime

import torch

from collections.abc import Callable
from utils.types import AcquisitionFunctionType, OptimizationProblemType, SamplerType


def validate_experiment_name(name: str):
    if not isinstance(name, str):
        raise ValueError("Experiment name must be a string.")


def validate_X(X: torch.Tensor):
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch.Tensor.")


def validate_Yobj(Yobj: torch.Tensor):
    if not isinstance(Yobj, torch.Tensor):
        raise ValueError("Y must be a torch.Tensor.")


def validate_Ycon(Ycon: torch.Tensor | None):
    if Ycon is not None and not isinstance(Ycon, torch.Tensor):
        raise ValueError("Ycon must be None or a torch.tensor.")


def validate_Yobj_var(Yobj_var: torch.Tensor | None):
    if Yobj_var is not None and not isinstance(Yobj_var, torch.Tensor):
        raise ValueError("Yvar must be None or a torch.tensor.")


def validate_Ycon_var(Ycon_var: torch.Tensor | None):
    if Ycon_var is not None and not isinstance(Ycon_var, torch.Tensor):
        raise ValueError("Ycon_var must be None or a torch.tensor.")


def validate_bounds(bounds: torch.Tensor):
    if not isinstance(bounds, torch.Tensor):
        raise ValueError("Bounds must be a torch.tensor.")
    if bounds.shape[0] != 2:
        raise ValueError("Bounds must be a 2 x D tensor.")


def validate_objective(objective: Callable | None):
    if not isinstance(objective, Callable) and objective is not None:
        raise ValueError("Objective must be a callable or None.")


def validate_constraints(constraints: list[Callable] | None):
    if not isinstance(constraints, list) and constraints is not None:
        raise ValueError("Constraints must be a list or None.")
    if isinstance(constraints, list):
        for constraint in constraints:
            if not isinstance(constraint, Callable):
                raise ValueError("Constraints must be a list of callables.")


def validate_sampler_type(sampler_type: SamplerType):
    if not isinstance(sampler_type, SamplerType):
        raise ValueError("sampler_type must be a SamplerType.")


def validate_mc_samples(mc_samples: int):
    if not isinstance(mc_samples, int):
        raise ValueError("mc_samples must be an integer.")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be a positive integer. Recommended: 2^n.")


def validate_raw_samples(raw_samples: int):
    if not isinstance(raw_samples, int):
        raise ValueError("raw_samples must be an integer.")
    if raw_samples <= 0:
        raise ValueError("raw_samples must be a positive integer. Recommended: 2^n.")


def validate_batch_size(batch_size: int):
    if not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer. Recommended: 2^n.")


# TODO: understand what a true objective could be in BoTorch and amend the code
def validate_true_objective(true_objective: Callable | None):
    if not isinstance(true_objective, Callable) and true_objective is not None:
        raise ValueError("True objective must be a callable or None.")


def validate_datetime(date_time: datetime.datetime):
    if not isinstance(date_time, datetime.datetime):
        raise ValueError("Datetime must be a datetime.datetime object.")


def validate_acquisition_function(acquisition_function: AcquisitionFunctionType):
    if not isinstance(acquisition_function, AcquisitionFunctionType):
        raise ValueError(
            f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}."
        )


def validate_optimization_problem(optimization_problem_type: OptimizationProblemType):
    if not isinstance(optimization_problem_type, OptimizationProblemType):
        raise ValueError("optimization_problem_type must be a OptimizationProblemType.")


def validate_n_acqf_opt_iter(n_acqf_opt_iter: int):
    if not isinstance(n_acqf_opt_iter, int):
        raise ValueError("n_acqf_opt_iter must be an integer.")
    if n_acqf_opt_iter <= 0:
        raise ValueError("n_acqf_opt_iter must be a positive integer.")


def validate_max_n_acqf_opt_restarts(n_max_acqf_opt_restarts: int):
    if not isinstance(n_max_acqf_opt_restarts, int):
        raise ValueError("n_max_acqf_opt_restarts must be an integer.")
    if n_max_acqf_opt_restarts <= 0:
        raise ValueError("n_max_acqf_opt_restarts must be a positive integer.")
