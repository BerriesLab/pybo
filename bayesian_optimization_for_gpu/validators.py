import datetime

import torch
from utils.types import AcquisitionFunctionType, OptimizationProblemType


def validate_experiment_name(name: str):
    if not isinstance(name, str):
        raise ValueError("Experiment name must be a string.")

def validate_n_objectives(n_objectives: int):
    if not isinstance(n_objectives, int):
        raise ValueError("Number of objectives must be an integer.")
    if n_objectives < 1:
        raise ValueError("Number of objectives must be at least 1.")

def validate_n_constraints(n_constraints: int):
    if not isinstance(n_constraints, int):
        raise ValueError("Number of constraints must be an integer.")
    if n_constraints < 0:
        raise ValueError("Number of constraints must be at least 0.")

def validate_datetime(date_time: datetime.datetime):
    if not isinstance(date_time, datetime.datetime):
        raise ValueError("Datetime must be a datetime.datetime object.")

def validate_X(X: torch.Tensor):
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch.Tensor.")


def validate_Y(Y: torch.Tensor):
    if not isinstance(Y, torch.Tensor):
        raise ValueError("Y must be a torch.Tensor.")


def validate_Y_var(Yvar: torch.Tensor):
    if Yvar is not None and not isinstance(Yvar, torch.Tensor):
        raise ValueError("Yvar must be None or a torch.tensor.")


def validate_acquisition_function(acquisition_function: AcquisitionFunctionType):
    if not isinstance(acquisition_function, AcquisitionFunctionType):
        raise ValueError(
            f"Invalid acquisition function. Supported values are {AcquisitionFunctionType.values()}."
        )

def validate_bounds(bounds: torch.Tensor):
    if not isinstance(bounds, torch.Tensor):
        raise ValueError("Bounds must be a torch.tensor.")
    if bounds.shape[0] != 2:
        raise ValueError("Bounds must be a 2 x D tensor.")


def validate_optimization_problem(optimization_problem_type: OptimizationProblemType):
    if not isinstance(optimization_problem_type, OptimizationProblemType):
        raise ValueError("optimization_problem_type must be a OptimizationProblemType.")