import datetime

import torch
from utils.types import AcquisitionFunctionType


def validate_experiment_name(name: str):
    if not isinstance(name, str):
        raise ValueError("Experiment name must be a string.")

def validate_datetime(date_time: datetime.datetime):
    if not isinstance(date_time, datetime.datetime):
        raise ValueError("Datetime must be a datetime.datetime object.")

def validate_X(X: torch.Tensor):
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch.Tensor.")


def validate_Y(Y: torch.Tensor):
    if not isinstance(Y, torch.Tensor):
        raise ValueError("Y must be a torch.Tensor.")


def validate_Yvar(Yvar: torch.Tensor):
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