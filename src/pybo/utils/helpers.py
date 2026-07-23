import datetime
from pathlib import Path
import torch
from enum import Enum
from torch import Tensor, device, dtype


def serialize_value(value):
    """
    Recursively serializes a Python object into a JSON-compatible format.
    Handles:
      - datetime -> ISO string
      - torch.Tensor -> list
      - torch.device -> string
      - torch.dtype -> string
      - Enum -> value
      - basic types -> as-is
      - lists and dicts -> recursively
    Returns None for unsupported types.
    """
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, Tensor):
        return value.tolist()
    if isinstance(value, device):
        return str(value)
    if isinstance(value, dtype):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, list):
        serialized = [serialize_value(v) for v in value]
        return serialized if all(v is not None for v in serialized) else None
    if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
        serialized = {k: serialize_value(v) for k, v in value.items() if serialize_value(v) is not None}
        return serialized or None
    return None


def deserialize_value(value, target_type=None):
    """
    Recursively deserializes a JSON-compatible value back to its original type.

    Parameters:
        value: The value to deserialize (from JSON).
        target_type: Optional type hint for reconstruction (e.g., Tensor, Enum class)

    Returns:
        The deserialized value.
    """
    if value is None:
        return None

    # Handle types based on target_type
    if target_type is not None:
        if target_type is Tensor:
            return torch.tensor(value)
        if target_type is device:
            return torch.device(value)
        if target_type is dtype:
            return getattr(torch, value)
        if isinstance(target_type, type) and issubclass(target_type, Enum):
            return target_type(value)
        if target_type is datetime.datetime:
            return datetime.datetime.fromisoformat(value)

    # Automatic reconstruction if no target_type
    if isinstance(value, list):
        return [deserialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: deserialize_value(v) for k, v in value.items()}

    return value  # basic types remain as-is


def create_experiment_directory(main_directory: Path or str, experiment_name: str):
    if isinstance(main_directory, str):
        main_directory = Path(main_directory)
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = main_directory / Path(f'{date_time} - {experiment_name}')
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def project_linear_equalities(X: Tensor, lin_eq_cons) -> Tensor:
    # build stacked A,b (m x d), (m,)
    A_rows, b_rows = [], []
    d = X.shape[-1]
    for indices, coeffs, rhs in lin_eq_cons:
        a = torch.zeros(d, device=X.device, dtype=X.dtype)
        a[indices] = coeffs.to(dtype=X.dtype)
        A_rows.append(a)
        if torch.is_tensor(rhs):
            b_rows.append(rhs.to(dtype=X.dtype))
        else:
            b_rows.append(torch.tensor(rhs, device=X.device, dtype=X.dtype))
    A = torch.stack(A_rows, dim=0)  # (m, d)
    b = torch.stack(b_rows, dim=0).view(-1)  # (m,)

    # X_proj = X - A^T (A A^T)^{-1} (A X - b)
    # error: (n, m)
    error = X @ A.T - b
    # solve (A A^T) Y^T = error^T
    Y = torch.linalg.solve(A @ A.T, error.T).T  # (n, m)
    return X - Y @ A
