import torch


def _set_tensor(value: torch.Tensor | None, device: torch.device = None, dtype: torch.dtype = None,
                ndim: int | None = None) -> torch.Tensor | None:
    # Check if the value is None
    if value is None:
        return None
    # if it is not None, check if it is a torch.Tensor.
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Must be a torch.Tensor or None.")
    # If it is a torch.Tensor, check whether its dimensions match
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"ndim = {value.ndim} != {ndim} (must be n x d)")
    # if it is a Tensor and its dimensions match, return a tensor on hardware
    return value.to(device=device, dtype=dtype)


def _check_tensor(value: torch.Tensor, *, name: str, last_dim: int, n_rows: int | None = None) -> None:
    """Check a tensor's width against the objective, and its length against X."""

    if value.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have {last_dim} columns, got {value.shape[-1]}."
        )

    if n_rows is not None and value.shape[0] != n_rows:
        raise ValueError(
            f"{name} must have one row per observation ({n_rows}), "
            f"got {value.shape[0]}."
        )
