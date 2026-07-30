import torch


def _set_tensor(
        value: torch.Tensor | None,
        device: torch.device = None,
        dtype: torch.dtype = None,
) -> torch.Tensor | None:
    if value is None:
        return None

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Must be a torch.Tensor or None.")

    return value.to(device=device, dtype=dtype)


def _check_tensor(
        value: torch.Tensor,
        *,
        name: str,
        last_dim: int,
        n_rows: int | None = None,
) -> None:
    """Check a tensor's width against the objective, and its length against X.

    The setters only coerce, so this is where a tensor is held to the problem it
    describes. Called from OptimizerBase._validate_state() once per run step, on
    tensors already known to be present."""

    if value.shape[-1] != last_dim:
        raise ValueError(
            f"{name} must have {last_dim} columns, got {value.shape[-1]}."
        )

    if n_rows is not None and value.shape[0] != n_rows:
        raise ValueError(
            f"{name} must have one row per observation ({n_rows}), "
            f"got {value.shape[0]}."
        )
