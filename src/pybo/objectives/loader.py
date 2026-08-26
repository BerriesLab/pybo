"""Load an objective instance from a module on disk.

Anything that only knows a path to an ``objective.py`` - a GUI, a ground-truth fit, a
plot script - needs to get from that path to a live objective the same way, so the
import machinery lives here once rather than once per caller.
"""
import importlib.util
import inspect
from pathlib import Path

import torch

from pybo.objectives.base_class import MCObjectiveBase


def load_objective(path: str | Path, device: torch.device | None = None,
                   dtype: torch.dtype = torch.float64) -> MCObjectiveBase:
    """Import `path` and instantiate the objective it defines.

    The module is loaded under its own name so a tutorial's relative imports still
    resolve. A file defining several objectives is ambiguous, so it is rejected rather
    than guessed at.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path} is not an importable Python module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    found = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
             if issubclass(obj, MCObjectiveBase) and obj.__module__ == module.__name__
             and not inspect.isabstract(obj)]
    if not found:
        raise SystemExit(f"{path} defines no concrete objective (a MCObjectiveBase subclass).")
    if len(found) > 1:
        names = ", ".join(o.__name__ for o in found)
        raise SystemExit(f"{path} defines several objectives ({names}); it is ambiguous "
                         f"which one the run used.")
    return found[0](device=device or torch.device("cpu"), dtype=dtype)
