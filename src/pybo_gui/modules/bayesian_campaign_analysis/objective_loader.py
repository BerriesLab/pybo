"""Load a problem definition from an objective module on disk.

The per-step records name their values but carry no problem definition, so the senses,
bounds and reference point a campaign plot needs have to come from somewhere. They come
from the objective itself: point this at the ``objective.py`` a run used and it returns
what the plots ask for, rather than making the caller retype it into the GUI.
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


def problem_definition(objective: MCObjectiveBase) -> dict:
    """The parts of an objective a campaign plot needs, as plain data.

    Keyed by label throughout, so it lines up with the columns `steps.step_frame`
    produces without going back through tensor positions.
    """
    objectives = [
        {"label": cfg.label,
         "to_minimize": bool(cfg.to_minimize),
         "ref_point": getattr(cfg, "ref_point", None),
         "bounds": list(cfg.bounds) if getattr(cfg, "bounds", None) is not None else None}
        for cfg in objective.obj_cfg or []
    ]
    return {
        "parameters": [{"label": cfg.label, "bounds": list(cfg.bounds)}
                       for cfg in objective.par_cfg or []],
        "objectives": objectives,
        "constraints": [{"label": cfg.label} for cfg in objective.ineq_Y_con_cfg or []],
        "trackers": [{"label": cfg.label} for cfg in objective.trk_cfg or []],
        # Where the hypervolume is measured from, in the objectives' own order.
        "ref_point": [o["ref_point"] for o in objectives],
        "minimized": {o["label"]: o["to_minimize"] for o in objectives},
        "max_hv": getattr(objective, "max_hv", None),
        "best_value": getattr(objective, "best_value", None),
    }