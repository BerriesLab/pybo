"""Load a problem definition from an objective module on disk.

The per-step records name their values but carry no problem definition, so the senses,
bounds and reference point a campaign plot needs have to come from somewhere. They come
from the objective itself: point this at the ``objective.py`` a run used and it returns
what the plots ask for, rather than making the caller retype it into the GUI.
"""
from pybo.objectives.base_class import MCObjectiveBase
# The import itself has nothing GUI-specific in it - it moved to core pybo so
# build_polynomial_gt.py (which stays runnable from a terminal without pybo_gui
# installed) can use it too. Re-exported here so every existing import of this module
# keeps working unchanged.
from pybo.objectives.loader import load_objective  # noqa: F401


def problem_definition(objective: MCObjectiveBase) -> dict:
    """The parts of an objective a campaign plot needs, as plain data.

    Keyed by label throughout, so it lines up with the columns `steps.step_frame`
    produces without going back through tensor positions.
    """
    objectives = [
        {"label": cfg.label,
         "to_minimize": bool(cfg.to_minimize),
         "ref_point": getattr(cfg, "ref_point", None),
         "unit": cfg.unit,
         "bounds": list(cfg.bounds) if getattr(cfg, "bounds", None) is not None else None}
        for cfg in objective.obj_cfg or []
    ]
    return {
        "parameters": [{"label": cfg.label, "bounds": list(cfg.bounds),
                        "unit": cfg.unit, "resolution": getattr(cfg, "resolution", None)}
                       for cfg in objective.par_cfg or []],
        "objectives": objectives,
        "constraints": [{"label": cfg.label} for cfg in objective.ineq_Y_con_cfg or []],
        "trackers": [{"label": cfg.label, "unit": cfg.unit,
                      "bounds": list(cfg.bounds) if cfg.bounds is not None else None}
                     for cfg in objective.trk_cfg or []],
        # Where the hypervolume is measured from, in the objectives' own order.
        "ref_point": [o["ref_point"] for o in objectives],
        "minimized": {o["label"]: o["to_minimize"] for o in objectives},
        "max_hv": getattr(objective, "max_hv", None),
        "best_value": getattr(objective, "best_value", None),
    }