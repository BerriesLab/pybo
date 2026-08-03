"""Loading for the per-step records a run writes.

Each ``step_NNN/experiment.json`` holds the observations of one optimization step,
named from the problem definition (see ``OptimizerBase.to_json``). These files carry
values and provenance but no problem definition, so anything needing senses, bounds or
a reference point pairs them with an objective loaded from disk - see
``objective_loader``.

The unit here is the observation, not the run: `step_frame` returns one row per
observation, so a selection of steps from several runs pools into a single frame the
way the campaign plots want it.
"""
import json
from pathlib import Path

import pandas as pd


def find_steps(roots: list[Path]) -> list[Path]:
    """Every step record under each root, sorted.

    A root may be a step directory, a run directory, or a whole study: the glob
    matches at any depth, so the same call serves each level of the layout. A root
    that is itself an experiment.json is taken as-is.
    """
    found = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
        else:
            found.extend(sorted(root.glob("**/experiment.json")))
    return found


def step_frame(paths: list[Path]) -> pd.DataFrame:
    """Long frame of the selected steps: one row per observation.

    Parameters, objectives, constraints and trackers become columns named by their
    labels, so a column is addressed the way the problem calls it rather than by tensor
    position. `run` is the directory holding the step, which is what separates replicates
    when several are pooled.
    """
    rows = []
    for path in paths:
        path = Path(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        step_dir = path.parent
        for observation in record.get("data", []):
            row = {
                "run": step_dir.parent.name,
                "step": step_dir.name,
                "path": str(path),
                "datetime": record.get("datetime"),
                "experiment_type": record.get("experiment_type"),
                "observation_n": observation.get("observation_n"),
                "source": observation.get("source"),
                "is_initial": observation.get("source") == "initial",
            }
            for group in ("parameters", "objectives", "constraints", "trackers"):
                row.update(observation.get(group) or {})
            rows.append(row)
    if not rows:
        raise SystemExit("No observations found in the selected steps.")
    return pd.DataFrame(rows)


def group_labels(paths: list[Path]) -> dict:
    """Which columns of `step_frame` came from which group, read back from the records.

    The frame flattens every group into columns, so this is what tells a caller that
    "Branin" is an objective and "par 00" a parameter without consulting the objective.
    """
    labels = {"parameters": [], "objectives": [], "constraints": [], "trackers": []}
    for path in paths:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        for observation in record.get("data", []):
            for group in labels:
                for label in (observation.get(group) or {}):
                    if label not in labels[group] and not label.endswith("_var"):
                        labels[group].append(label)
    return labels