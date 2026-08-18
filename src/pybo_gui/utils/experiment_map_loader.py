"""Loads the experiment map the plot scripts read.

Same module name, function name and returned record shape as the ds_edm original, so the
plot scripts call it unchanged. The one difference is where the records come from: pybo
writes no per-experiment metadata.json, so the builder puts complete records into
experiment_map.json and this module returns them, joining group_map.json for `parameters`
exactly as the original does.
"""
import json
import os
import warnings
from pathlib import Path

# What a group_map entry carries besides the parameters it stands for. Mirrors the keys
# build_group_map writes around them.
GROUP_IDENTITY_KEYS = frozenset(
    ("group_id", "run", "technology", "optimizer", "experiment_type", "source",
     "replicates"))


def load_experiments_from_map(map_path: str, valid_only: bool = False) -> list[dict]:
    """Return experiments in chronological order.

    Joins two sources:
    - experiment_map.json : the records, already carrying results and provenance
    - group_map.json      : parameters per group

    `valid_only` is accepted for compatibility with the original signature. A pybo
    observation is either recorded or absent, so there is no validation flag to filter on
    and every record is returned.
    """
    map_dir = os.path.dirname(map_path)

    with open(map_path, encoding="utf-8") as file:
        exp_map = json.load(file)

    group_params = {}
    group_map_path = os.path.join(map_dir, "group_map.json")
    if Path(group_map_path).exists():
        with open(group_map_path, encoding="utf-8") as file:
            group_list = json.load(file)
        # The parameters a group stands for, and only those: a group also carries what
        # identifies it (the run, the arm, how the rows were made) and how many rows it
        # holds, none of which is a parameter. Those ride along harmlessly for a row that
        # records its own parameters - the join below never reaches for them - but a row
        # with none, a fixed technology measured as a baseline, would otherwise come out
        # of here with "replicates" among its parameters.
        group_params = {g["group_id"]: {k: v for k, v in g.items()
                                        if k not in GROUP_IDENTITY_KEYS}
                        for g in group_list}
    else:
        warnings.warn(f"group_map.json not found next to {map_path}; "
                      f"parameters will be empty.")

    experiments = []
    for entry in exp_map["experiments"]:
        raw_results = entry.get("results", {})
        experiments.append({
            "index":                  entry["index"],
            "iteration":              entry.get("iteration"),
            "group_id":               entry["group_id"],
            "experiment_id":          entry["experiment_id"],
            "experiment_type":        entry["experiment_type"],
            "optimizer":              entry.get("optimizer"),
            # What produced the measurement (the rig, the problem), as opposed to the
            # optimizer that chose where to take it. None on a record that names none.
            "technology":             entry.get("technology"),
            # "experimental" (real rig) or "synthetic" (a pybo trial) - orthogonal to
            # the optimizer and to a row's own source. None on a record from before this
            # field existed.
            "provenance":             entry.get("provenance"),
            # Which run produced it. The rig had no equivalent - one machine, one
            # sequence - but a campaign here spans several independent runs.
            "run":                    entry.get("run"),
            "start_time":             entry["start_time"],
            "parameters":             entry.get("parameters")
                                      or group_params.get(entry["group_id"], {}),
            "results":                {k: round(v, 6) if v is not None else None
                                       for k, v in raw_results.items()},
            # Recorded by the EDM rig, not by a pybo run; kept so the correlation matrix
            # can ask for them without special-casing.
            "external_temperature_c": entry.get("external_temperature_c"),
            "machine_temperature":    entry.get("machine_temperature", {}),
        })

    return experiments
