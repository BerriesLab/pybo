"""Assign group IDs to observations and write group_map.json.

A group is one setting within one run: the observations of a run taken at the same
parameters. That is what --grouped averages, so its error bars measure the spread between
repeats of a setting - what a noisy objective, or a setting deliberately repeated, gives.

The run is part of the key because two runs at the same parameters are two different
campaigns, not repeats of one measurement; pooling them would mix run-to-run variation
into a bar meant to show measurement spread. The arm needs no place in the key, since a
run belongs to one arm.

A deterministic objective evaluated twice at the same parameters returns the same value,
so those groups have zero-width bars - correct, and not a sign of anything wrong.

As in the original, the group_id is assigned here rather than by the map builder, and
written back into the experiment map - so build_experiment_map records the facts and this
decides how they group.
"""
import argparse
import json
from pathlib import Path

# Shared with build_polynomial_gt.py, which needs the same rig-grid snap and lives in
# core pybo - so this can't stay pybo_gui-only without pulling the GUI package into a
# script meant to keep running from a terminal on its own.
from pybo.utils.helpers import snap as _snap


def _setting_key(entry: dict, resolutions: dict) -> tuple:
    """The run and parameters an observation was taken at, on the rig's own grid."""
    parameters = entry.get("parameters") or {}
    return (entry.get("run"),) + tuple(
        (label, _snap(parameters[label], resolutions.get(label)))
        for label in sorted(parameters)
    )


def build_groups(exp_map: dict, resolutions: dict | None = None) -> list:
    """Assign a group_id to every experiment in `exp_map`, and return the group list.

    `resolutions` maps a parameter label to the rig's step for it. A parameter left out
    is compared as measured, which is what an objective that never declared a resolution
    gets - and what every map got before resolutions existed.

    `exp_map` is modified in place: the entries gain the group_id the plot scripts read.
    Separate from the file writing so the GUI can group a map it holds in memory.
    """
    resolutions = resolutions or {}
    key_to_gid = {}
    groups = {}
    seen = {}
    next_gid = 1

    # Every parameter any observation in the map records, not just the ones a given group
    # has: a selection spanning runs of different problems - or a reference run with no
    # parameters at all - is one table, and a column missing from a row means that row
    # never took a value there. Written as None rather than dropped, so the rows line up
    # and the gap is visible instead of implied by a shorter row.
    all_labels = sorted({label for entry in exp_map.get("experiments", [])
                         for label in (entry.get("parameters") or {})})

    for entry in exp_map.get("experiments", []):
        key = _setting_key(entry, resolutions)
        if key not in key_to_gid:
            key_to_gid[key] = next_gid
            parameters = entry.get("parameters") or {}
            groups[next_gid] = {
                "group_id": next_gid,
                "run": entry.get("run"),
                # What produced the measurements, as opposed to what chose where to take
                # them. One value per group, a group being one setting within one run.
                "technology": entry.get("technology"),
                "experiment_type": None,
                "optimizer": entry.get("optimizer"),
                # How the rows were made: the step record's own experiment_type (real rig
                # data or a simulated trial - the map carries it as `provenance`) and the
                # rows' source (the initial design, an optimizer's proposal, ...). Filled
                # in below rather than here: a group's rows are only all known once the
                # pass is over.
                "source": None,
                # The parameters spread across the row, on the same grid the key used, so
                # a group reads as the setting it stands for. Every label in the map, so
                # every group has the same columns; None where this one has no value.
                **{label: _snap(parameters.get(label), resolutions.get(label))
                   for label in all_labels},
                "replicates": 0,
            }
            seen[next_gid] = {"experiment_type": set(), "source": set()}
            next_gid += 1
        gid = key_to_gid[key]
        entry["group_id"] = gid
        groups[gid]["replicates"] += 1
        seen[gid]["experiment_type"].add(entry.get("provenance"))
        seen[gid]["source"].add(entry.get("source"))

    # One value each, normally - a group is one setting within one run. Joined rather than
    # picked from when it is not, so a group holding both (a design point an optimizer
    # later proposed again) says so instead of hiding half of itself behind the other.
    for gid, fields in seen.items():
        for field, values in fields.items():
            present = sorted(value for value in values if value)
            groups[gid][field] = "/".join(present) if present else None

    return [groups[gid] for gid in sorted(groups)]


def build_group_map(out_dir, resolutions: dict | None = None) -> Path:
    """Group the experiment map under `out_dir`, writing both files back."""
    out_dir = Path(out_dir)
    map_path = out_dir / "experiment_map.json"
    exp_map = json.loads(map_path.read_text(encoding="utf-8"))
    group_list = build_groups(exp_map, resolutions)

    # The map gains the group_id, so the two files agree about what a group is.
    map_path.write_text(json.dumps(exp_map, indent=2), encoding="utf-8")
    out_path = out_dir / "group_map.json"
    out_path.write_text(json.dumps(group_list, indent=2), encoding="utf-8")
    repeated = sum(1 for g in group_list if g["replicates"] > 1)
    print(f"Wrote {len(group_list)} groups ({repeated} with repeats) -> {out_path}")
    print(f"Updated {map_path} with group_id")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Group observations by run and parameter values, and write "
                    "group_map.json.")
    parser.add_argument("out_dir", help="Directory holding experiment_map.json")
    parser.add_argument("--resolution", action="append", default=[], metavar="LABEL=STEP",
                        help="The rig's smallest distinguishable step for one parameter, "
                             "in that parameter's units, e.g. --resolution V0=0.1. Records "
                             "closer together than this are the same setting. Repeat the "
                             "flag per parameter; a parameter left out is compared as "
                             "measured. The objective normally carries these, and the GUI "
                             "reads them from there.")
    args = parser.parse_args()

    resolutions = {}
    for item in args.resolution:
        label, _, step = item.partition("=")
        if not _:
            raise SystemExit(f"--resolution wants LABEL=STEP, got {item!r}")
        resolutions[label] = float(step)

    build_group_map(args.out_dir, resolutions)


if __name__ == "__main__":
    main()
