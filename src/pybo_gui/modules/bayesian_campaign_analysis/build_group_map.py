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

# Not a grouping policy and not a knob: what decides whether two records are the same
# setting is the parameter's resolution. This only absorbs float formatting for the
# parameters that have none, so two records meaning the same number still meet.
_DUST_DECIMALS = 6


def _snap(value, resolution):
    """The grid point `value` stands for, or None when it was never recorded.

    A resolution is the rig's own step, in the parameter's units, so the value snaps to
    that grid: two records land together exactly when the rig could not have told them
    apart. Without one there is no grid to snap to and the value is compared as measured,
    give or take float formatting.
    """
    if value is None:
        return None
    if resolution:
        # Rounding the multiple rather than the product keeps the key free of the float
        # dust that value/resolution*resolution leaves behind.
        return round(round(value / resolution) * resolution, 10)
    return round(value, _DUST_DECIMALS)


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
    next_gid = 1

    for entry in exp_map.get("experiments", []):
        key = _setting_key(entry, resolutions)
        if key not in key_to_gid:
            key_to_gid[key] = next_gid
            parameters = entry.get("parameters") or {}
            groups[next_gid] = {
                "group_id": next_gid,
                "run": entry.get("run"),
                "technology": entry.get("technology"),
                # The parameters spread across the row, on the same grid the key used, so
                # a group reads as the setting it stands for.
                **{label: _snap(parameters[label], resolutions.get(label))
                   for label in sorted(parameters)},
                "replicates": 0,
            }
            next_gid += 1
        gid = key_to_gid[key]
        entry["group_id"] = gid
        groups[gid]["replicates"] += 1

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
