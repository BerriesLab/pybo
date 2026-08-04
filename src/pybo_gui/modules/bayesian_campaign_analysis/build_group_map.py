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

DEFAULT_DECIMALS = 6


def _setting_key(entry: dict, decimals: int) -> tuple:
    """The run and parameters an observation was taken at.

    Parameters are compared rounded, because two floats meant to be the same setting
    rarely agree bit for bit.
    """
    parameters = entry.get("parameters") or {}
    return (entry.get("run"),) + tuple(
        (label, round(parameters[label], decimals) if parameters[label] is not None else None)
        for label in sorted(parameters)
    )


def build_groups(exp_map: dict, decimals: int = DEFAULT_DECIMALS) -> list:
    """Assign a group_id to every experiment in `exp_map`, and return the group list.

    `exp_map` is modified in place: the entries gain the group_id the plot scripts read.
    Separate from the file writing so the GUI can group a map it holds in memory.
    """
    key_to_gid = {}
    groups = {}
    next_gid = 1

    for entry in exp_map.get("experiments", []):
        key = _setting_key(entry, decimals)
        if key not in key_to_gid:
            key_to_gid[key] = next_gid
            parameters = entry.get("parameters") or {}
            groups[next_gid] = {
                "group_id": next_gid,
                "run": entry.get("run"),
                "technology": entry.get("technology"),
                # The parameters spread across the row, as the original wrote them, so a
                # group reads as the setting it stands for.
                **{label: (round(parameters[label], decimals)
                           if parameters[label] is not None else None)
                   for label in sorted(parameters)},
                "replicates": 0,
            }
            next_gid += 1
        gid = key_to_gid[key]
        entry["group_id"] = gid
        groups[gid]["replicates"] += 1

    return [groups[gid] for gid in sorted(groups)]


def build_group_map(out_dir, decimals: int = DEFAULT_DECIMALS) -> Path:
    """Group the experiment map under `out_dir`, writing both files back."""
    out_dir = Path(out_dir)
    map_path = out_dir / "experiment_map.json"
    exp_map = json.loads(map_path.read_text(encoding="utf-8"))
    group_list = build_groups(exp_map, decimals)

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
    parser.add_argument("--decimals", type=int, default=DEFAULT_DECIMALS,
                        help="Decimals a parameter is rounded to before comparing "
                             "(default: %(default)s).")
    args = parser.parse_args()
    build_group_map(args.out_dir, args.decimals)


if __name__ == "__main__":
    main()
