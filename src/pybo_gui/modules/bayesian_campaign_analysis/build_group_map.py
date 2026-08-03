"""Assign group parameters and write group_map.json.

A group is one observation index across replicate runs, so the group's "parameters" are
not a shared setting the way they were on the rig - each replicate sampled its own point.
What the group does share is where it sits in the campaign, so that is what is recorded:
the evaluation count, and how many runs reached it.
"""
import argparse
import json
from pathlib import Path


def build_groups(exp_map: dict) -> list:
    """The group list for an experiment map, in memory.

    Separate from the file writing so the GUI can hold a map without putting one on disk.
    """
    groups = {}
    for entry in exp_map["experiments"]:
        gid = entry["group_id"]
        group = groups.setdefault(gid, {"group_id": gid, "observation": gid,
                                        "replicates": 0, "sources": set()})
        group["replicates"] += 1
        group["sources"].add(entry["experiment_type"])

    group_list = []
    for gid in sorted(groups, key=lambda g: (g is None, g)):
        group = groups[gid]
        # A group is normally one provenance throughout; join rather than pick, so a
        # campaign whose runs disagree on where an index came from says so.
        group["sources"] = "/".join(sorted(s for s in group["sources"] if s))
        group_list.append(group)

    return group_list


def build_group_map(out_dir) -> Path:
    """Read experiment_map.json under `out_dir` and write group_map.json beside it."""
    out_dir = Path(out_dir)
    exp_map = json.loads((out_dir / "experiment_map.json").read_text(encoding="utf-8"))
    group_list = build_groups(exp_map)
    out_path = out_dir / "group_map.json"
    out_path.write_text(json.dumps(group_list, indent=2), encoding="utf-8")
    print(f"Wrote {len(group_list)} groups -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Group observations by index and write group_map.json.")
    parser.add_argument("out_dir", help="Directory holding experiment_map.json")
    args = parser.parse_args()
    build_group_map(args.out_dir)


if __name__ == "__main__":
    main()
