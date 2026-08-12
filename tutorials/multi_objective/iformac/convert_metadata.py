import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

# The repo root, so the absolute import below resolves when this is launched as a script
# from a terminal. An IDE puts the content root on the path already; python does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

# Labels carry units - "Tool Wear (um)" is spelled with a real mu - and a Windows console
# is cp1252, which cannot encode it. Files are written utf-8 regardless; this only keeps
# the preview and the error messages from dying on a character they are quoting.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tutorials.multi_objective.iformac.objective import IFormACConstrained

# --- mapping ---
# label -> how to get it out of one step's metadata.json. Written out rather than matched
# by position or by case, so renaming a label breaks here, loudly, instead of pairing the
# wrong column. Parameters map straight to a key; everything else is a callable over the
# step's values and the objective, because two of the three are derived rather than logged.
PAR_MAP = {"Maximum Current": "I_MAX",
           "Pedestal Current": "I_P",
           "Maximum Ramp Time": "tau_R_max"}

# The rig logs a down time, not a rate. Every step in data/iformac satisfies
# MRR * down_time_minutes = 144.4 to 3e-15, so that constant is the volume the cycle
# removes, and the rate is the volume over the time.
MACHINED_VOLUME = 144.4  # mm^3

OBJ_MAP = {
    "Material Removal Rate": lambda v, o: MACHINED_VOLUME / v["down_time_minutes"],
    "Tool Wear": lambda v, o: v["wear_microns"],
}

# The constraint column carries the distance from the feasibility band, not the measured
# minutes - the same quantity IFormACConstrained.evaluate_true_constraint produces, so a
# ported row and a simulated one mean the same thing under the same label. The band is
# read off the objective rather than restated here, so there is one definition of it.
CON_MAP = {
    "Orbiting Time Deviation": lambda v, o: max(
        0.0,
        o._orbiting_target - o._delta - v["orbiting_time_minutes"],
        v["orbiting_time_minutes"] - o._orbiting_target - o._delta),
}

# The measurement itself, kept because the constraint column above is a distance.
TRK_MAP = {"Orbiting Time": lambda v, o: v["orbiting_time_minutes"]}

# Every metadata key the maps above read. A step missing any of them cannot be converted.
REQUIRED_KEYS = ["I_MAX", "I_P", "tau_R_max",
                 "wear_microns", "down_time_minutes", "orbiting_time_minutes"]

# Nothing is rescaled: the ramp time stays in the nanoseconds the rig logs it in, and the
# objective now declares that parameter in nanoseconds too, so the two agree.
SCALE = {}

parser = argparse.ArgumentParser(
    description="Convert IFormAC's metadata.json into pybo's experiment.json, one per step.")
parser.add_argument("root", help="Run folder holding step_NNN/metadata.json")
parser.add_argument("--out", default=None,
                    help="Where the converted tree is written (default: beside each "
                         "metadata.json, leaving it in place)")
parser.add_argument("--source", default=None,
                    help="Provenance recorded on every observation. Default: each step's "
                         "own experiment_type, lowercased.")
parser.add_argument("--apply", action="store_true",
                    help="Actually write. Without it, the first record is previewed and "
                         "nothing is written.")
args = parser.parse_args()

root = Path(args.root).resolve()
steps = sorted(root.glob("step_*/metadata.json"))
if not steps:
    raise SystemExit(f"No step_*/metadata.json under {root}.")

# --- the mapping must still describe the objective ---
# The objective is the authority on what a label is. Checking against it means an amended
# problem - a renamed objective, a constraint added or dropped - stops the port rather
# than writing files that no longer match the problem they claim to belong to.
objective = IFormACConstrained(device=torch.device("cpu"), dtype=torch.float64)
groups = [("parameters", PAR_MAP, [c.label for c in objective.par_cfg or []]),
          ("objectives", OBJ_MAP, [c.label for c in objective.obj_cfg or []]),
          ("constraints", CON_MAP, [c.label for c in objective.ineq_Y_con_cfg or []]),
          ("trackers", TRK_MAP, [c.label for c in objective.trk_cfg or []])]
for name, mapping, labels in groups:
    if set(mapping) != set(labels):
        raise SystemExit(f"{name}: the objective declares {labels}, the mapping covers "
                         f"{sorted(mapping)}. Update the map at the top of this script.")
print("mapping checks out against IFormACConstrained: "
      + ", ".join(f"{len(m)} {n}" for n, m, _ in groups))

# --- convert ---
# observation_n counts across the whole run, the way to_json numbers a batch: these are
# single-observation steps, so it is the step's position among those that converted.
records, skipped = [], []
for path in steps:
    meta = json.loads(path.read_text(encoding="utf-8"))
    # Inputs and outputs sit in two dicts; nothing downstream cares which one a value came
    # from. Scaled on the way in, so every value below is already in the problem's units.
    values = dict(meta.get("parameters") or {}, **(meta.get("results") or {}))
    values = {key: value * SCALE.get(key, 1.0) if isinstance(value, (int, float)) else value
              for key, value in values.items()}

    # Three steps record no parameters at all. They cannot be converted, and skipping is
    # the only option - but it is reported: a silently short campaign is worse than none.
    missing = sorted(key for key in REQUIRED_KEYS if key not in values)
    if missing:
        skipped.append((path.parent.name, missing))
        continue

    # Every _var is null. Nothing here was measured twice, and noise estimated from
    # replicate fabrications belongs to the ground truth, not to an observation.
    named = {}
    for name, mapping, labels in groups[1:]:
        named[name] = {}
        for label in labels:
            named[name][label] = mapping[label](values, objective)
            named[name][f"{label}_var"] = None

    source = args.source or (meta.get("experiment_type") or "unknown").lower()
    records.append((
        (Path(args.out).resolve() / path.parent.name if args.out else path.parent) / "experiment.json",
        {
            "datetime": datetime.fromtimestamp(meta["start_time"]).isoformat(),
            "experiment_type": source,
            "batch_size": 1,
            "data": [{
                "observation_n": len(records),
                "source": source,
                # In the objective's own parameter order, which is the order to_json writes.
                "parameters": {label: values[PAR_MAP[label]] for label in groups[0][2]},
                "objectives": named["objectives"],
                "constraints": named["constraints"],
                "trackers": named["trackers"],
            }],
        }))

if skipped:
    print(f"\n{len(skipped)} step(s) skipped, missing the keys they map to:")
    for name, missing in skipped:
        print(f"  {name}: {missing}")
if not records:
    raise SystemExit("\nNothing converted.")

sources = {}
for _, payload in records:
    sources[payload["experiment_type"]] = sources.get(payload["experiment_type"], 0) + 1
print(f"\n{len(records)} step(s) converted, source: "
      + ", ".join(f"{name} x{count}" for name, count in sorted(sources.items())))
print("\nFirst record:\n")
print(json.dumps(records[0][1], indent=2, ensure_ascii=False))

if not args.apply:
    print(f"\nNothing written. Re-run with --apply to write {len(records)} files.")
    raise SystemExit(0)

for out_path, payload in records:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nwrote {len(records)} x experiment.json")
