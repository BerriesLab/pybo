import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Collect the metadata.json of every experiment under a Bayesian "
                "optimization root into a pybo-style <run>/step_NNN/ tree.")
parser.add_argument("root", help="Root folder of the optimization run")
parser.add_argument("out", help="Where the new tree is written")
parser.add_argument("--name", default="metadata.json",
                    help="File collected from each experiment folder (default: %(default)s)")
parser.add_argument("--step-level", type=int, default=1,
                    help="Which ancestor of the file is the step: 1 = its own folder, "
                         "2 = its grandparent, and so on (default: %(default)s)")
parser.add_argument("--run-level", type=int, default=0,
                    help="Which ancestor names the run: 0 = one single run named after "
                         "root, 1 = the step's parent, 2 = its grandparent")
parser.add_argument("--run-name", default=None,
                    help="Name of the run folder, overriding the one --run-level picked. "
                         "Only meaningful with --run-level 0.")
parser.add_argument("--recursive", action="store_true",
                    help="Search at any depth instead of only the folders directly under "
                         "root. Off by default: a metadata.json nested deeper inside an "
                         "experiment folder is part of that experiment, not a step of its own.")
parser.add_argument("--order-by", default=None, metavar="FIELD",
                    help="Number the steps by this top-level field of the collected file, "
                         "e.g. --order-by start_time. Use it whenever the source is more "
                         "than one folder: each one restarts its own numbering, so sorting "
                         "by name interleaves them and step_000 stops being the first "
                         "experiment. Read from inside the JSON, not from the file's "
                         "mtime. Default: sort by name.")
parser.add_argument("--keep-source-name", action="store_true",
                    help="Name each step after the folder it came from instead of "
                         "step_NNN, keeping the rig's own numbering as the name of record "
                         "- which is where a repeat says what it repeats. Safe because "
                         "nothing downstream reads order out of the name: build_map sorts "
                         "on the datetime inside each record. Refuses if two source "
                         "folders in one run share a name.")
parser.add_argument("--apply", action="store_true",
                    help="Actually copy. Without it, the plan is printed and nothing is written.")
args = parser.parse_args()

root = Path(args.root).resolve()
out = Path(args.out).resolve()
if not root.is_dir():
    raise SystemExit(f"{root} is not a directory.")

# --- find ---
# One layer down, so a step is a folder sitting directly under the root. Anything the
# experiment folders hold deeper belongs to that experiment; sweeping it in would promote
# it to a step of its own and silently pad the campaign with duplicates.
found = sorted(root.rglob(args.name) if args.recursive else root.glob(f"*/{args.name}"))
if not found:
    raise SystemExit(f"No {args.name} in the folders directly under {root}."
                     + ("" if args.recursive else " Use --recursive to search deeper."))
print(f"{len(found)} x {args.name} under {root}\n")

# What was left behind, so a layout that does nest them is not silently half-collected.
if not args.recursive:
    deeper = len(list(root.rglob(args.name))) - len(found)
    if deeper:
        print(f"note: {deeper} more {args.name} sit deeper and were ignored "
              f"(--recursive to include them)\n")

# --- what the source layout actually looks like ---
# The naming convention is the whole job, so it gets read rather than assumed. Each shape
# is a path with the varying names blanked, so repeated levels collapse into one line.
print("source shapes (digits blanked, with how many files sit at each):")
shapes = Counter()
for path in found:
    parts = path.relative_to(root).parts[:-1]
    shapes[tuple(re.sub(r"\d+", "N", p) for p in parts)] += 1
for shape, count in shapes.most_common():
    print(f"  {count:4d}  {'/'.join(shape) or '.'}/{args.name}")

depths = sorted({len(path.relative_to(root).parts) - 1 for path in found})
print(f"\nfolder depths below root: {depths}")
if len(depths) > 1:
    print("  WARNING: mixed depths - a single --step-level will not suit every file.")

# --- order ---
# Natural sort by default, so step_2 precedes step_10 whatever the zero padding: the
# folder names are what the loop itself wrote, and for one run they carry the order.
#
# They stop carrying it as soon as the source is several folders, because each restarts
# its own numbering - a campaign collected from spark_accelerator_1-5/, _6-20/ and _12/
# comes out interleaved, and step_000 is then not the first experiment. --order-by fixes
# that by reading a recorded timestamp *inside* the file. The objection to clocks is about
# filesystem mtimes, which copying between machines destroys; a start_time written into
# the payload survives it intact.
def natural(path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(path.relative_to(root)))]

if args.order_by:
    stamps, unstamped = {}, []
    for path in found:
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(args.order_by)
        except Exception:  # noqa: BLE001 - unreadable files are reported further down too
            value = None
        if value is None:
            unstamped.append(path)
        else:
            stamps[path] = value
    # Refused rather than mixed: ordering the files that carry the field by time and the
    # rest by name would order part of the campaign one way and part the other, which is
    # worse than either and invisible afterwards.
    if unstamped:
        raise SystemExit(
            f"--order-by {args.order_by}: {len(unstamped)} of {len(found)} file(s) do not "
            f"carry it, e.g. {unstamped[0].relative_to(root)}. Pick a field every file has.")
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                  for v in stamps.values())
    if not (numeric or all(isinstance(v, str) for v in stamps.values())):
        raise SystemExit(f"--order-by {args.order_by}: the values are a mix of types, so "
                         f"they cannot be ordered against each other.")
    found.sort(key=lambda p: stamps[p])
    print(f"\nordered by {args.order_by}: {found[0].parent.name} ({stamps[found[0]]}) "
          f"-> {found[-1].parent.name} ({stamps[found[-1]]})")
else:
    found.sort(key=natural)

# --- plan ---
# The step folder is an ancestor of the file, not necessarily its own parent: an
# experiment folder may hold the metadata beside the heavy files it is being separated
# from. --run-level then decides whether the tree is one run or several.
plan = []
for path in found:
    step_dir = path.parents[args.step_level - 1] if args.step_level >= 1 else path.parent
    run = (args.run_name or root.name) if args.run_level == 0 \
        else path.parents[args.run_level - 1].name
    plan.append((path, run, step_dir))

runs = {}
for path, run, step_dir in plan:
    runs.setdefault(run, []).append((path, step_dir))

# A source name is only usable if it identifies the step on its own. Two batches that
# each hold an experiment_0001 would collapse into one folder, silently losing a step.
if args.keep_source_name:
    for run, entries in runs.items():
        seen = {}
        for path, step_dir in entries:
            seen.setdefault(step_dir.name, []).append(str(step_dir.relative_to(root)))
        clashes = {name: where for name, where in seen.items() if len(where) > 1}
        if clashes:
            raise SystemExit(
                f"--keep-source-name: {len(clashes)} name(s) are not unique within run "
                f"{run}, e.g. {list(clashes)[0]} <- {clashes[list(clashes)[0]]}. Collect "
                f"without the flag, or split the runs so each name appears once.")

print("\nplan:")
targets = []
for run, entries in runs.items():
    print(f"  {run}/  ({len(entries)} steps)")
    for i, (path, step_dir) in enumerate(entries):
        # Order does not live in this name: build_map sorts records on the datetime each
        # one carries, so the name is free to be the rig's own rather than a counter.
        name = step_dir.name if args.keep_source_name else f"step_{i:03d}"
        targets.append((path, out / run / name / args.name))
        # The full source path is kept in the line so a mis-ordered mapping, or a
        # --step-level that split the wrong ancestor, is visible here and not after the copy.
        print(f"    {str(path.relative_to(root)):55s} -> {run}/{name}/{args.name}")

# --- readable? ---
# A metadata file that will not parse is worth knowing about now, not when the ground
# truth script chokes on it.
bad = []
for path, _ in targets:
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        bad.append((path, error))
if bad:
    print(f"\n{len(bad)} file(s) do not parse as JSON:")
    for path, error in bad[:10]:
        print(f"  {path.relative_to(root)}: {error}")

if not args.apply:
    print(f"\nNothing written. Re-run with --apply to copy {len(targets)} files.")
    raise SystemExit(0)

# --- copy ---
# Copied, never moved: the heavy originals on the other machine stay untouched.
for path, target in targets:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
print(f"\ncopied {len(targets)} files -> {out}")

# --- provenance ---
# step_002 says nothing about which EXP folder it came from, and the source sits on
# another machine. One index per run keeps the mapping recoverable - and with it the
# rig's own numbering, which is where a repeat says which experiment it repeats.
for run, entries in runs.items():
    index = {(step_dir.name if args.keep_source_name else f"step_{i:03d}"):
             str(step_dir.relative_to(root))
             for i, (path, step_dir) in enumerate(entries)}
    (out / run / "index.json").write_text(
        json.dumps({"root": str(root), "steps": index}, indent=2), encoding="utf-8")
print(f"wrote index.json for {len(runs)} run(s)")
