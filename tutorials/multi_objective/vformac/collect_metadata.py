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
# Natural sort, so step_2 precedes step_10 whatever the zero padding. Nothing here reads
# the clock: file times survive copying between machines badly, and the folder names are
# what the loop itself wrote.
def natural(path):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(path.relative_to(root)))]

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
    plan.append((path, run, step_dir.name))

runs = {}
for path, run, _ in plan:
    runs.setdefault(run, []).append(path)

print("\nplan:")
targets = []
for run, paths in runs.items():
    print(f"  {run}/  ({len(paths)} steps)")
    for i, path in enumerate(paths):
        target = out / run / f"step_{i:03d}" / args.name
        targets.append((path, target))
        # The full source path is kept in the line so a mis-ordered mapping, or a
        # --step-level that split the wrong ancestor, is visible here and not after the copy.
        print(f"    {str(path.relative_to(root)):55s} -> {run}/step_{i:03d}/{args.name}")

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
# another machine. One index per run keeps the mapping recoverable.
for run, paths in runs.items():
    index = {f"step_{i:03d}": str(path.parent.relative_to(root))
             for i, path in enumerate(paths)}
    (out / run / "index.json").write_text(
        json.dumps({"root": str(root), "steps": index}, indent=2), encoding="utf-8")
print(f"wrote index.json for {len(runs)} run(s)")
