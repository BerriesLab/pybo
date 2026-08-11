"""
Renames an entry in every experiment.json at or under a root, in place.

Paths use dot notation, and "*" fans out over a list, which is how the observations
inside "data" are reached. The new name is the leaf only: a rename stays inside its
own group, and moving a channel between groups is a different operation.

Usage:
    python -m pybo.metadata_fixes.rename_entry <root> <path> <new_name> [--apply]
    python -m pybo.metadata_fixes.rename_entry data/vformac \\
        "data.*.objectives.mrr_mm3_min" "Material Removal Rate (mm3/min)" --apply

The "<label>_var" companion is renamed along with the label. Every pybo channel is
that pair, and renaming one half leaves a broken one — the kind of mistake nobody
notices until a fit reads a column that is half there. A channel without a _var is
left alone rather than treated as an error.

The key keeps its position in the file, so the diff is the renamed lines and nothing
else. A name already taken in the same dict is refused: the file is skipped with a
message rather than one value overwriting the other.

Nothing is written without --apply: the run is previewed and the files are left alone.
"""
import argparse
import sys

from pybo.metadata_fixes._common import find_experiments, resolve, load, save

# Labels like "Tool Wear (μm)" get printed, and stdout defaults to cp1252 on
# Windows, which cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(
    description="Rename an entry in every experiment.json under a root.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("root", help="Folder searched recursively for experiment.json files")
parser.add_argument("path", help="Dot-notation path to the entry, '*' fanning out over a list")
parser.add_argument("new_name", help="The new leaf name, without the path")
parser.add_argument("--apply", action="store_true",
                    help="Actually write. Without it, the renames are printed and "
                         "nothing is written.")
args = parser.parse_args()

parts = args.path.split(".")
new = args.new_name
renamed_files = untouched = collisions = 0

for path in find_experiments(args.root):
    doc = load(path)
    hits = 0
    clash = False
    # create=False: renaming must never bring into existence the key it fails to find.
    for parent, old, _ in resolve(doc, parts, doc, create=False):
        if old not in parent:
            continue
        if new in parent:
            clash = True
            continue
        # Rebuilt rather than popped and reinserted, so the key keeps its position
        # and the diff is the renamed lines instead of a deletion plus an append.
        rebuilt = {}
        for key, value in parent.items():
            if key == old:
                rebuilt[new] = value
            elif key == f"{old}_var":
                rebuilt[f"{new}_var"] = value
            else:
                rebuilt[key] = value
        parent.clear()
        parent.update(rebuilt)
        hits += 1

    if clash:
        print(f"  SKIPPED {path.parent.name}: '{new}' is already there")
        collisions += 1
        continue
    if not hits:
        untouched += 1
        continue
    where = f"{hits}x " if hits > 1 else ""
    if args.apply:
        save(path, doc)
        print(f"  {path.parent.name}: renamed {where}{parts[-1]} -> {new}")
    else:
        print(f"  [preview] {path.parent.name}: would rename {where}{parts[-1]} -> {new}")
    renamed_files += 1

summary = [f"{renamed_files} file(s)"]
if untouched:
    summary.append(f"{untouched} with no matching path")
if collisions:
    summary.append(f"{collisions} skipped, name already taken")
print(f"Done. ({', '.join(summary)})" + ("" if args.apply else " — preview, nothing written"))
