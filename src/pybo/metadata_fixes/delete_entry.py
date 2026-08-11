"""
Deletes an entry from every experiment.json at or under a root.

Paths use dot notation, and "*" fans out over a list, which is how the observations
inside "data" are reached:

    experiment_type                        the file's own top level
    data.*.source                          every observation
    data.*.trackers.Orbiting Time (min)    every observation's tracker

Usage:
    python -m pybo.metadata_fixes.delete_entry <root> <path> [<path> ...] --apply
    python -m pybo.metadata_fixes.delete_entry data/iformac data.*.source --apply
    python -m pybo.metadata_fixes.delete_entry data/iformac "data.*.trackers.Orbiting Time (min)" \\
        "data.*.trackers.Orbiting Time (min)_var" --apply

A label and its _var companion are two separate entries: pass both to drop a channel
whole, or pybo reads a column that is half there.

Nothing is written without --apply: the run is previewed and the files are left alone.
"""
import argparse
import sys

from pybo.metadata_fixes._common import find_experiments, resolve, load, save

# Labels like "Tool Wear (μm)" get printed, and stdout defaults to cp1252 on
# Windows, which cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(
    description="Delete entries from every experiment.json under a root.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("root", help="Folder searched recursively for experiment.json files")
parser.add_argument("paths", nargs="+",
                    help="Dot-notation path(s) to delete, '*' fanning out over a list")
parser.add_argument("--apply", action="store_true",
                    help="Actually write. Without it, the removals are printed and "
                         "nothing is written.")
args = parser.parse_args()

updated = untouched = 0

for path in find_experiments(args.root):
    doc = load(path)
    removed = {}
    for dotted in args.paths:
        # create=False: deleting must never bring a path into existence on the way to
        # not finding it.
        for parent, key, _ in resolve(doc, dotted.split("."), doc, create=False):
            if key in parent:
                removed[dotted] = parent.pop(key)

    if not removed:
        untouched += 1
        continue
    if args.apply:
        save(path, doc)
        print(f"  {path.parent.name}: removed {removed}")
    else:
        print(f"  [preview] {path.parent.name}: would remove {removed}")
    updated += 1

summary = f"{updated} file(s), {untouched} with no matching path"
print(f"Done. ({summary})" + ("" if args.apply else " — preview, nothing written"))