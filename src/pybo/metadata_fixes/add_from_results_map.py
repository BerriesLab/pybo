"""
Adds a result from a results map to every experiment.json at or under a root.

The map is {experiment name: {result: value}}, the name being the folder the
experiment.json sits in - which is what joins the two. Use this rather than add_entry
when the value differs per experiment: add_entry's --expr sees only the observation it
landed on, so it has no way to look a value up by folder.

Paths use dot notation and "*" fans out over a list, exactly as in add_entry, and the
leaf is the label the value is written under:

    python -m pybo.metadata_fixes.add_from_results_map <root> <map> <result> <path> [--apply]

    python -m pybo.metadata_fixes.add_from_results_map data/vformac_converted \\
        data/results_map.json cavity_depth_mm "data.*.trackers.Cavity Depth" --apply

An experiment the map does not name, or names without that result, is reported and left
alone - a result not every experiment has is the case this is for, and writing null would
claim it was measured and found empty.

The "<label>_var" companion is created alongside, at null, since every pybo channel is
that pair and half a channel is the kind of thing nothing notices until a fit reads it.
An existing _var is left as it is.

Nothing is written without --apply: the run is previewed and the files are left alone.
"""
import argparse
import sys
from pathlib import Path

from pybo.metadata_fixes._common import find_experiments, resolve, load, save, split_roots

# Labels like "Tool Wear (μm)" get printed, and stdout defaults to cp1252 on
# Windows, which cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(
    description="Add a per-experiment result from a results map to every experiment.json.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("root", help="Folder(s) searched recursively for experiment.json "
                                 "files - comma-separated for more than one.")
parser.add_argument("map", help="JSON file of {experiment name: {result: value}}")
parser.add_argument("result", help="Which result to take from the map")
parser.add_argument("path", help="Dot-notation path to write, '*' fanning out over a list. "
                                 "Its leaf is the label the value is written under.")
parser.add_argument("--apply", action="store_true",
                    help="Actually write. Without it, the changes are printed and "
                         "nothing is written.")
args = parser.parse_args()

results_map = load(Path(args.map))
parts = args.path.split(".")
label = parts[-1]
updated = unnamed = absent = 0

for path in find_experiments(split_roots(args.root)):
    name = path.parent.name
    if name not in results_map:
        print(f"  {name}: not in the map")
        unnamed += 1
        continue
    if args.result not in results_map[name]:
        absent += 1
        continue

    value = results_map[name][args.result]
    doc = load(path)
    written = 0
    # create=True: the group is usually there already (trackers, objectives) but the
    # label being added is by definition not, which is what this script is for.
    for parent, key, _ in resolve(doc, parts, doc, create=True):
        parent[key] = value
        parent.setdefault(f"{key}_var", None)
        written += 1

    if not written:
        print(f"  {name}: '{args.path}' landed nowhere")
        continue
    where = f"{written}x " if written > 1 else ""
    if args.apply:
        save(path, doc)
        print(f"  {name}: wrote {where}{label} = {value!r}")
    else:
        print(f"  [preview] {name}: would write {where}{label} = {value!r}")
    updated += 1

summary = [f"{updated} file(s)"]
if absent:
    summary.append(f"{absent} without {args.result}")
if unnamed:
    summary.append(f"{unnamed} not in the map")
print(f"Done. ({', '.join(summary)})" + ("" if args.apply else " — preview, nothing written"))
