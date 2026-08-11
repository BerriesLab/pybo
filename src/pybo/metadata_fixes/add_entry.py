"""
Adds or updates an entry in every experiment.json at or under a root.

Paths use dot notation, and "*" fans out over a list, which is how the observations
inside "data" are reached:

    experiment_type                        the file's own top level
    data.*.source                          every observation
    data.*.objectives.Tool Wear (μm)_var   every observation's objective variance

Usage — literal value:
    python -m pybo.metadata_fixes.add_entry <root> <path> <value> --apply
    python -m pybo.metadata_fixes.add_entry data/iformac data.*.source manual --apply

Usage — derived expression:
    python -m pybo.metadata_fixes.add_entry <root> <path> --expr EXPR --apply
    python -m pybo.metadata_fixes.add_entry data/iformac "data.*.trackers.Orbiting Time Deviation (min)" \\
        --expr "objectives['Machining Time (min)'] - 42" --apply

    The expression sees the observation the path landed on — parameters, objectives,
    constraints, trackers — plus math. Labels carry spaces and parentheses, so they
    are reached as objectives["Tool Wear (μm)"] rather than by attribute. A path that
    never crosses a list is evaluated against the file's top level instead.

Nothing is written without --apply: the run is previewed and the files are left alone.

Options:
    --apply           Actually write. Without it, the changes are printed and nothing is written.
    --skip-existing   Leave the entry alone where it already has a value
    --edit-if-eq OLD  Only write where the current value == OLD
"""
import argparse
import sys

from pybo.metadata_fixes._common import (
    find_experiments, resolve, coerce, evaluate, load, save,
)

# Labels like "Tool Wear (μm)" get printed, and stdout defaults to cp1252 on
# Windows, which cannot encode them.
sys.stdout.reconfigure(encoding="utf-8")

parser = argparse.ArgumentParser(
    description="Add or update an entry in every experiment.json under a root.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("root", help="Folder searched recursively for experiment.json files")
parser.add_argument("path", help="Dot-notation path to set, '*' fanning out over a list")
parser.add_argument("value", nargs="?", default=None, help="Literal value to write")
parser.add_argument("--expr", metavar="EXPR",
                    help="Python expression evaluated against the observation the path "
                         "landed on, plus math")
parser.add_argument("--apply", action="store_true",
                    help="Actually write. Without it, the changes are printed and "
                         "nothing is written.")
guard = parser.add_mutually_exclusive_group()
guard.add_argument("--skip-existing", action="store_true",
                   help="Leave the entry alone where it already has a value")
guard.add_argument("--edit-if-eq", metavar="OLD",
                   help="Only write where the current value == OLD")
args = parser.parse_args()

if args.value is None and args.expr is None:
    parser.error("either value or --expr is required")
if args.value is not None and args.expr is not None:
    parser.error("value and --expr are mutually exclusive")

parts = args.path.split(".")
updated = skipped = errors = 0

for path in find_experiments(args.root):
    doc = load(path)
    written = []
    # create=True: a path that does not exist yet is what this script is for. It only
    # builds the dicts along the way, never the list a "*" fans out over — there is no
    # sensible number of observations to invent.
    for parent, key, scope in resolve(doc, parts, doc, create=True):
        current = parent.get(key)
        if args.skip_existing and current is not None:
            skipped += 1
            continue
        if args.edit_if_eq is not None and current != coerce(args.edit_if_eq):
            skipped += 1
            continue
        try:
            value = evaluate(args.expr, scope) if args.expr else coerce(args.value)
        except Exception as error:  # noqa: BLE001 - one bad record must not stop the run
            print(f"  ERROR in {path.parent.name}: {type(error).__name__}: {error}")
            errors += 1
            continue
        parent[key] = value
        written.append(value)

    if not written:
        continue
    where = f"{len(written)}x " if len(written) > 1 else ""
    if args.apply:
        save(path, doc)
        print(f"  {path.parent.name}: wrote {where}{args.path} = {written[0]!r}")
    else:
        print(f"  [preview] {path.parent.name}: would write {where}{args.path} = {written[0]!r}")
    updated += 1

summary = [f"{updated} file(s)"]
if skipped:
    summary.append(f"{skipped} entr(ies) skipped")
if errors:
    summary.append(f"{errors} error(s)")
print(f"Done. ({', '.join(summary)})" + ("" if args.apply else " — preview, nothing written"))