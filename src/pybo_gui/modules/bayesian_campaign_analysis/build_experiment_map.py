"""Build the experiment map the plot scripts read, from a campaign's step records.

The plot scripts were written for a rig where one experiment is one measured point with a
flat dict of results. A pybo observation is exactly that, so this walks the selected steps
and emits one record per observation.

Each record carries where it sits in its run and which arm produced it; build_group_map
turns that pair into the group_id the plots aggregate by, so a group holds the same
evaluation count of the same arm across replicate runs. That is what makes --grouped mean
"mean +- std over seeds at that point in the campaign", the same shape of statement the
original made about a repeated setting.

`experiment_type` is what the scripts label, colour and draw a separate Pareto front by,
so what goes into it decides what a front is drawn per - see LABEL_BY. `optimizer`
carries the arm ("bayesian" / "sobol" / "random", or the technology name on a baseline
run that no optimizer proposed) whatever the labelling, so nothing else loses it.

That "experiment_type" is a map-level field this module computes, not to be confused
with the step record's own "experiment_type" key, which means something else again -
real rig data vs a simulated trial. The record's arm lives under "optimizer" instead.
Both of the record's fields survive into the map, as `optimizer` and `provenance`.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


# What an observation is labelled by, and so what gets its own series and Pareto front.
LABEL_BY = ("run", "strategy", "strategy+run", "provenance")
DEFAULT_LABEL_BY = "run"

# Marks a label as the initial design of whatever it is qualifying. The plots read it to
# style those series as exploration - dotted front, the design's marker - and to pair them
# with the proposals they belong to.
INITIAL_SUFFIX = " (initial)"

# What a label falls back to when the dimension it is drawn from is not recorded. The
# plot scripts already spell an unrecorded arm this way when they read one themselves.
UNKNOWN_LABEL = "unknown"


def _label_for(observation: dict, optimizer, run: str, label_by: str) -> str:
    """The label an observation carries, under the chosen dimension.

    The initial design is a series of its own *within* the dimension, not across it: a
    run's design is labelled separately from that run's proposals, but two runs never
    share one design series. That is what lets a plot show, per run, what its design
    alone reached against what the arm found afterwards.

    A record may leave the chosen dimension unset - a row no optimizer proposed and no
    campaign designed carries neither, honestly - and it is named UNKNOWN_LABEL rather
    than None: a label is what a series is keyed, sorted and coloured by, and none of
    that works on a null.
    """
    source = observation.get("source")
    if label_by == "provenance":
        # The dimension already is the provenance, so there is nothing to qualify.
        return source or UNKNOWN_LABEL
    if label_by == "strategy":
        base = optimizer or UNKNOWN_LABEL
    elif label_by == "strategy+run":
        base = f"{optimizer or UNKNOWN_LABEL}/{run}"
    else:
        base = run
    return f"{base}{INITIAL_SUFFIX}" if source == "initial" else base


def _epoch(stamp):
    """Seconds since the epoch, from the ISO string a record carries.

    The plot scripts pass start_time to datetime.fromtimestamp, so the map has to hold a
    number. Converting here keeps them unmodified.
    """
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (TypeError, ValueError):
        return None


def find_steps(roots) -> list:
    """Every step record under each root, sorted.

    A root may be a step directory, a run directory or a whole study: the glob matches at
    any depth, so one call serves each level of the layout.
    """
    found = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.append(root)
        else:
            found.extend(sorted(root.glob("**/experiment.json")))
    return found


def build_map(roots, label_by: str = DEFAULT_LABEL_BY) -> dict:
    """One record per observation, in chronological order.

    `label_by` decides what each observation is labelled by, and so what gets its
    own series and Pareto front downstream.
    """
    records = []
    for path in find_steps(roots):
        step_dir = path.parent
        record = json.loads(path.read_text(encoding="utf-8"))
        run = step_dir.parent.name
        # The step record's own "experiment_type" means something else entirely (real
        # rig data vs a simulated trial - see "provenance" below); the arm comes from
        # "optimizer" instead.
        optimizer = record.get("optimizer")
        for observation in record.get("data", []):
            # Values only. A record always carries a <label>_var partner and leaves it
            # null when the run measured no noise, and an unmeasured column is not
            # something a plot can put on an axis.
            results = {}
            for group in ("objectives", "constraints", "trackers"):
                results.update({label: value
                                for label, value in (observation.get(group) or {}).items()
                                if value is not None})
            records.append({
                "iteration":       step_dir.name,
                # Where this sits in its run. build_group_map turns it, together with the
                # arm, into the group_id the plots aggregate by.
                "observation":     observation.get("observation_n"),
                "experiment_id":   f"{step_dir.parent.name}/{step_dir.name}"
                                   f"#{observation.get('observation_n')}",
                "experiment_type": _label_for(observation, optimizer, run, label_by),
                "source":          observation.get("source"),
                "optimizer":       optimizer,
                # What produced the measurement - the rig, the problem - as opposed to
                # what chose where to measure. None on a record that does not name one.
                "technology":      record.get("technology"),
                # Real rig data vs a simulated trial - the step record's own
                # "experiment_type", orthogonal to optimizer (which arm proposed the
                # point) and to source (initial vs proposed). Missing on a record
                # written before this field existed; callers that care read that as
                # unknown rather than assuming either.
                "provenance":      record.get("experiment_type"),
                "start_time":      _epoch(record.get("datetime")),
                "run":             run,
                "path":            str(step_dir),
                "parameters":      observation.get("parameters") or {},
                "results":         results,
            })

    records.sort(key=lambda r: (r["start_time"] or 0.0, r["observation"] or 0))
    for i, record in enumerate(records, start=1):
        record["index"] = i
    return {"experiments": records}


def build_and_save(out_dir, roots=None, label_by: str = DEFAULT_LABEL_BY) -> Path:
    """Write experiment_map.json under `out_dir`, over `roots` (default: out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_map(roots or [out_dir], label_by)
    out_path = out_dir / "experiment_map.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {len(result['experiments'])} observations -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Scan pybo step records and build a chronological experiment index.")
    parser.add_argument("out_dir", help="Where experiment_map.json is written")
    parser.add_argument("--step", action="append", default=[],
                        help="A step, run or study directory (repeatable). "
                             "Defaults to out_dir.")
    parser.add_argument("--label-by", choices=LABEL_BY, default=DEFAULT_LABEL_BY,
                        dest="label_by",
                        help="What an observation is labelled by, and so what gets its "
                             "own series and Pareto front (default: %(default)s).")
    args = parser.parse_args()
    build_and_save(args.out_dir, args.step or None, args.label_by)


if __name__ == "__main__":
    main()
