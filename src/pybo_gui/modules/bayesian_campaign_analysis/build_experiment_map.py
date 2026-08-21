"""Build the experiment map the plot scripts read, from a campaign's step records.

The plot scripts were written for a rig where one experiment is one measured point with a
flat dict of results. A pybo observation is exactly that, so this walks the selected steps
and emits one record per observation.

Each record carries the parameters it was taken at, the run it belongs to, where it sits
in that run and which arm produced it. Two different things are built on those, and they
are easy to mistake for one another:

  build_group_map keys a group on the run *and the parameters*, snapped to the rig's
  resolutions - so a group is one setting within one run, and --grouped averages the
  repeats of a setting. Its error bars measure measurement spread: a noisy objective, or
  a setting deliberately repeated with --repeats.

  The aggregate view (--aggregate-runs, see _aggregate) keys on the *arm* instead and
  aligns runs by evaluation index, so it averages whole runs of one arm against each
  other. Its band measures how differently the optimizer behaves from one seed to the
  next, and the parameters play no part in it.

The two compose rather than compete: with both on, repeats are averaged into a point and
then runs into a curve.

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
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


# What an observation is labelled by, and so what gets its own series and Pareto front.
LABEL_BY = ("run", "strategy", "strategy+run", "provenance")
DEFAULT_LABEL_BY = "run"

# A sweep study (see studies/variability_study.py) names a run
# "<strategy>_ninit<n>_replicate<k>_seed<s>" when --n-initial was swept - study
# metadata read back from that name, not anything the optimizer itself records, so a
# run this wasn't written by (or one since renamed) simply has none.
_NINIT_RE = re.compile(r"ninit(\d+)")

# A trial writes one step directory per repetition, "step_<i>_rep<r>". Counting an
# initial design means counting each point once, so only repetition 00 is read; a
# layout with no repetition in its names (a ported rig campaign, whose directories
# are the rig's own experiment ids) has nothing to skip and is counted whole.
_REP_RE = re.compile(r"_rep(\d+)$")

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


def _count_initial(run_dir: Path) -> int | None:
    """How many points a run measured as its initial design, counted from its records.

    Counted rather than declared, so it holds for a run nothing wrote a size into: a
    tutorial run by hand, or a rig campaign ported from the machine's own files. This is
    what OptimizerBase.n_initial_samples does for a live run, over the records instead.

    None, not 0, when the run records no provenance at all - a baseline whose rows name
    no source has an unknown initial design, not an empty one.
    """
    total = 0
    saw_source = False
    for path in sorted(run_dir.glob("*/experiment.json")):
        repetition = _REP_RE.search(path.parent.name)
        if repetition is not None and int(repetition.group(1)) != 0:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for observation in record.get("data", []):
            source = observation.get("source")
            if source is not None:
                saw_source = True
            if source == "initial":
                total += 1
    return total if saw_source else None


# The shape of a record written below. Anything cached against an older number was
# built by code that wrote different fields, and no stamp taken from the data can see
# that - adding run_dir, for instance, left existing maps without it and the plots that
# had come to expect it failed on them. Bump this whenever a record gains, loses or
# renames a field.
MAP_SCHEMA = 1


def map_stamp(roots, references=(), label_by: str = DEFAULT_LABEL_BY,
              resolutions: dict | None = None) -> dict:
    """Everything the built map depends on, small enough to compare.

    Statting is cheap exactly where reading is not: opening these records costs tens of
    milliseconds apiece on a scanned filesystem, while the directory walk and the stat
    behind it are a fraction of a second for thousands of them. So a stamp costs well
    under a second where the build it can skip costs a minute.

    Per record rather than per directory: a directory's mtime does not move when a file
    inside a subdirectory is rewritten, and scoring a campaign writes gain.json into a run
    without touching any step record. The (mtime, size) pair of each record is what
    actually tracks the inputs.
    """
    records = []
    for path in find_steps(roots):
        try:
            info = path.stat()
        except OSError:
            # Vanished between the walk and the stat: leaving it out is what a rebuild
            # would see anyway, and a stamp that cannot be taken twice the same way is
            # worse than one that simply reflects what is there.
            continue
        records.append((str(path), info.st_mtime_ns, info.st_size))
    records.sort()
    return {
        "schema": MAP_SCHEMA,
        "roots": sorted(str(Path(root).resolve()) for root in roots),
        "references": sorted(str(Path(root).resolve()) for root in references),
        "label_by": label_by,
        "resolutions": {str(k): v for k, v in sorted((resolutions or {}).items())},
        "n_records": len(records),
        "records": hashlib.sha256(repr(records).encode("utf-8")).hexdigest(),
    }


def stamp_digest(stamp: dict) -> str:
    """A short name for a stamp, for the directory a map is cached under."""
    return hashlib.sha256(
        json.dumps(stamp, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_map(roots, label_by: str = DEFAULT_LABEL_BY, reference_roots=None) -> dict:
    """One record per observation, in chronological order.

    `label_by` decides what each observation is labelled by, and so what gets its
    own series and Pareto front downstream.

    `reference_roots` marks the user's benchmark: a step directory under any of
    these (or one of these itself) is tagged "reference", so a plot can draw it
    apart from the ordinary runs it is being compared against. Independent of
    `roots` - a reference directory not otherwise selected still needs its
    observations in the map, so callers are expected to fold it into `roots`
    themselves (the GUI unions checked_paths and reference_paths before calling
    this); find_steps is not asked to do that union on their behalf.
    """
    reference_roots = [Path(root).resolve() for root in (reference_roots or [])]

    def _is_reference(step_dir: Path) -> bool:
        step_dir = step_dir.resolve()
        return any(step_dir == root or root in step_dir.parents
                  for root in reference_roots)

    records = []
    # One count per run, not per record: every step of a run shares the run's initial
    # design, and _count_initial reads the whole run directory to work it out.
    counted_initial: dict[Path, int | None] = {}
    for path in find_steps(roots):
        step_dir = path.parent
        record = json.loads(path.read_text(encoding="utf-8"))
        run_dir = step_dir.parent
        run = run_dir.name
        if run_dir not in counted_initial:
            counted_initial[run_dir] = _count_initial(run_dir)
        # The step record's own "experiment_type" means something else entirely (real
        # rig data vs a simulated trial - see "provenance" below); the arm comes from
        # "optimizer" instead.
        optimizer = record.get("optimizer")
        # Three ways to know a run's initial design, most trustworthy first: a size the
        # record itself declares (older data, written while the trial scripts recorded
        # one), the size counted from the run's own observations, and finally the
        # --n-initial a sweep study spelled into the run directory name (see _NINIT_RE).
        # Counting covers what the other two cannot: a run nobody declared a size for and
        # no study named, which is every hand-run trial and every ported rig campaign.
        if record.get("n_initial") is not None:
            n_initial = record["n_initial"]
        elif counted_initial[run_dir] is not None:
            n_initial = counted_initial[run_dir]
        else:
            n_initial_match = _NINIT_RE.search(run)
            n_initial = int(n_initial_match.group(1)) if n_initial_match else None
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
                # Where the run lives, not just what it is called. Two studies can hold
                # a run of the same name, and anything writing back beside a run (see
                # campaign_gain's per-run score) needs the path rather than the label.
                "run_dir":         str(step_dir.parent.resolve()),
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
                # The sweep's own --n-initial, read back from the run name a study
                # gave it - see _NINIT_RE. None on a run no sweep named this way.
                "n_initial":       n_initial,
                "path":            str(step_dir),
                "reference":       _is_reference(step_dir),
                "parameters":      observation.get("parameters") or {},
                "results":         results,
            })

    records.sort(key=lambda r: (r["start_time"] or 0.0, r["observation"] or 0))
    for i, record in enumerate(records, start=1):
        record["index"] = i
    return {"experiments": records}


def build_and_save(out_dir, roots=None, label_by: str = DEFAULT_LABEL_BY,
                   reference_roots=None) -> Path:
    """Write experiment_map.json under `out_dir`, over `roots` (default: out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_map(roots or [out_dir], label_by, reference_roots)
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
    parser.add_argument("--reference", action="append", default=[],
                        dest="reference_roots",
                        help="A step, run or study directory to mark as the "
                             "benchmark (repeatable). Must also be covered by "
                             "--step to appear in the map at all.")
    args = parser.parse_args()
    build_and_save(args.out_dir, args.step or None, args.label_by,
                   args.reference_roots or None)


if __name__ == "__main__":
    main()
