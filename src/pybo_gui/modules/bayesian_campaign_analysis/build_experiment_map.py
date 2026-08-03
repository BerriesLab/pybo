"""Build the experiment map the plot scripts read, from a campaign's step records.

The plot scripts were written for a rig where one experiment is one measured point with a
flat dict of results. A pybo observation is exactly that, so this walks the selected steps
and emits one record per observation.

`group_id` is the observation index, so a group holds the same evaluation count across
replicate runs - which is what makes --grouped mean "mean +- std over seeds at that
point in the campaign", the same shape of statement the original made about repeated
settings.

`experiment_type` carries the observation's provenance ("initial" / "proposed") because
that is what the scripts label, colour and draw separate exploration fronts by;
`technology` carries the arm that produced it ("bayesian" / "sobol").
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


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


def build_map(roots) -> dict:
    """One record per observation, in chronological order."""
    records = []
    for path in find_steps(roots):
        step_dir = path.parent
        record = json.loads(path.read_text(encoding="utf-8"))
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
                "group_id":        observation.get("observation_n"),
                "experiment_id":   f"{step_dir.parent.name}/{step_dir.name}"
                                   f"#{observation.get('observation_n')}",
                "experiment_type": observation.get("source"),
                "technology":      record.get("experiment_type"),
                "start_time":      _epoch(record.get("datetime")),
                "run":             step_dir.parent.name,
                "path":            str(step_dir),
                "parameters":      observation.get("parameters") or {},
                "results":         results,
            })

    records.sort(key=lambda r: (r["start_time"] or 0.0, r["group_id"] or 0))
    for i, record in enumerate(records, start=1):
        record["index"] = i
    return {"experiments": records}


def build_and_save(out_dir, roots=None) -> Path:
    """Write experiment_map.json under `out_dir`, over `roots` (default: out_dir)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_map(roots or [out_dir])
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
    args = parser.parse_args()
    build_and_save(args.out_dir, args.step or None)


if __name__ == "__main__":
    main()
