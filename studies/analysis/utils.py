"""Shared loading for the campaign-analysis scripts.

These read what a finished run left on disk: each trial's ``summary.json``,
which carries the problem definition alongside the observations.

Discovery globs ``**/summary.json`` under each ``--study`` root, which matches both a
study (``replicateN_seedM/summary.json``) and the single summary a tutorial run writes.
One root means the series are its replicates; several mean the series are the studies,
with replicates aggregated inside each. Every script follows that one rule, which is what
lets them serve a single study and a comparison of studies without branching.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

_SEED_RE = re.compile(r"seed(\d+)")


@dataclass
class Trial:
    """One finished run: its summary, and where it sat in the study.
    Note: a trial is "whatever" produced a "summary.json", i.e. a
    replicate inside a study, or a single run of a standalone tutorial run."""
    study: str  # series label when several studies are compared
    name: str  # the run directory's name, e.g. replicate0_seed2063
    seed: int | None  # parsed from the name; series label within one study
    path: Path  # the path to that study
    summary: dict  # the experiment summary file as dict

    @property
    def n_initial(self) -> int:
        return self.summary["config"]["n_initial_samples"] or 0

    @property
    def objectives(self) -> list:
        return self.summary["problem"]["objectives"]

    @property
    def parameters(self) -> list:
        return self.summary["problem"]["parameters"]


def discover_trials(roots: list[Path], labels: list[str] | None = None) -> list[Trial]:
    """Every trial under each root, labeled by study.

    `labels` is matched positionally to `roots`; a root without one is labeled by its
    directory name.
    """
    labels = list(labels or [])
    trials = []
    for i, root in enumerate(roots):
        root = Path(root)
        study = labels[i] if i < len(labels) else root.name
        found = sorted(root.glob("**/summary.json"))
        if not found:
            raise SystemExit(f"No summary.json under {root} — is that a run or study directory?")
        for path in found:
            summary = json.loads(path.read_text(encoding="utf-8"))
            name = path.parent.name
            match = _SEED_RE.search(name)
            trials.append(Trial(study=study, name=name,
                                seed=int(match.group(1)) if match else None,
                                path=path, summary=summary))
    return trials


def series_key(trials: list[Trial]) -> str:
    """Which column separates the lines: studies when several are given, else replicates.

    Returning the column name rather than the values keeps every script's grouping to one
    `groupby(series_key(...))`.
    """
    return "study" if len({t.study for t in trials}) > 1 else "run"


def metric_frame(trials: list[Trial]) -> pd.DataFrame:
    """Long frame of the per-iteration metric: one row per (trial, iteration).

    The metric is hypervolume for a multi-objective problem and best value for a single
    one — chosen by which key `metrics` holds. Regret needs a known optimum, so it is NaN
    for problems that declare none.
    """
    rows = []
    for t in trials:
        metrics = t.summary["metrics"]
        problem = t.summary["problem"]
        if "hypervolume" in metrics:
            name, values, optimum = "hypervolume", metrics["hypervolume"], problem.get("max_hv")
        else:
            name, values, optimum = "best_value", metrics["best_values"], problem.get("best_value")
        for i, (value, elapsed) in enumerate(zip(values, metrics["elapsed_time"])):
            rows.append({
                "study": t.study, "run": t.name, "seed": t.seed,
                "iteration": i + 1,
                # The x axis readers care about is evaluations, not loop steps: the
                # initial design is already on the board before iteration 1.
                "evaluations": t.n_initial + i + 1,
                "metric": name,
                "value": value,
                "regret": (optimum - value) if optimum is not None else np.nan,
                "elapsed_time": elapsed,
                "n_initial": t.n_initial,
            })
    return pd.DataFrame(rows)


def observation_frame(trials: list[Trial]) -> pd.DataFrame:
    """Long frame of the observations: one row per (trial, observation).

    Parameters, objectives and constraints become columns named by their labels from the
    problem definition. `is_initial` marks the rows that came from the initial design
    rather than from the optimizer.
    """
    rows = []
    for t in trials:
        obs = t.summary["observations"]
        X = obs.get("X") or []
        Y_obj = obs.get("Y_obj") or []
        Y_con = obs.get("Y_con") or []
        par_labels = [p["label"] for p in t.parameters]
        obj_labels = [o["label"] for o in t.objectives]
        con_labels = [c["label"] for c in t.summary["problem"].get("constraints", [])]
        for i, x in enumerate(X):
            row = {"study": t.study, "run": t.name, "seed": t.seed,
                   "obs_index": i, "is_initial": i < t.n_initial}
            row.update(dict(zip(par_labels, x)))
            if i < len(Y_obj):
                row.update(dict(zip(obj_labels, Y_obj[i])))
            if i < len(Y_con):
                row.update(dict(zip(con_labels, Y_con[i])))
            rows.append(row)
    return pd.DataFrame(rows)


def objective_labels(trials: list[Trial]) -> list[str]:
    return [o["label"] for o in trials[0].objectives]


def parameter_labels(trials: list[Trial]) -> list[str]:
    return [p["label"] for p in trials[0].parameters]


def minimized(trials: list[Trial]) -> dict:
    """label -> True when the objective is minimized. Read from the problem, so no
    --maximize flag is needed."""
    return {o["label"]: bool(o["to_minimize"]) for o in trials[0].objectives}
