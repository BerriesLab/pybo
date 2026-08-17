"""Score a campaign: how much each run improved on its initial design, and how fast.

Reads the experiment map and reduces every run to a few numbers. Prints one row per run
and one summary row per arm, and writes the same figures to gain.json. Undefined values
are null there, not NaN.

The metric is the hypervolume, or the best value when one objective is given.

COLUMNS

  gamma       improvement over the initial design, in %
  gamma_norm  how much of the distance from the initial design to the known optimum was
              covered: 0 = no better than the design, 1 = optimum reached
  eta         gamma divided by the evaluations spent getting there, in % per evaluation
  it0.5       evaluations, past the initial design, needed to cover 50% of that distance;
  it0.9       likewise 90% and 99%. Set your own targets with --tau.
  it0.99
  n_c         evaluation at which the run stopped improving (--patience/--tol)

  In the per-arm summary, gains are mean +- std over that arm's runs and each it column
  is a median plus how many runs got there at all: "15 (1/3)" means one run of three
  reached that target, after 15 evaluations. The others are not averaged in; discarding
  them would make an optimizer that usually fails look fast.

READING THEM

  gamma is the headline number but it blows up when the initial design scores near zero
  (a common case: its front dominates almost nothing). Prefer gamma_norm, which stays in
  0..1 - but it needs a known optimum, from --optimum.

  eta and it_tau both measure speed, against different finish lines. eta uses the run's
  own stopping point, so a run that stalls early looks fast; it_tau uses a target you
  chose, and a run that stalls simply never reaches it. Trust it_tau.

  With no --optimum the targets are relative to the best this campaign reached, which
  makes them incomparable with another campaign's numbers.

EXAMPLES

  python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_gain --x Branin --y Currin
  python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_gain --objective Branin \
      --objective Currin --tau 0.9 --tau 0.99 --optimum 59.36
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from pybo_gui.configs.settings import data_path
from pybo_gui.modules.bayesian_campaign_analysis._constraints import (
    ConstraintError, is_feasible, parse_constraints,
)
from pybo_gui.modules.bayesian_campaign_analysis._hypervolume import (
    hypervolume_nd, pareto_front_nd,
)
from pybo_gui.modules.bayesian_campaign_analysis._labels import is_initial
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective, problem_definition

REF_MARGIN = 0.10  # reference point sits this fraction of the data range beyond the worst

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--x", default="", help="Result key for the first objective.")
parser.add_argument("--y", default="", help="Result key for the second objective.")
parser.add_argument("--z", default="", help="Result key for a third objective.")
parser.add_argument("--objective", action="append", default=[],
                    help="Result key for an objective (repeatable). Overrides --x/--y/--z.")
parser.add_argument("--maximize", action="append", default=[],
                    help="Result key of an objective to maximize (repeatable). "
                         "Default: minimize.")
parser.add_argument("--constraint", action="append", default=[],
                    help="Feasibility constraint (repeatable). Infeasible observations "
                         "never contribute.")
parser.add_argument("--tau", action="append", default=[], type=float,
                    help="Target as a fraction of the achievable gap (repeatable, "
                         "default: 0.5 0.9 0.99).")
parser.add_argument("--optimum", type=float, default=None,
                    help="Reference optimum m*. Needed for gamma_norm and the it columns "
                         "to mean anything across campaigns.")
parser.add_argument("--ground-truth", default="", dest="ground_truth",
                    help="Path to the run's objective.py. When given, the hypervolume "
                         "metric is measured from each objective's own declared "
                         "ref_point instead of a corner padded past whatever runs are "
                         "in this report - the latter moves every run's score if the "
                         "selection changes, which makes gamma/eta incomparable "
                         "between one run of this script and the next.")
parser.add_argument("--patience", type=int, default=10,
                    help="Iterations of improvement below --tol that mark convergence "
                         "(default: %(default)s, as OptimizerBase.is_converged).")
parser.add_argument("--tol", type=float, default=1e-3,
                    help="Improvement below which an iteration counts as flat "
                         "(default: %(default)s).")
args = parser.parse_args()

try:
    constraints = parse_constraints(args.constraint)
except ConstraintError as exc:
    print(exc)
    sys.exit(2)

objective_keys = args.objective or [k for k in (args.x, args.y, args.z) if k]
if not objective_keys:
    print("Give at least one objective, with --x/--y or --objective.")
    sys.exit(2)
# Minimization space, so a maximized objective improves downwards like the rest.
signs = [-1.0 if k in set(args.maximize) else 1.0 for k in objective_keys]
taus = sorted(args.tau or [0.5, 0.9, 0.99])

MAP_PATH = os.path.join(data_path, "experiment_map.json")

# ---- LOAD, ONE SEQUENCE PER RUN ----
runs = {}
for exp in load_experiments_from_map(MAP_PATH):
    results = exp.get("results", {})
    raw = tuple(results.get(k) for k in objective_keys)
    if any(v is None for v in raw) or not is_feasible(results, constraints):
        continue
    runs.setdefault(exp["run"], []).append({
        "point": tuple(s * v for s, v in zip(signs, raw)),
        "initial": is_initial(exp["experiment_type"]),
        "technology": exp.get("technology"),
    })

if not runs:
    print("No data with the requested keys.")
    sys.exit(1)

# ---- REFERENCE POINT ----
# The objective's own ref_point, when available - fixed regardless of which runs are in
# this report, unlike a corner padded past the loaded data's own range (below). A run's
# score read from two different --constraint/selection combinations has to be the same
# number, or it is not measuring the run - it is measuring the report.
ref = None
if args.ground_truth:
    problem = problem_definition(load_objective(args.ground_truth))
    ref_by_label = {o["label"]: o["ref_point"] for o in problem["objectives"]}
    missing = [k for k in objective_keys if k not in ref_by_label]
    unset = [k for k in objective_keys
             if k in ref_by_label and ref_by_label[k] is None]
    if missing or unset:
        print(f"! --ground-truth {args.ground_truth}: "
              + (f"no objective named {missing} - " if missing else "")
              + (f"no ref_point declared for {unset} - " if unset else "")
              + "falling back to a reference point derived from the loaded data, "
                "which will move if the selection changes.")
    else:
        ref = tuple(s * ref_by_label[k] for s, k in zip(signs, objective_keys))

if ref is None:
    # Over every run currently loaded, so at least this report is internally
    # consistent - but see the ground_truth branch above for why that is not enough to
    # make its numbers comparable with another run of this script.
    ref = []
    for d in range(len(objective_keys)):
        values = [r["point"][d] for items in runs.values() for r in items]
        lo, hi = min(values), max(values)
        span = hi - lo
        ref.append(hi + (REF_MARGIN * span if span > 0 else (abs(hi) * REF_MARGIN or 1.0)))
    ref = tuple(ref)


def metric_trace(items):
    """The metric after each observation of one run, in maximization space.

    Hypervolume for several objectives; for one, the best value so far - negated into
    the same "larger is better" space so every comparison below reads the same way.
    """
    seen, trace = [], []
    for item in items:
        seen.append(item["point"])
        if len(ref) == 1:
            trace.append(-min(p[0] for p in seen))
        else:
            trace.append(hypervolume_nd(pareto_front_nd(seen), ref))
    return trace


traces = {run: metric_trace(items) for run, items in runs.items()}
# One optimum for the campaign, not one per run: measured against its own final value
# every run would score gamma_norm = 1, which says nothing. With no --optimum the best
# any run reached stands in, so the others are read as a fraction of that.
optimum = (args.optimum if args.optimum is not None
           else float(np.nanmax([v for trace in traces.values() for v in trace])))

rows = []
for run, items in runs.items():
    trace = traces[run]
    n_initial = sum(1 for item in items if item["initial"])
    if n_initial == 0 or n_initial >= len(trace):
        # Nothing was proposed after the design, so there is no gain to report.
        continue
    m = np.asarray(trace, dtype=float)
    n = np.arange(1, len(trace) + 1, dtype=float)
    # The design is one block: the run's starting point is the metric once all of it is in.
    m0, n0, m_final = m[n_initial - 1], float(n_initial), m[-1]

    # Convergence as the run itself would have called it: `patience` consecutive
    # improvements all below `tol`. Never converged means the horizon is the budget,
    # which makes eta a lower bound rather than a missing value.
    n_c, converged = n[-1], False
    for i in range(args.patience - 1, len(m)):
        window = np.diff(m[i - args.patience + 1:i + 1])
        if window.size and np.all(np.abs(window) < args.tol):
            n_c, converged = n[i], True
            break

    gap = optimum - m0
    row = {"run": run, "arm": items[0]["technology"], "n_initial": int(n0),
           "m_initial": m0, "m_final": m_final,
           "gamma": 100.0 * (m_final - m0) / abs(m0) if np.isfinite(m0) and m0 != 0 else np.nan,
           "gamma_norm": (m_final - m0) / gap if np.isfinite(gap) and gap > 0 else np.nan,
           "n_c": int(n_c), "converged": converged}
    row["eta"] = row["gamma"] / (n_c - n0) if n_c > n0 else np.nan
    for tau in taus:
        # Censored on purpose: a run that never clears the target has no it_tau, and the
        # aggregate below reports how many did rather than averaging the rest.
        reached = np.flatnonzero(m >= m0 + tau * gap) if np.isfinite(gap) and gap > 0 else []
        row[f"it{tau:g}"] = n[reached[0]] - n0 if len(reached) else np.nan
    rows.append(row)

if not rows:
    print("No run has both an initial design and proposals after it.")
    sys.exit(1)

per_run = pd.DataFrame(rows)
metric_name = "best value" if len(ref) == 1 else "hypervolume"
source = "--optimum" if args.optimum is not None else "the best value reached in this campaign"

print(f"\nMetric: {metric_name} over {', '.join(objective_keys)}")
print(f"Optimum m*, from {source}.")
if args.optimum is None:
    print("  ! No declared optimum: targets are relative to the best this campaign "
          "reached,\n    so gamma_norm and it_tau are not comparable against another "
          "campaign's numbers.")
print(f"Convergence: {args.patience} iterations improving by less than {args.tol:g}.\n")
print(per_run.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


def jsonable(value):
    """numpy scalars unwrapped and non-finite floats as None, for a strict parser."""
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


agg, arms = [], []
for arm, g in per_run.groupby("arm", sort=False):
    row = {"arm": arm, "runs": len(g),
           "gamma": f"{g['gamma'].mean():.4g} +- {g['gamma'].std(ddof=1):.3g}",
           "gamma_norm": f"{g['gamma_norm'].mean():.4g} +- {g['gamma_norm'].std(ddof=1):.3g}",
           "eta": f"{g['eta'].mean():.4g} +- {g['eta'].std(ddof=1):.3g}",
           "converged": f"{int(g['converged'].sum())}/{len(g)}"}
    entry = {"arm": arm, "runs": len(g), "converged": int(g["converged"].sum()),
             "targets": {}}
    for name in ("gamma", "gamma_norm", "eta"):
        entry[name] = {"mean": g[name].mean(), "std": g[name].std(ddof=1)}
    for tau in taus:
        col = g[f"it{tau:g}"]
        hit = int(col.notna().sum())
        row[f"it{tau:g}"] = f"{col.median():.4g} ({hit}/{len(g)})" if hit else f"- (0/{len(g)})"
        # reached is what makes the median readable: a median over 1 of 5 runs is not the
        # same claim as a median over 5.
        entry["targets"][f"{tau:g}"] = {"median": col.median(), "reached": hit,
                                        "total": len(g)}
    entry["runs_detail"] = g.to_dict("records")
    agg.append(row)
    arms.append(entry)

print("\nPer arm - mean +- std, it_tau as median (reached/total):\n")
print(pd.DataFrame(agg).to_string(index=False))

report = {"metric": metric_name, "objectives": objective_keys,
          "optimum": args.optimum, "optimum_source": source,
          "convergence": {"patience": args.patience, "tol": args.tol},
          "taus": taus, "arms": arms}
out_path = os.path.join(data_path, "gain.json")
with open(out_path, "w", encoding="utf-8") as file:
    json.dump(jsonable(report), file, indent=2, allow_nan=False)
print("\nSaved", out_path)