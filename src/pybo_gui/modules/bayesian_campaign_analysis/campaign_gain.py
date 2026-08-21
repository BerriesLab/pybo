"""Score a campaign: how much each run improved on its initial design, and how fast.

Reads the experiment map and reduces every run to a few numbers. Prints one row per run
and one summary row per arm, and writes the same figures to gain.json. Undefined values
are null there, not NaN.

The metric is the hypervolume, or the best value when one objective is given.

COLUMNS

  gamma       improvement over the initial design at the end of the budget, in %
  gamma_c     the same, but measured at convergence rather than at the end of the
              budget: 100 * (m_c - m_initial) / |m_initial|. The metric only grows, so
              gamma_c <= gamma, and they part company by whatever the run found after it
              had stopped counting as improving.
  gamma_norm  how much of the distance from the initial design to the known optimum was
              covered: 0 = no better than the design, 1 = optimum reached
  eta         gamma divided by the evaluations spent getting there, in % per evaluation
  it0.5       evaluations, past the initial design, needed to cover 50% of the distance
  it0.9       to the optimum; likewise 90% and 99%. Set your own targets with --tau.
  it0.99
  n0.5        evaluations from the start of the run to cover 50% of that run's own gain
  n0.9        to convergence; likewise 90% and 99%. Counted from the first evaluation, so
  n0.99       the initial design is included - the whole experimental budget.
  prop0.5     the same instant, counted as the optimizer's own spend: n_tau - n_initial.
  prop0.9     Reported beside n_tau because the two separate what the experiment cost
  prop0.99    from what the search cost. A large design with a quick search and a small
              design with a slow one can share an n_tau and differ entirely in this.
  m_c         the metric at convergence
  n_c         evaluation at which the run stopped improving (--patience/--tol), counted
              from the run's first evaluation, so the initial design is included
  eps         the flatness threshold this run was judged against - equal to --tol, or
              --tol-rel times this run's own HV(n0)

CONVERGENCE, AND WHAT IT CANNOT SEE

  n_c is the end of the first run of --patience consecutive iterations that all improved
  by less than the threshold, searched only past the initial design (a design is drawn
  blind, so its metric is routinely flat for several points together, and a window
  overlapping it would stop the run before the optimizer proposed anything).

  It is a *first* match, so it cannot tell a plateau apart from an ending. A run that
  sits flat for longer than --patience and then climbs again is called converged at the
  plateau, and everything after it is invisible to m_c, gamma_c and n_tau - while gamma,
  measured at the end of the budget, still sees it. A large gap between gamma and gamma_c
  is the symptom, and the cure is a --patience longer than the longest plateau the
  experiment plausibly has.

TWO FINISH LINES

  it_tau and n_tau both answer "how fast", against different targets, and neither is
  wrong - they say different things.

  it_tau is a fraction of the distance to the optimum: a shared finish line, so runs and
  arms are directly comparable, and a run that never gets there simply has none. Counted
  past the initial design, since the design is not the optimizer's work.

  n_tau is a fraction of that run's own gain to convergence, counted from the run's
  first evaluation. Every run that gained anything reaches it, by construction at n_c at
  the latest, so it is always defined - but it is self-referential: a run that barely
  improved reaches 90% of its own small gain quickly and scores well. prop_tau is that
  same instant minus n_initial, and is not a third finish line - the target is identical,
  only the origin of the count differs.

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
from pybo_gui.modules.bayesian_campaign_analysis._labels import arm_label, is_initial
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
                    help="Improvement below which an iteration counts as flat, in the "
                         "metric's own units (default: %(default)s). Absolute, so it has "
                         "to be re-tuned for every campaign whose hypervolume lives on a "
                         "different scale - see --tol-rel.")
parser.add_argument("--tol-rel", type=float, default=None,
                    help="--tol as a fraction of the run's own starting metric instead: "
                         "1e-4 means 'flat' is an improvement below 0.01%% of HV(n0). "
                         "Takes precedence over --tol when given, and adapts to each "
                         "run's scale, so one setting reads the same across campaigns. "
                         "Falls back to --tol for a run starting from zero, which no "
                         "fraction can measure.")
parser.add_argument("--out-dir", default=None,
                    help="Where gain.json is written (default: the campaign directory). "
                         "The GUI builds its map in a scratch directory that does not "
                         "outlive the session, so it passes the campaign root here "
                         "instead - a score is worth keeping, unlike the map it was "
                         "computed from, which is rebuilt from the selection each time.")
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
# Where each run lives, so its score can be written back beside it. Absent on a map
# built before build_experiment_map recorded it, and then that run simply gets no file.
run_dirs = {}
for exp in load_experiments_from_map(MAP_PATH):
    results = exp.get("results", {})
    raw = tuple(results.get(k) for k in objective_keys)
    if any(v is None for v in raw):
        continue
    if exp.get("run_dir"):
        run_dirs.setdefault(exp["run"], exp["run_dir"])
    runs.setdefault(exp["run"], []).append({
        "point": tuple(s * v for s, v in zip(signs, raw)),
        # Kept in the sequence even when it fails a constraint, and left out of the front
        # instead (see metric_trace). An infeasible point is an outcome, not a choice: the
        # initial design is drawn at random, so whether a point turns out feasible is
        # known only after it has been measured - and measured is what it cost. Dropping
        # it here would shorten the run, shrink the design size it reports, and make
        # n_c * hours-per-evaluation understate the machine time actually spent.
        "feasible": is_feasible(results, constraints),
        "initial": is_initial(exp["experiment_type"]),
        # Both optimizer (the arm) and provenance (experimental/synthetic), so the
        # per-arm summary below never pools a real run in with a simulated one just
        # because they happen to share an optimizer.
        "arm": arm_label(exp, exp["run"]),
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
        # Feasible points only: the corner has to bound the space the front lives in, and
        # an infeasible outlier would push it out without any front ever reaching there.
        values = [r["point"][d] for items in runs.values() for r in items if r["feasible"]]
        lo, hi = min(values), max(values)
        span = hi - lo
        ref.append(hi + (REF_MARGIN * span if span > 0 else (abs(hi) * REF_MARGIN or 1.0)))
    ref = tuple(ref)


def metric_trace(items):
    """The metric after each observation of one run, in maximization space.

    Hypervolume for several objectives; for one, the best value so far - negated into
    the same "larger is better" space so every comparison below reads the same way.

    One entry per observation the run made, feasible or not: an infeasible measurement
    moves the trace along without improving it, which is exactly what it did to the run.
    That keeps every index here a count of measurements, so n_initial, n_c and n_tau all
    mean evaluations spent rather than evaluations that happened to pass.
    """
    seen, trace = [], []
    for item in items:
        if item["feasible"]:
            seen.append(item["point"])
        if not seen:
            # Nothing feasible yet: no front, so nothing is dominated. -inf for a single
            # objective is the same statement - no best value has been found - and the
            # gamma guards below already read a non-finite start as no gain to report.
            trace.append(float("-inf") if len(ref) == 1 else 0.0)
        elif len(ref) == 1:
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
    #
    # Only windows lying entirely past the initial design are considered. A design is
    # drawn blind, so its hypervolume is routinely flat for several points together -
    # a window overlapping it reads that flatness as convergence and stops the run
    # before the optimizer has proposed anything, which then reports m_c == m_initial
    # and a gain of zero.
    # --tol-rel reads the threshold off this run's own starting point, so one setting
    # means the same thing on a campaign whose hypervolume is 300 and on one where it is
    # 3. A run starting from zero has no scale to take a fraction of, and keeps --tol.
    eps = args.tol
    if args.tol_rel is not None and np.isfinite(m0) and m0 != 0:
        eps = args.tol_rel * abs(m0)

    n_c, converged = n[-1], False
    for i in range(int(n0) + args.patience - 1, len(m)):
        window = np.diff(m[i - args.patience + 1:i + 1])
        if window.size and np.all(np.abs(window) < eps):
            n_c, converged = n[i], True
            break

    # The metric at convergence, which is what gamma_c and n_tau are measured against.
    # Distinct from m_final: the run keeps going to the end of its budget, and the
    # hypervolume only ever grows, so m_final >= m_c whenever anything was found after
    # the run stopped counting as improving.
    m_c = m[int(n_c) - 1]
    gain_c = m_c - m0

    gap = optimum - m0
    row = {"run": run, "arm": items[0]["arm"], "n_initial": int(n0),
           "m_initial": m0, "m_final": m_final, "m_c": m_c,
           "gamma": 100.0 * (m_final - m0) / abs(m0) if np.isfinite(m0) and m0 != 0 else np.nan,
           "gamma_c": 100.0 * gain_c / abs(m0) if np.isfinite(m0) and m0 != 0 else np.nan,
           "gamma_norm": (m_final - m0) / gap if np.isfinite(gap) and gap > 0 else np.nan,
           "n_c": int(n_c), "converged": converged,
           # The threshold this run was actually judged against. With --tol-rel it is
           # this run's own, so the shared setting alone does not say what was applied.
           "eps": eps}
    row["eta"] = row["gamma"] / (n_c - n0) if n_c > n0 else np.nan
    for tau in taus:
        # Censored on purpose: a run that never clears the target has no it_tau, and the
        # aggregate below reports how many did rather than averaging the rest.
        reached = np.flatnonzero(m >= m0 + tau * gap) if np.isfinite(gap) and gap > 0 else []
        row[f"it{tau:g}"] = n[reached[0]] - n0 if len(reached) else np.nan
        # n_tau against this run's own gain to convergence, counted from the start of the
        # run rather than from the end of its design. Always defined when the run gained
        # anything, since the target is by construction reached at n_c at the latest.
        own = np.flatnonzero(m >= m0 + tau * gain_c) if gain_c > 0 else []
        row[f"n{tau:g}"] = n[own[0]] if len(own) else np.nan
        # The same instant counted as the optimizer's own spend. n_tau is the whole
        # experimental budget, design included, which is what a reader pays for; this is
        # what the search cost on top of it. Reported side by side because a campaign
        # with a large design and a quick search, and one with a small design and a slow
        # search, can land on the same n_tau while being entirely different experiments.
        row[f"prop{tau:g}"] = row[f"n{tau:g}"] - n0
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
print(f"Convergence: {args.patience} iterations improving by less than "
      + (f"{args.tol_rel:g} of each run's own HV(n0).\n" if args.tol_rel is not None
         else f"{args.tol:g}.\n"))
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
           "gamma_c": f"{g['gamma_c'].mean():.4g} +- {g['gamma_c'].std(ddof=1):.3g}",
           "gamma_norm": f"{g['gamma_norm'].mean():.4g} +- {g['gamma_norm'].std(ddof=1):.3g}",
           "eta": f"{g['eta'].mean():.4g} +- {g['eta'].std(ddof=1):.3g}",
           "converged": f"{int(g['converged'].sum())}/{len(g)}"}
    entry = {"arm": arm, "runs": len(g), "converged": int(g["converged"].sum()),
             "targets": {}}
    for name in ("gamma", "gamma_c", "gamma_norm", "eta"):
        entry[name] = {"mean": g[name].mean(), "std": g[name].std(ddof=1)}
    for tau in taus:
        col = g[f"it{tau:g}"]
        hit = int(col.notna().sum())
        row[f"it{tau:g}"] = f"{col.median():.4g} ({hit}/{len(g)})" if hit else f"- (0/{len(g)})"
        # n_tau is defined for every run that gained anything, so it is a plain median
        # over the arm - no reached count to qualify it the way it_tau needs one.
        own = g[f"n{tau:g}"]
        row[f"n{tau:g}"] = f"{own.median():.4g}" if own.notna().any() else "-"
        prop = g[f"prop{tau:g}"]
        row[f"prop{tau:g}"] = f"{prop.median():.4g}" if prop.notna().any() else "-"
        # reached is what makes the median readable: a median over 1 of 5 runs is not the
        # same claim as a median over 5.
        entry["targets"][f"{tau:g}"] = {"median": col.median(), "reached": hit,
                                        "total": len(g), "n_tau_median": own.median(),
                                        "prop_tau_median": prop.median()}
    entry["runs_detail"] = g.to_dict("records")
    agg.append(row)
    arms.append(entry)

print("\nPer arm - mean +- std, it_tau as median (reached/total), n_tau as median:\n")
print(pd.DataFrame(agg).to_string(index=False))

report = {"metric": metric_name, "objectives": objective_keys,
          "optimum": args.optimum, "optimum_source": source,
          "convergence": {"patience": args.patience, "tol": args.tol,
                          "tol_rel": args.tol_rel},
          "taus": taus, "arms": arms}
# What a per-run score is only valid under. Two runs scored against different reference
# points, objectives or convergence settings are not comparable, and the plots that read
# these files back check this before combining them.
context = {"objectives": objective_keys,
           "signs": list(signs),
           "constraints": sorted(args.constraint),
           "reference": list(ref),
           # A reference taken from the problem definition is fixed, so a run scored
           # against it stays comparable with any other. One derived from the loaded data
           # is not: it moves with the selection, which makes a cached score valid only
           # for the selection it was computed in.
           "reference_source": "ground-truth" if args.ground_truth else "selection",
           "patience": args.patience, "tol": args.tol, "tol_rel": args.tol_rel,
           "taus": taus}

# The self-contained half of a run's score, written beside the run itself. gamma_norm and
# it_tau are deliberately not here: both normalise by the best the whole campaign reached,
# so they mean nothing on their own and would go stale the moment a different set of runs
# was scored. They stay in the campaign-level report below.
SELF_CONTAINED = ("run", "arm", "n_initial", "m_initial", "m_final", "m_c",
                  "gamma", "gamma_c", "n_c", "converged", "eta", "eps")
written = 0
for row in rows:
    run_dir = run_dirs.get(row["run"])
    if not run_dir or not os.path.isdir(run_dir):
        continue
    payload = {key: row[key] for key in SELF_CONTAINED if key in row}
    for tau in taus:
        # Both counts of the same instant: the whole budget, and the optimizer's
        # own spend within it.
        payload[f"n{tau:g}"] = row.get(f"n{tau:g}")
        payload[f"prop{tau:g}"] = row.get(f"prop{tau:g}")
    payload["context"] = context
    with open(os.path.join(run_dir, "gain.json"), "w", encoding="utf-8") as file:
        json.dump(jsonable(payload), file, indent=2, allow_nan=False)
    written += 1
print(f"\nWrote {written} per-run score{'' if written == 1 else 's'} "
      f"(gain.json beside each run).")
if written < len(rows):
    print(f"! {len(rows) - written} run(s) had no directory recorded in the map - "
          f"rebuild it so build_experiment_map writes run_dir.")
if not args.ground_truth:
    print("! The reference point was derived from the runs loaded now, so these scores "
          "hold only for this selection. Pass --ground-truth to fix it.")

out_dir = args.out_dir or data_path
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "gain.json")
with open(out_path, "w", encoding="utf-8") as file:
    json.dump(jsonable(report), file, indent=2, allow_nan=False)
print("\nSaved", out_path)