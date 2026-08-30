"""Score a campaign: how much each run improved on its initial design, and how fast.

Reads the experiment map and reduces every run to a few numbers. Prints one row per run
and one summary row per arm, and writes the same figures to gain.json. Undefined values
are null there, not NaN.

Two gain.json are written, in two different places, and neither lands inside a run's own
directory - see the SELF_CONTAINED block below for why a run's own score belongs in the
workspace's per-run cache (workspace.gain_cache_dir, keyed by that run alone, fingerprinted
the same way build_experiment_map.map_stamp fingerprints a whole selection) rather than
beside its records, and the report assembled after it for why the aggregate, selection-wide
figures belong in --out-dir instead.

The metric is the hypervolume, or the best value when one objective is given.

COLUMNS

  gamma       improvement over the initial design at convergence, in %:
              100 * (m_c - m_initial) / |m_initial|. The relative gain, and the headline
              number - the metric only grows, so this is a floor on what the run
              eventually reached rather than what it reached at the end.
  gamma_budget
              the same measured at the end of the budget instead of at convergence.
              gamma <= gamma_budget always, and on a converged run they agree to within
              the flatness threshold - the plateau gamma is measured at the start of is
              flat by construction. A run that never converged has only this one.
  gamma_norm  how much of the distance from the initial design to the known optimum was
              covered: 0 = no better than the design, 1 = optimum reached
  rho_c       m_c as a fraction of the optimum: 1 means the run reached it. Needs a
              known optimum; blank without one.
  regret_c    how far short of the optimum the run stopped, in the metric's own units.
              Reaches 0 at the optimum. Needs a known optimum.
  eta         gamma divided by the evaluations spent reaching it, in % per evaluation.
              Both halves are measured at convergence, so it is the rate the run gained
              at while it was still gaining, not an average over a budget it spent part
              of sitting still.
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
  n_c         evaluation at which the run stopped improving - the start of the plateau
              it finished on (--patience/--tol). Counted from the run's first evaluation,
              so the initial design is included. Blank for a run still improving when its
              budget ran out.
  eps         the flatness threshold this run was judged against - equal to --tol, or
              --tol-rel times this run's own HV(n0)

CONVERGENCE, AND RUNS THAT NEVER GET THERE

  n_c is where the plateau the run *finished* on begins - found by walking back from the
  last evaluation for as long as each step improved by less than the threshold. The
  plateau has to be at least --patience iterations long to count as one, and has to start
  past the initial design (a design is drawn blind, so its metric is routinely flat for
  several points together, and a plateau running back into it would report the run as
  having stopped improving before the optimizer proposed anything).

  Backwards, and not forwards, because forwards cannot tell a plateau from an ending: the
  first flat stretch wins, so a run that goes quiet and then finds something is called
  converged at the quiet patch, and everything after it vanishes from m_c, gamma and
  n_tau. Searching back from the end there is nothing to confuse - the terminal plateau is
  the only one considered, and it is by definition the last time the run improved.

  A run still improving in its final --patience iterations has no terminal plateau, and
  gets no n_c, m_c, gamma or eta at all. This is the run whose budget ran out before it
  settled, and the budget's end is not a convergence point - it is where the money
  stopped. Reporting it as one is what let a censored run be averaged in with finished
  ones and drag the arm's mean down invisibly. What such a run does still have is
  gamma_budget: how much it had gained when the budget ended, which is a real measurement
  and a lower bound on what it would have reached.

  The per-arm summary therefore averages gamma, eta and m_c over the converged runs only,
  and prints "converged: 7/10" beside them. gamma_budget is averaged over all of them,
  since every run has one.

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

  In the per-arm summary, each it column is a median plus how many runs got there at
  all: "15 (1/3)" means one run of three reached that target, after 15 evaluations. The
  others are not averaged in; discarding them would make an optimizer that usually fails
  look fast. The gain columns carry the same qualification whenever a run is missing one
  - see the convergence section above.

READING THEM

  gamma is the headline number but it blows up when the initial design scores near zero
  (a common case: its front dominates almost nothing). Prefer rho_c, which stays in 0..1
  and says how much of the optimum the run actually reached - but it needs a known
  optimum. gamma_norm needs one too, and answers a different question: not how close to
  the optimum the run got, but how much of the distance left open by its initial design
  it closed.

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

from pybo_gui.configs import workspace
from pybo_gui.configs.settings import data_path
from pybo_gui.modules.bayesian_campaign_analysis._constraints import (
    ConstraintError, is_feasible, parse_constraints,
)
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import run_gain_path
from pybo_gui.modules.bayesian_campaign_analysis._convergence import (
    terminal_plateau,
)
from pybo_gui.modules.bayesian_campaign_analysis._hypervolume import (
    hypervolume_nd, pareto_front_nd,
)
from pybo_gui.modules.bayesian_campaign_analysis._labels import arm_label, is_initial
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map

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
                    help="Reference optimum m*. Needed for gamma_norm, rho_c, regret_c "
                         "and the it columns to mean anything across campaigns. Given "
                         "none, it is looked for in optimum.json (campaign_optimum) and "
                         "then in the problem's declared max_hv/best_value, before "
                         "falling back to the best value this campaign reached - which "
                         "is not comparable with another campaign's.")
parser.add_argument("--ground-truth", default="", dest="ground_truth",
                    help="Path to the run's objective.py. When given, the hypervolume "
                         "metric is measured from each objective's own declared "
                         "ref_point instead of a corner padded past whatever runs are "
                         "in this report - the latter moves every run's score if the "
                         "selection changes, which makes gamma/eta incomparable "
                         "between one run of this script and the next.")
parser.add_argument("--true-objective", action="store_true", default=False,
                    dest="true_objective",
                    help="Score each observation by the noiseless objective at the "
                         "parameters it used, instead of the value the run recorded. "
                         "Needs --ground-truth, and only means anything on a simulated "
                         "campaign. m* is an optimum of the true surface, so on a noisy "
                         "problem a front of recorded values beats it and rho_c comes out "
                         "above 1 - this puts both ends of that comparison on the same "
                         "surface. Pass it to plot_hypervolume too, or the table and the "
                         "curve are scoring different things.")
parser.add_argument("--input-feasible", action="store_true", default=False,
                    dest="input_feasible",
                    help="Drop from the front every observation whose parameters break "
                         "the problem's own input constraints. Needs --ground-truth. Such "
                         "a point is recorded like any other but the problem would never "
                         "have allowed it, and m* never sees one, so a handful can double "
                         "a campaign's hypervolume. Counted as an evaluation still, the "
                         "way a --constraint violation is.")
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
# Where each run lives, so its score can be fingerprinted and cached under run_gain_path.
# Absent on a map built before build_experiment_map recorded it, and then that run simply
# gets no cached score.
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
        # Kept so --true-objective / --input-feasible can ask the problem about this
        # observation. Empty for a record carrying none, which both passes then leave be.
        "parameters": exp.get("parameters") or {},
    })

if not runs:
    print("No data with the requested keys.")
    sys.exit(1)

# ---- WHAT THE PROBLEM SAYS ABOUT THESE RECORDS ----
# Both passes run here, on single observations, before any trace is built: replacing a
# point's coordinates or striking it off the front has to happen while it is still one
# observation. Flattened across runs so the objective is loaded and evaluated once for the
# whole campaign rather than once per run.
if args.true_objective or args.input_feasible:
    if not args.ground_truth:
        print("--true-objective and --input-feasible need --ground-truth: only the "
              "problem can say what an observation truly measured, or whether it was "
              "allowed at all.")
        sys.exit(2)
    flat = [item for items in runs.values() for item in items]
    parameters = [item["parameters"] for item in flat]
    if args.true_objective:
        from pybo_gui.modules.bayesian_campaign_analysis._problem_view import true_results
        truth = true_results(args.ground_truth, parameters, objective_keys)
        replaced = 0
        for item, values in zip(flat, truth):
            if values is None:
                continue
            item["point"] = tuple(s * values[k] for s, k in zip(signs, objective_keys))
            replaced += 1
        print(f"--true-objective: {replaced} of {len(flat)} observations re-read off the "
              f"noiseless objective.")
    if args.input_feasible:
        from pybo_gui.modules.bayesian_campaign_analysis._problem_view import input_feasible
        allowed = input_feasible(args.ground_truth, parameters)
        dropped = 0
        for item, ok in zip(flat, allowed):
            # `is False`, not a falsy test: None means the row carried no parameters to
            # check, and treating that as a violation would strike off every reference
            # measurement in a campaign that has any.
            if ok is False:
                item["feasible"] = False
                dropped += 1
        print(f"--input-feasible: {dropped} of {len(flat)} observations break the "
              f"problem's input constraints and contribute no point.")

# ---- REFERENCE POINT ----
# The objective's own ref_point, when available - fixed regardless of which runs are in
# this report, unlike a corner padded past the loaded data's own range (below). A run's
# score read from two different --constraint/selection combinations has to be the same
# number, or it is not measuring the run - it is measuring the report.
ref = None
problem = None
ref_from_problem = False
if args.ground_truth:
    # Imported here, not at the top: a pybo objective is a torch object, so this line
    # costs five seconds of import - and every run of this script that does not ask for a
    # ground truth was paying it for nothing. Same reasoning as _ground_truth's own lazy
    # botorch import, and as _hypervolume being extracted torch-free in the first place.
    from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import (
        load_objective, problem_definition)
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
        ref_from_problem = True

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

# Where the corner above actually came from, which is not the same question as whether
# --ground-truth was passed: a problem that names no ref_point for one of the objectives
# falls through to the data-derived corner having been asked for the other. Everything
# that turns on the reference being fixed - whether a declared max_hv can be believed,
# whether a cached score stays valid when the selection changes - has to read this rather
# than the flag.
reference_source = "ground-truth" if (args.ground_truth and problem is not None
                                      and ref_from_problem) else "selection"


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


def declared_optimum():
    """m* from the problem, or from a cached estimate of it, or None.

    Tried in order of how much each is worth. A number given on the command line is not
    tried here at all - it wins outright. Then:

    * optimum.json, written by campaign_optimum from a dense sample of the true objective.
      Only when its context matches this report's: an HV* measured over other objectives,
      other senses or another reference point is a number from a different problem, and
      dividing by it would be worse than having none. `signs` is compared as a list of
      floats because JSON has no integer/float distinction to rely on.
    * the problem's own declared max_hv (or best_value on one objective), which is a
      literal an author put there. It carries no record of the reference point it was
      measured from, so it is only trusted when this report uses the problem's own - the
      case the --ground-truth branch above already established.

    Both are skipped silently when absent; the caller reports which source won.
    """
    # Beside the objective, where campaign_optimum leaves it: HV* belongs to the problem
    # rather than to any campaign run against it, and the objective's path is the one thing
    # a reader always has. Failing that, the report's own directory, for an estimate that
    # was deliberately put somewhere else with --out-dir.
    candidates = []
    if args.ground_truth:
        candidates.append(os.path.dirname(os.path.abspath(args.ground_truth)))
    candidates.append(args.out_dir or data_path)

    cached, path = None, None
    for directory in candidates:
        path = os.path.join(directory, "optimum.json")
        try:
            with open(path, encoding="utf-8") as file:
                cached = json.load(file)
            break
        except (OSError, ValueError):
            continue

    if cached:
        context_now = {"objectives": objective_keys, "signs": [float(s) for s in signs],
                       "reference": [float(v) for v in ref]}
        stored = cached.get("context") or {}
        matches = (stored.get("objectives") == context_now["objectives"]
                   and [float(v) for v in stored.get("signs", [])] == context_now["signs"]
                   and [float(v) for v in stored.get("reference", [])] == context_now["reference"])
        if matches and cached.get("hv_star") is not None:
            return float(cached["hv_star"]), f"optimum.json ({path})"
        if not matches:
            print(f"! {path} was computed for a different objective set, sense or "
                  f"reference point - ignoring it.")
    if args.ground_truth and reference_source == "ground-truth":
        key = "best_value" if len(ref) == 1 else "max_hv"
        value = problem.get(key) if problem is not None else None
        if value is not None:
            if len(ref) == 1:
                # A best_value is in the objective's own units, while the single-objective
                # trace is -min(sign * raw) - the same "larger is better" space the
                # hypervolume already lives in. -sign is what carries the declared value
                # into it, for a minimized objective and a maximized one alike.
                value = -signs[0] * float(value)
            return float(value), f"the problem's declared {key}"
    return None, None


# One optimum for the campaign, not one per run: measured against its own final value
# every run would score gamma_norm = 1, which says nothing. Failing every declared source,
# the best any run reached stands in, so the others are read as a fraction of that.
if args.optimum is not None:
    optimum, optimum_source = float(args.optimum), "--optimum"
else:
    optimum, optimum_source = declared_optimum()
    if optimum is None:
        optimum = float(np.nanmax([v for trace in traces.values() for v in trace]))
        optimum_source = "the best value reached in this campaign"
# Whether m* came from outside this selection. rho_c and regret_c are absolute statements
# about a run - how much of the optimum it reached, how far short it stopped - so an m*
# that is merely the best run in the report cannot support them: that run would score
# rho_c = 1 by construction and every other would be graded against whichever seed
# happened to do well. gamma_norm and the it columns carry the same caveat and are left
# defined regardless, for continuity with the reports written before this.
known_optimum = optimum_source != "the best value reached in this campaign"

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

    # Where the run stopped improving, or None if it never did - see _convergence for why
    # the plateau is looked for from the end rather than from the start, and what a run
    # without one is.
    start = terminal_plateau(m, eps, args.patience, int(n0))
    converged = start is not None
    n_c = float(start + 1) if converged else np.nan

    # The metric at convergence. On a converged run the plateau is flat by construction, so
    # m_c is within patience * eps of m_final - which is the point. They part company only
    # when the run never settled, and then there is no m_c at all rather than the budget's
    # last value standing in for one.
    m_c = m[start] if converged else np.nan
    gain_c = m_c - m0

    gap = optimum - m0
    row = {"run": run, "arm": items[0]["arm"], "n_initial": int(n0),
           "m_initial": m0, "m_final": m_final, "m_c": m_c,
           "gamma": 100.0 * gain_c / abs(m0) if np.isfinite(m0) and m0 != 0 else np.nan,
           "gamma_budget": 100.0 * (m_final - m0) / abs(m0) if np.isfinite(m0) and m0 != 0 else np.nan,
           "gamma_norm": (m_final - m0) / gap if np.isfinite(gap) and gap > 0 else np.nan,
           # The two the paper draws as traces, at the one point a table can hold: where
           # the run stopped improving. rho_c is that fraction of the optimum reached and
           # regret_c what it still fell short by, so they answer "how good is this front"
           # where gamma answers "how much did the optimizer add". Both are meaningless
           # against an optimum that is merely the best this campaign happened to reach -
           # every run would then be scored against the luckiest one - so they are left
           # blank unless the optimum came from somewhere outside the selection.
           "rho_c": m_c / optimum if known_optimum and optimum else np.nan,
           "regret_c": optimum - m_c if known_optimum else np.nan,
           # None rather than the budget's end when the run never settled - see the
           # backward scan above. A count that is not a convergence point must not be
           # readable as one.
           "n_c": int(n_c) if converged else np.nan, "converged": converged,
           # The threshold this run was actually judged against. With --tol-rel it is
           # this run's own, so the shared setting alone does not say what was applied.
           "eps": eps}
    # Undefined without an n_c to divide by, which is the honest answer for a run that
    # never stopped improving: it has no rate-to-convergence, not a slow one.
    row["eta"] = row["gamma"] / (n_c - n0) if converged and n_c > n0 else np.nan
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
source = optimum_source

print(f"\nMetric: {metric_name} over {', '.join(objective_keys)}")
print(f"Optimum m* = {optimum:.6g}, from {source}.")
if not known_optimum:
    print("  ! No declared optimum: targets are relative to the best this campaign "
          "reached,\n    so gamma_norm and it_tau are not comparable against another "
          "campaign's numbers, and rho_c/regret_c are blank.\n"
          "    Run campaign_optimum to measure m* from the problem itself.")
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


def summarise(column, total):
    """mean +- std over the values that exist, and how many that was.

    The count is not decoration. gamma, eta, m_c and the two optimum-relative columns are
    undefined for a run that never converged, and pandas drops those silently - so without
    it, an arm where three runs of ten settled reports a mean that looks like the arm's,
    and reads identically to one where all ten did. It is the same qualification the
    it_tau columns already carry, for the same reason.

    A single value has no sample spread to report, and "+- nan" would suggest the
    calculation failed rather than that there was nothing to calculate.
    """
    values = column.dropna()
    if values.empty:
        return f"- (0/{total})"
    spread = "" if len(values) < 2 else f" +- {values.std(ddof=1):.3g}"
    count = "" if len(values) == total else f" ({len(values)}/{total})"
    return f"{values.mean():.4g}{spread}{count}"


agg, arms = [], []
for arm, g in per_run.groupby("arm", sort=False):
    total = len(g)
    row = {"arm": arm, "runs": total,
           "converged": f"{int(g['converged'].sum())}/{total}",
           # Over the converged runs alone - a run with no n_c has no gain-at-convergence
           # and no rate to reach it, and averaging the budget's end in as though it were
           # one is what this reporting exists to prevent.
           "gamma": summarise(g["gamma"], total),
           "n_c": summarise(g["n_c"], total),
           "eta": summarise(g["eta"], total),
           # Over every run, converged or not: each one reached the end of its budget, so
           # each one has a gain measured there.
           "gamma_budget": summarise(g["gamma_budget"], total),
           "gamma_norm": summarise(g["gamma_norm"], total)}
    # Only where m* came from outside the selection. Without that they are all-NaN
    # columns, and a column of "nan +- nan" in every row reads as a broken report rather
    # than as a metric that was not available.
    if known_optimum:
        row["rho_c"] = summarise(g["rho_c"], total)
        row["regret_c"] = summarise(g["regret_c"], total)
    entry = {"arm": arm, "runs": total, "converged": int(g["converged"].sum()),
             "targets": {}}
    for name in ("gamma", "gamma_budget", "gamma_norm", "rho_c", "regret_c", "eta", "n_c"):
        # `n` alongside the mean: for the convergence-dependent columns it is the number of
        # runs that converged, not the number in the arm, and a reader of the JSON has no
        # other way to tell the two apart.
        values = g[name].dropna()
        entry[name] = {"mean": g[name].mean(), "std": g[name].std(ddof=1),
                       "n": int(values.size)}
    for tau in taus:
        col = g[f"it{tau:g}"]
        hit = int(col.notna().sum())
        row[f"it{tau:g}"] = f"{col.median():.4g} ({hit}/{len(g)})" if hit else f"- (0/{len(g)})"
        # n_tau is defined for every run that gained anything, so it needs no reached
        # count the way it_tau does - but it still gets a mean +- std beside the median,
        # since the median alone hides how spread out the arm's runs are.
        own = g[f"n{tau:g}"]
        own_valid = own.dropna()
        if own_valid.empty:
            row[f"n{tau:g}"] = "-"
        else:
            spread = "" if len(own_valid) < 2 else f" +- {own_valid.std(ddof=1):.3g}"
            row[f"n{tau:g}"] = f"{own_valid.median():.4g} (mean {own_valid.mean():.4g}{spread})"
        prop = g[f"prop{tau:g}"]
        row[f"prop{tau:g}"] = f"{prop.median():.4g}" if prop.notna().any() else "-"
        # reached is what makes the median readable: a median over 1 of 5 runs is not the
        # same claim as a median over 5.
        entry["targets"][f"{tau:g}"] = {
            "median": col.median(), "reached": hit, "total": len(g),
            "n_tau_median": own.median(), "n_tau_mean": own_valid.mean() if not own_valid.empty else np.nan,
            "n_tau_std": own_valid.std(ddof=1) if len(own_valid) >= 2 else np.nan,
            "prop_tau_median": prop.median()}
    entry["runs_detail"] = g.to_dict("records")
    agg.append(row)
    arms.append(entry)

print("\nPer arm - mean +- std, it_tau as median (reached/total), "
      "n_tau as median (mean +- std):\n")
print(pd.DataFrame(agg).to_string(index=False))

report = {"metric": metric_name, "objectives": objective_keys,
          "optimum": optimum, "optimum_source": source,
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
           "reference_source": reference_source,
           "patience": args.patience, "tol": args.tol, "tol_rel": args.tol_rel,
           "taus": taus}

# The self-contained half of a run's score, cached per-run (see run_gain_path) rather
# than written into the run itself. gamma_norm and it_tau are deliberately not here: both
# normalise by the best the whole campaign reached, so they mean nothing on their own and
# would go stale the moment a different set of runs was scored. They stay in the
# campaign-level report below.
SELF_CONTAINED = ("run", "arm", "n_initial", "m_initial", "m_final", "m_c",
                  "gamma", "gamma_budget", "n_c", "converged", "eta", "eps")
written = 0
skipped_no_dir = 0
cached = workspace.gain_cache_dir() is not None
for row in rows:
    run_dir = run_dirs.get(row["run"])
    if not run_dir or not os.path.isdir(run_dir):
        skipped_no_dir += 1
        continue
    gain_path = run_gain_path(run_dir, create=True)
    payload = {key: row[key] for key in SELF_CONTAINED if key in row}
    for tau in taus:
        # Both counts of the same instant: the whole budget, and the optimizer's
        # own spend within it.
        payload[f"n{tau:g}"] = row.get(f"n{tau:g}")
        payload[f"prop{tau:g}"] = row.get(f"prop{tau:g}")
    payload["context"] = context
    with open(gain_path, "w", encoding="utf-8") as file:
        json.dump(jsonable(payload), file, indent=2, allow_nan=False)
    written += 1
where = "the workspace cache" if cached else "a temporary cache (no workspace configured)"
print(f"\nWrote {written} per-run score{'' if written == 1 else 's'} to {where}.")
if skipped_no_dir:
    print(f"! {skipped_no_dir} run(s) had no directory recorded in the map - "
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