import argparse
import json
import math
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.modules.bayesian_campaign_analysis._hypervolume import (
    hypervolume_nd, pareto_front_nd,
)
from pybo_gui.modules.bayesian_campaign_analysis._labels import (
    arm_label, base_label, is_initial, styler)
from pybo_gui.modules.bayesian_campaign_analysis._aggregate import (
    BAND_MODES, mean_band, arm_legend_label)
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map
from pybo_gui.modules.bayesian_campaign_analysis._constraints import parse_constraints, is_feasible, ConstraintError

parser = argparse.ArgumentParser()
parser.add_argument("--x", default="down_time_minutes", help="Result key for first objective")
parser.add_argument("--y", default="wear_microns",      help="Result key for second objective")
parser.add_argument("--z", default="",                  help="Result key for third objective (empty = 2D hypervolume)")
parser.add_argument("--objective", action="append", default=[],
                    help="Result key for an objective (repeatable). When given, "
                         "overrides --x/--y/--z and enables N-D hypervolume. Give "
                         "exactly one and the metric becomes the best value so far, "
                         "which is what a single-objective campaign converges on.")
parser.add_argument("--maximize", action="append", default=[],
                    help="Result key of an objective to maximize (repeatable). "
                         "Its values are negated so the front/hypervolume is "
                         "always computed as a minimization. Default: minimize.")
parser.add_argument("--constraint", action="append", default=[],
                    help="Feasibility constraint as key:op:value (repeatable). "
                         "An infeasible experiment contributes no point to the "
                         "hypervolume but still counts as an evaluation, so the curve "
                         "goes flat there rather than losing a step off the x axis.")
parser.add_argument("--grouped", action="store_true", default=False,
                    help="Average replicate experiments per group_id into one point.")
parser.add_argument("--aggregate-runs", action="store_true", default=False,
                    help="Collapse the runs of each arm into one mean curve with a band, "
                         "instead of drawing a curve per run. Runs are matched by step "
                         "and truncated to the shortest.")
parser.add_argument("--band", default="ci95", choices=BAND_MODES,
                    help="What the band around the aggregated mean shows (default: "
                         "%(default)s). ci95 says where the arm's mean lies, so it is the "
                         "one that answers whether two arms differ; std says where a "
                         "single run lands and does not shrink with more replicates.")
parser.add_argument("--improvement", action="store_true", default=False,
                    help="Plot per-step hypervolume improvement (ΔHV) instead of the "
                         "cumulative hypervolume.")
parser.add_argument("--metric", default="hv", choices=("hv", "normalized", "regret"),
                    help="What the y axis measures (default: %(default)s). 'hv' is the "
                         "hypervolume itself. 'normalized' is HV(n)/HV*, which reaches 1 "
                         "at the optimum. 'regret' is HV* - HV(n), which reaches 0 there "
                         "and is drawn on a log axis, so the *rate* a run converges at is "
                         "readable rather than only where it ended up. The last two need "
                         "an optimum: --optimum, or optimum.json from campaign_optimum, "
                         "or the problem's declared max_hv.")
parser.add_argument("--optimum", type=float, default=None,
                    help="HV* for --metric normalized/regret. Given none, optimum.json is "
                         "read from --score-dir when its context matches this plot's, and "
                         "then the problem's declared max_hv is tried.")
parser.add_argument("--score-dir", default=None, dest="score_dir",
                    help="Where to look for the optimum.json campaign_optimum wrote "
                         "(default: the campaign directory). The GUI keeps a selection's "
                         "score in the workspace cache rather than beside the records, so "
                         "it passes that entry here - the campaign directory holds the "
                         "runs, not the reports about them.")
parser.add_argument("--true-objective", action="store_true", default=False,
                    dest="true_objective",
                    help="Score each observation by the noiseless objective at the "
                         "parameters it used, instead of the value the run recorded. "
                         "Needs --ground-truth, and only means anything on a simulated "
                         "campaign. HV* is an optimum of the true surface, so on a noisy "
                         "problem a front of recorded values dominates more than HV* and "
                         "the regret goes negative for most points - this puts both ends "
                         "of that subtraction on the same surface. It changes only the "
                         "score put on the optimizer's choices, never which points it "
                         "chose or how many evaluations it spent.")
parser.add_argument("--input-feasible", action="store_true", default=False,
                    dest="input_feasible",
                    help="Drop from the front every observation whose parameters break "
                         "the problem's own input constraints. Needs --ground-truth. Such "
                         "a point is recorded like any other but the problem would never "
                         "have allowed it, and HV* never sees one - campaign_optimum draws "
                         "through the sampler that rejects them - so a handful can double "
                         "a campaign's hypervolume and put it above the optimum. Treated "
                         "exactly as a --constraint violation is: the evaluation still "
                         "counts, it just contributes no point.")
parser.add_argument("--ground-truth", default="", dest="ground_truth",
                    help="Path to the run's objective.py. When given, the hypervolume "
                         "is measured from each objective's own declared ref_point "
                         "instead of a corner padded past whatever is currently "
                         "selected - the latter moves every trace's value each time a "
                         "run is added to or dropped from the plot, which makes them "
                         "incomparable between one plot and the next.")
args = parser.parse_args()
try:
    constraints = parse_constraints(args.constraint)
except ConstraintError as exc:
    print(exc)
    sys.exit(2)

# Objective keys: explicit --objective list (N-D) or the --x/--y[/--z] trio.
objective_keys = list(args.objective) or [k for k in (args.x, args.y, args.z) if k]
if not objective_keys:
    print("Give at least one objective, with --x/--y or --objective.")
    sys.exit(1)
# One objective has no volume to measure, so the campaign's metric is the best value it
# has reached - the same substitution campaign_gain makes, so the two agree on what a
# single-objective run is scored by. Everything below (per-run traces, phase shading,
# grouping, constraints) is about the campaign rather than the metric, so it is unchanged.
single = len(objective_keys) == 1

# Per-objective optimisation sense: +1 minimize, -1 maximize. Maximized axes are
# negated so the whole pipeline (front + hypervolume) stays a pure minimization.
maximize = set(args.maximize)
signs = [-1.0 if k in maximize else 1.0 for k in objective_keys]

# ---- CONFIG ----
MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path
REF_MARGIN = 0.10  # reference point sits this fraction of the data range beyond the worst
# Technologies excluded from the hypervolume curve: they are reference points, not part
# of the optimisation campaign. Keyed on what produced the measurement rather than on the
# arm that chose it, because a reference run has no arm to be keyed on - the machine's own
# technology, measured as a baseline, records no optimizer at all.
EXCLUDED_TECHNOLOGIES = {"standard", "reference"}

MARKERS      = fig_cfg["markers"]["label"]
LABEL_COLORS = fig_cfg["colors"]["label"]
SCATTER      = fig_cfg["scatter"]

FONT_LABEL  = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]
DPI         = fig_cfg["dpi"]

# Axis-label overrides by result key. The problem definition already names its
# objectives, so this is empty until a key needs a unit or LaTeX in the label.
PRETTY_NAMES = {}

def _label(exp):
    return (exp.get("experiment_type") or exp.get("optimizer") or "unknown").lower()


# ---- LOAD ----
rows = []
for exp in load_experiments_from_map(MAP_PATH):
    # Skip baseline/reference groups — they are not part of the optimisation
    # campaign and must not contribute to the hypervolume improvement curve.
    if str(exp.get("technology") or "").lower() in EXCLUDED_TECHNOLOGIES:
        continue
    r = exp.get("results", {})
    raw = tuple(r.get(k) for k in objective_keys)
    if any(v is None for v in raw):
        continue
    # An infeasible evaluation is kept, flagged rather than dropped. It must not join
    # the Pareto set - nothing it attained is allowed to count - but it did spend a slot
    # of the budget, and dropping it here took that slot off the x axis too. An arm that
    # wastes evaluations on infeasible points then read as though it had never made
    # them, and two arms that waste different numbers were compared over budgets that
    # were not the same length.
    # Negate maximized objectives so the point lives in minimization space.
    point = tuple(s * v for s, v in zip(signs, raw))
    rows.append({
        "label":    _label(exp),
        # The arm the run belongs to, which is what several runs get averaged within.
        # Both optimizer and provenance, not the label: the label is whatever the map
        # was built by, and under the default it names the run, so it tells the runs of
        # one arm apart instead of holding them together.
        "arm":      arm_label(exp, _label(exp)),
        "group_id": exp["group_id"],
        "feasible": is_feasible(r, constraints),
        "point":    point,
        # Kept so --true-objective can re-read this observation off the noiseless
        # surface. Empty for a record that carries none, which that pass then leaves
        # alone.
        "parameters": exp.get("parameters") or {},
    })

if not rows:
    print("No data with the requested keys.")
    sys.exit(1)

# ---- THE NOISELESS SURFACE, WHERE ASKED FOR ----
# Before anything is grouped or made into a front: replacing a point's coordinates has to
# happen while it is still one observation, or a group's mean would be an average of noisy
# values that this then failed to correct.
if args.true_objective:
    if not args.ground_truth:
        print("--true-objective needs --ground-truth: the noiseless value of an "
              "observation can only come from the problem that produced it.")
        sys.exit(2)
    from pybo_gui.modules.bayesian_campaign_analysis._problem_view import true_results
    truth = true_results(args.ground_truth, [r["parameters"] for r in rows], objective_keys)
    replaced = 0
    for row, values in zip(rows, truth):
        if values is None:
            continue
        row["point"] = tuple(s * values[k] for s, k in zip(signs, objective_keys))
        replaced += 1
    print(f"--true-objective: {replaced} of {len(rows)} observations re-read off the "
          f"noiseless objective.")
    if replaced < len(rows):
        print(f"  ! {len(rows) - replaced} recorded no parameters to re-read them at, and "
              f"keep their measured values - so this plot mixes the two.")

# Folded into the same `feasible` flag an output-constraint violation sets, so everything
# downstream - the flat stretch in the curve, the evaluation still counting on the x axis,
# the reference point ignoring it - already does the right thing without knowing which kind
# of infeasibility it was.
if args.input_feasible:
    if not args.ground_truth:
        print("--input-feasible needs --ground-truth: only the problem knows which "
              "parameter vectors it allows.")
        sys.exit(2)
    from pybo_gui.modules.bayesian_campaign_analysis._problem_view import input_feasible
    allowed = input_feasible(args.ground_truth, [r["parameters"] for r in rows])
    dropped = unknown = 0
    for row, ok in zip(rows, allowed):
        if ok is None:
            unknown += 1
        elif not ok:
            row["feasible"] = False
            dropped += 1
    print(f"--input-feasible: {dropped} of {len(rows)} observations break the problem's "
          f"input constraints and contribute no point.")
    if unknown:
        print(f"  ! {unknown} recorded no parameters to check, and are left as they were.")

# In grouped mode, collapse replicate experiments (same group_id) into one point per
# group whose coordinates are the elementwise mean. Label comes from the group's first
# row. Only the feasible members are averaged, so a group's point is never pulled by a
# reading that violated a constraint; a group with no feasible member keeps its slot on
# the x axis and contributes no point.
if args.grouped:
    grouped, order = {}, []
    for r in rows:
        gid = r["group_id"]
        if gid not in grouped:
            grouped[gid] = []
            order.append(gid)
        grouped[gid].append(r)
    rows = [{
        "label": grouped[gid][0]["label"],
        "feasible": any(it["feasible"] for it in grouped[gid]),
        # A group is repeats of one setting within one run, so the whole of it belongs
        # to that run's arm. Carried through, or aggregating would lose the very field
        # it pools by the moment the two options are used together.
        "arm": grouped[gid][0]["arm"],
        "point": tuple(
            sum(it["point"][d] for it in grouped[gid] if it["feasible"])
            / max(1, sum(1 for it in grouped[gid] if it["feasible"]))
            for d in range(len(objective_keys))),
    } for gid in order]

# ---- FIXED REFERENCE POINT ----
# From the feasible points alone: the reference exists to bound the volume the front
# encloses, and a point that violates a constraint is not on any front. Letting one push
# the corner out would inflate every arm's hypervolume by the same wasted evaluation.
feasible_rows = [row for row in rows if row["feasible"]]
if not feasible_rows:
    print("Every observation violates the constraints - nothing to measure a volume "
          "against.")
    sys.exit(1)

# The objective's own ref_point, when available - fixed regardless of what is
# currently selected, unlike a corner padded past the loaded data's own range (below).
# A run's hypervolume read off two different selections has to be the same number, or
# it is not measuring the run - it is measuring the selection.
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
    # Range-based margin so the reference stays "worse" than every point in
    # minimization space, valid for both positive and negated (maximized) axes. Depends
    # on whatever is currently loaded - see the ground_truth branch above for why that
    # makes a run's own value shift as other runs join or leave the selection.
    ref = []
    for d in range(len(objective_keys)):
        lo = min(row["point"][d] for row in feasible_rows)
        hi = max(row["point"][d] for row in feasible_rows)
        span = hi - lo
        pad = REF_MARGIN * span if span > 0 else (abs(hi) * REF_MARGIN or 1.0)
        ref.append(hi + pad)
    ref = tuple(ref)

# ---- INCREMENTAL HYPERVOLUME, ONE TRACE PER SERIES ----
# Each run is its own campaign, so its hypervolume starts from its own first observation
# rather than continuing wherever the previous run left off. A run's initial design and
# its proposals share a trace - they are one campaign - which is what base_label groups.
# The reference point stays global, so the traces are on one scale and comparable.
series = {}
for r in rows:
    series.setdefault(base_label(r["label"]), []).append(r)

traces = {}
series_arm = {}
for name, items in series.items():
    s_steps, s_hvs, s_labels, seen = [], [], [], []
    for spent, r in enumerate(items, start=1):
        # Only a feasible point joins the set the metric is measured on; an infeasible
        # one still advances the count, leaving the curve flat for that evaluation.
        # That flat stretch is the honest picture: the budget was spent and bought
        # nothing.
        if r["feasible"]:
            seen.append(r["point"])
        if not seen:
            # Nothing attained yet, and no volume without a point. float("nan") holds
            # the slot so the x axis still counts the evaluation, and matplotlib leaves
            # the curve unstarted rather than drawing it up from a floor of zero.
            s_hvs.append(float("nan"))
        else:
            # Back out of minimization space for the single-objective trace, so the curve
            # is in the objective's own units and reads against the problem rather than
            # against a reference point that exists only to make a volume finite.
            s_hvs.append(signs[0] * min(p[0] for p in seen) if single
                         else hypervolume_nd(pareto_front_nd(seen), ref))
        s_steps.append(spent)
        s_labels.append(r["label"])
    traces[name] = (s_steps, s_hvs, s_labels)
    # Falls back to the series' own name: a row that reached here without an arm still
    # gets a trace of its own rather than taking a plot down that was not aggregating.
    series_arm[name] = items[0].get("arm") or name

# ---- RESCALE AGAINST THE OPTIMUM ----
# Done here, once, on the traces themselves rather than at each of the three drawing
# branches below: everything downstream - the per-arm averaging, the band, the shading -
# then works on the quantity actually being plotted, and none of it has to know which.
regret_floored = 0
if args.metric != "hv":
    if args.improvement:
        # Both rewrite what a point means, and stacking them would take per-step gains of
        # a quantity that is already a difference from the optimum.
        print("--metric normalized/regret and --improvement do not combine: the first "
              "measures the gap to the optimum, the second the gain over the previous "
              "step. Pick one.")
        sys.exit(2)

    def find_optimum():
        """HV* for this plot, and where it came from, or (None, None).

        Same order as campaign_gain resolves it in, and the same reasons: an explicit
        number, then a cached estimate whose context matches this plot's, then a literal
        the problem declares - which is only believable when this plot is measuring from
        the problem's own reference point, since the literal records no corner of its own.
        """
        if args.optimum is not None:
            return float(args.optimum), "--optimum"
        path = os.path.join(args.score_dir or data_path, "optimum.json")
        try:
            with open(path, encoding="utf-8") as file:
                cached = json.load(file)
        except (OSError, ValueError):
            cached = None
        if cached:
            stored = cached.get("context") or {}
            matches = (stored.get("objectives") == objective_keys
                       and [float(v) for v in stored.get("signs", [])] == list(signs)
                       and [float(v) for v in stored.get("reference", [])] == [float(v) for v in ref])
            if matches and cached.get("hv_star") is not None:
                return float(cached["hv_star"]), f"optimum.json ({path})"
            if not matches:
                print(f"! {path} was computed for a different objective set, sense or "
                      f"reference point - ignoring it.")
        if problem is not None and ref_from_problem:
            key = "best_value" if single else "max_hv"
            value = problem.get(key)
            if value is not None:
                return float(value), f"the problem's declared {key}"
        return None, None

    optimum, optimum_source = find_optimum()
    if optimum is None:
        print(f"--metric {args.metric} measures the gap to the optimum, and no optimum "
              f"is known for this campaign. Compute one with campaign_optimum (it writes "
              f"optimum.json beside the campaign), declare max_hv on the objective, or "
              f"pass --optimum. Drawn against a guess, this curve would be a shape rather "
              f"than a measurement.")
        sys.exit(2)
    print(f"HV* = {optimum:.6g}, from {optimum_source}.")

    for name, (s_steps, s_hvs, s_labels) in traces.items():
        if args.metric == "normalized":
            scaled = [v / optimum if optimum else float("nan") for v in s_hvs]
        else:
            scaled = [optimum - v for v in s_hvs]
        traces[name] = (s_steps, scaled, s_labels)

    if args.metric == "regret":
        # A run can dominate more than the estimated optimum, and then its regret is zero
        # or negative and has no place on a log axis. Two things cause it, and they call
        # for opposite responses, so the count is reported rather than quietly clipped:
        # an HV* that is still climbing when the sampling stopped (raise --samples), and
        # observations carrying noise, whose front sits above the noiseless surface it was
        # drawn from by roughly the noise amplitude (nothing to fix - the run really did
        # record that, and the true front is what it is).
        values = [v for _s, ys, _l in traces.values() for v in ys]
        positive = [v for v in values if v > 0]
        regret_floored = sum(1 for v in values if v == v and v <= 0)
        floor = (min(positive) * 0.1) if positive else 1e-9
        for name, (s_steps, ys, s_labels) in traces.items():
            traces[name] = (s_steps, [v if v > 0 else floor for v in ys], s_labels)
        if regret_floored:
            print(f"! {regret_floored} of {len(values)} points reached or passed HV*, so "
                  f"their regret is not positive and cannot be drawn on a log axis. They "
                  f"are pinned to {floor:.3g} (the dotted line). Either HV* is understated "
                  f"- raise campaign_optimum --samples - or the observations are noisy, "
                  f"and a noisy front legitimately dominates more than the true one.")

all_labels     = sorted({r["label"] for r in rows if r["label"] is not None})

# The runs of each arm, in the order their traces were built, ready to be averaged.
# n_initial travels alongside each curve so the aggregated view can shade its own
# initial-design span, the same distinction the single-campaign view draws.
arm_curves = {}
arm_n_initial = {}
for name, (_s_steps, s_hvs, s_labels) in traces.items():
    arm = series_arm[name]
    arm_curves.setdefault(arm, []).append(s_hvs)
    arm_n_initial.setdefault(arm, []).append(sum(1 for lbl in s_labels if is_initial(lbl)))

if args.aggregate_runs and args.improvement:
    # Both rewrite what a point on the curve means, and stacking them would average
    # per-step gains that were already floored onto a shared log decade.
    raise SystemExit("--aggregate-runs and --improvement do not combine: the first "
                     "averages runs of a cumulative curve, the second replaces that "
                     "curve with per-step gains. Pick one.")

# Arms take a colour of their own when aggregating, so the mean curve is not confused
# with any single run's. Left alone otherwise, since adding names here would shift the
# colours assigned by position to every existing label.
_color, _marker, _line_style, _front_style = styler(
    fig_cfg, all_labels + sorted(arm_curves) if args.aggregate_runs else all_labels)

def y_label():
    """What the y axis is measuring, named once for all three drawing branches.

    A single-objective campaign has no volume, so "hypervolume" is the wrong word for it
    throughout - the metric is the best value reached, and the normalized and regret forms
    of that are the same two questions asked of a value instead of a volume.
    """
    quantity = f"best {objective_keys[0]}" if single else "hypervolume"
    if args.metric == "normalized":
        return f"Normalized {quantity} " + r"$\rho$"
    if args.metric == "regret":
        return f"{quantity.capitalize()} regret " + r"$R(n)$"
    return quantity.capitalize()


# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["hypervolume"])

if len(traces) == 1 and not args.aggregate_runs:
    # One campaign: shade where each phase of it ran, as the single-sequence view did.
    # Initial and proposed share a colour by design (_labels.styler pairs a design with
    # the proposals it belongs to, so marker and dash are what tell them apart there) -
    # two same-coloured spans side by side would just read as one continuous background,
    # so the initial span also gets a hatch and a touch more alpha to stay a region of
    # its own rather than blending into the one after it.
    steps, _hvs, labels = next(iter(traces.values()))
    prev_lbl     = labels[0]
    region_start = steps[0]
    for i in range(1, len(steps) + 1):
        lbl = labels[i] if i < len(labels) else None
        if lbl != prev_lbl or i == len(steps):
            initial = is_initial(prev_lbl)
            ax.axvspan(
                region_start - 0.5, steps[i - 1] + 0.5,
                color=_color(prev_lbl), alpha=0.14 if initial else 0.08,
                hatch="//" if initial else None, linewidth=0,
            )
            if i < len(steps):
                region_start = steps[i]
                prev_lbl     = lbl

legend_handles = [
    mlines.Line2D([], [], linestyle="None", marker=_marker(lbl),
                  markerfacecolor=_color(lbl),
                  markeredgecolor=SCATTER["edge_color"], markeredgewidth=0.8,
                  markersize=SCATTER["marker_size"] ** 0.5, label=lbl.capitalize())
    for lbl in all_labels
]

if args.improvement:
    # Per-step marginal gain HV(k) - HV(k-1), with HV(0) = 0. Drawn as a log-scale
    # scatter, connected by a line for the multi-objective hypervolume; zero-improvement
    # steps are floored to one decade below the smallest positive gain so they pin to a
    # row at the bottom (a dotted line marks that floor). The line stays continuous
    # through the floored points. A single-objective trace has no line: its x-axis
    # already skips every infeasible evaluation (rows were dropped at load time), so
    # connecting what's left would draw a false continuity straight through the gaps
    # those exclusions leave.
    # Per-step marginal gain per trace, on one shared floor so the decades line up.
    per_trace = {}
    for name, (s_steps, s_hvs, s_labels) in traces.items():
        if single:
            # A value, not a volume: a step's gain is how far it moved the best value,
            # measured back in minimization space so an improvement is a decrease
            # whichever way the objective runs. The first observation improves on
            # nothing, so it has no gain and floors like any other flat step.
            best = [signs[0] * v for v in s_hvs]
            deltas = [0.0] + [best[i - 1] - best[i] for i in range(1, len(best))]
        else:
            deltas = [s_hvs[0]] + [s_hvs[i] - s_hvs[i - 1] for i in range(1, len(s_hvs))]
        per_trace[name] = deltas
    pos   = [d for deltas in per_trace.values() for d in deltas if d > 0]
    floor = (min(pos) * 0.1) if pos else 1e-9
    ax.set_yscale("log")
    ax.axhline(floor, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7, zorder=1)
    for name, (s_steps, _s_hvs, s_labels) in traces.items():
        ydraw = [d if d > 0 else floor for d in per_trace[name]]
        if not single:
            ax.plot(s_steps, ydraw, color=_color(name), linewidth=0.8, zorder=2)
        for lbl in sorted(set(s_labels)):
            idx = [i for i, l in enumerate(s_labels) if l == lbl]
            ax.scatter(
                [s_steps[i] for i in idx],
                [ydraw[i] for i in idx],
                s=SCATTER["marker_size"] * (1.0 if single else 0.7), marker=_marker(lbl),
                facecolors=_color(lbl), edgecolors=SCATTER["edge_color"],
                linewidths=SCATTER["edge_width"], alpha=SCATTER["alpha"], zorder=4,
            )
    ax.set_ylabel(f"Improvement in best {objective_keys[0]}" if single else
                  r"Hypervolume improvement ($\Delta$HV)", fontsize=FONT_LABEL)
elif args.aggregate_runs:
    # One curve per arm: the mean over its runs at each step, and the band asked for.
    # The individual runs are not drawn underneath - the point of asking for this view
    # is that a dozen overlapping curves is what made the arms hard to compare.
    legend_handles = []
    for arm in sorted(arm_curves):
        if args.metric == "regret":
            # Averaged in log space, because that is the space the axis draws in and the
            # space the quantity lives in: a regret runs over decades, and an arithmetic
            # mean of 100 and 0.1 is 50 - a number describing neither run. Worse, the
            # interval around such a mean routinely reaches below zero, which on a log axis
            # is not a wide band but a missing one, and the curve grows spikes to the
            # bottom of the frame wherever one run has nearly converged. The geometric mean
            # and a band symmetric in log space are positive by construction.
            mean, low, high = (10.0 ** value for value in mean_band(
                [[math.log10(v) for v in curve] for curve in arm_curves[arm]], args.band))
        else:
            mean, low, high = mean_band(arm_curves[arm], args.band)
        steps = range(1, len(mean) + 1)
        ax.fill_between(steps, low, high, color=_color(arm), alpha=0.18,
                        linewidth=0, zorder=2)
        # Marked as well as coloured and dashed: on a curve this long a marker every
        # tenth step is what identifies the series where the dashes of two arms happen
        # to coincide, without turning the line into a row of symbols.
        ax.plot(steps, mean, color=_color(arm), linewidth=1.6,
                linestyle=_line_style(arm), marker=_marker(arm),
                markevery=max(1, len(mean) // 10), markersize=5, zorder=3)
        legend_handles.append(mlines.Line2D(
            [], [], color=_color(arm), linewidth=1.6, linestyle=_line_style(arm),
            marker=_marker(arm), markersize=5,
            label=arm_legend_label(arm.capitalize(), args.band, len(arm_curves[arm]))))
    # Where each arm's initial design ends. Arms that differ only by strategy share one
    # design window, and then this is the single neutral span it always was. A sweep over
    # --n-initial makes the design size the very thing being compared, and there a span
    # at the smallest would claim every longer design had ended there too.
    #
    # Drawn as abutting bands rather than one span per arm: every arm's span would start
    # at the same left edge, so they would stack alpha over the shared region and double
    # the ink exactly where it says least. Each band instead covers the steps between one
    # design size and the next, coloured by the arm whose design ends at its right edge -
    # the extra exploration that arm bought over the one before it.
    arm_edge = {}
    for arm, lengths in arm_n_initial.items():
        if len(set(lengths)) > 1:
            print(f"! {arm}: runs disagree on the initial design's length "
                  f"({min(lengths)}-{max(lengths)} steps) - taking the shortest.")
        arm_edge[arm] = min(lengths)
    neutral = fig_cfg["colors"].get("ground_truth", "#8A8F98")
    left = 0.5
    for edge in sorted({n for n in arm_edge.values() if n > 0}):
        owners = [arm for arm, n in arm_edge.items() if n == edge]
        # Neutral for the first band, which every arm is still exploring in, and for any
        # band more than one arm ends at: one arm's colour would claim a region it does
        # not own alone. A band a single arm ends at gets that arm's colour.
        shared = left == 0.5 or len(owners) > 1
        ax.axvspan(left, edge + 0.5,
                   color=neutral if shared else _color(owners[0]),
                   alpha=0.14 if shared else 0.10,
                   hatch="//" if shared else "\\\\", linewidth=0, zorder=1)
        left = edge + 0.5
    # Runs of one arm rarely stop at the same step; the shortest is what they were
    # truncated to, and saying so keeps a curve that ends early from reading as a run
    # that stalled. Reported per arm, since each is truncated to its own shortest and
    # arms are free to differ from each other.
    for arm in sorted(arm_curves):
        lengths = {len(c) for c in arm_curves[arm]}
        if len(lengths) > 1:
            print(f"! {arm}: runs of unequal length ({min(lengths)}-{max(lengths)} "
                  f"steps), truncated to {min(lengths)}")
    ax.set_ylabel(y_label(), fontsize=FONT_LABEL)
else:
    for name, (s_steps, s_hvs, s_labels) in traces.items():
        if not single:
            ax.plot(s_steps, s_hvs, color=_color(name), linewidth=1.2, zorder=3)
        for lbl in sorted(set(s_labels)):
            idx = [i for i, l in enumerate(s_labels) if l == lbl]
            ax.scatter(
                [s_steps[i] for i in idx],
                [s_hvs[i]   for i in idx],
                s=SCATTER["marker_size"] * (1.0 if single else 0.7), marker=_marker(lbl),
                facecolors=_color(lbl), edgecolors=SCATTER["edge_color"],
                linewidths=SCATTER["edge_width"], alpha=SCATTER["alpha"], zorder=4,
            )
    ax.set_ylabel(y_label(), fontsize=FONT_LABEL)

if args.metric == "regret":
    # Log, because regret decays toward zero: on a linear axis every arm's curve is
    # pinned to the bottom of the frame long before it has finished converging, and the
    # rate - which is what the metric exists to compare - is invisible.
    ax.set_yscale("log")
    if regret_floored:
        floor = min(v for _s, ys, _l in traces.values() for v in ys)
        ax.axhline(floor, color="#888888", linestyle=":", linewidth=0.8, alpha=0.7,
                   zorder=1)
elif args.metric == "normalized":
    # Where the optimum is. Without it a curve flattening at 0.98 looks the same as one
    # flattening at 1.0, which is the whole distinction this metric was chosen to draw.
    ax.axhline(1.0, color=fig_cfg["colors"].get("ground_truth", "#8A8F98"),
               linestyle="--", linewidth=1.0, alpha=0.9, zorder=1)

ax.set_xlabel("Number of observations", fontsize=FONT_LABEL)

ax.tick_params(labelsize=FONT_LABEL - 1)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True, **fig_cfg["grid"])
leg_cfg = fig_cfg["legend"]
# loc="best": let matplotlib place the legend where it overlaps the data least
# (the hypervolume curve typically fills the lower-left, freeing other corners).
ax.legend(handles=legend_handles, fontsize=FONT_LEGEND,
          loc="best", frameon=leg_cfg["frameon"], framealpha=leg_cfg["framealpha"])

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
