import argparse
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
from pybo_gui.modules.bayesian_campaign_analysis._labels import base_label, styler
from pybo_gui.modules.bayesian_campaign_analysis._aggregate import (
    BAND_MODES, mean_band, band_label)
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
# Baseline groups (by `technology`) excluded from the hypervolume improvement
# curve: they are reference points, not part of the optimisation campaign.
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
    return (exp.get("experiment_type") or exp.get("technology") or "unknown").lower()


# ---- LOAD ----
rows = []
for exp in load_experiments_from_map(MAP_PATH):
    # Skip baseline/reference groups — they are not part of the optimisation
    # campaign and must not contribute to the hypervolume improvement curve.
    if str(exp.get("technology", "")).lower() in EXCLUDED_TECHNOLOGIES:
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
        # Read from technology rather than from the label: the label is whatever the map
        # was built by, and under the default it names the run, so it tells the runs of
        # one arm apart instead of holding them together.
        "arm":      str(exp.get("technology", "")).lower() or _label(exp),
        "group_id": exp["group_id"],
        "feasible": is_feasible(r, constraints),
        "point":    point,
    })

if not rows:
    print("No data with the requested keys.")
    sys.exit(1)

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
# Range-based margin so the reference stays "worse" than every point in
# minimization space, valid for both positive and negated (maximized) axes.
# From the feasible points alone: the reference exists to bound the volume the front
# encloses, and a point that violates a constraint is not on any front. Letting one push
# the corner out would inflate every arm's hypervolume by the same wasted evaluation.
feasible_rows = [row for row in rows if row["feasible"]]
if not feasible_rows:
    print("Every observation violates the constraints - nothing to measure a volume "
          "against.")
    sys.exit(1)
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

all_labels     = sorted({r["label"] for r in rows if r["label"] is not None})

# The runs of each arm, in the order their traces were built, ready to be averaged.
arm_curves = {}
for name, (_s_steps, s_hvs, _s_labels) in traces.items():
    arm_curves.setdefault(series_arm[name], []).append(s_hvs)

if args.aggregate_runs and args.improvement:
    # Both rewrite what a point on the curve means, and stacking them would average
    # per-step gains that were already floored onto a shared log decade.
    raise SystemExit("--aggregate-runs and --improvement do not combine: the first "
                     "averages runs of a cumulative curve, the second replaces that "
                     "curve with per-step gains. Pick one.")

# Arms take a colour of their own when aggregating, so the mean curve is not confused
# with any single run's. Left alone otherwise, since adding names here would shift the
# colours assigned by position to every existing label.
_color, _marker, _front_style = styler(
    fig_cfg, all_labels + sorted(arm_curves) if args.aggregate_runs else all_labels)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["hypervolume"])

if len(traces) == 1 and not args.aggregate_runs:
    # One campaign: shade where each phase of it ran, as the single-sequence view did.
    steps, _hvs, labels = next(iter(traces.values()))
    prev_lbl     = labels[0]
    region_start = steps[0]
    for i in range(1, len(steps) + 1):
        lbl = labels[i] if i < len(labels) else None
        if lbl != prev_lbl or i == len(steps):
            ax.axvspan(
                region_start - 0.5, steps[i - 1] + 0.5,
                color=_color(prev_lbl), alpha=0.08,
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
        mean, low, high = mean_band(arm_curves[arm], args.band)
        steps = range(1, len(mean) + 1)
        ax.fill_between(steps, low, high, color=_color(arm), alpha=0.18,
                        linewidth=0, zorder=2)
        ax.plot(steps, mean, color=_color(arm), linewidth=1.6, zorder=3)
        legend_handles.append(mlines.Line2D(
            [], [], color=_color(arm), linewidth=1.6,
            label=f"{arm.capitalize()} - {band_label(args.band, len(arm_curves[arm]))}"))
    # Runs of one arm rarely stop at the same step; the shortest is what they were
    # truncated to, and saying so keeps a curve that ends early from reading as a run
    # that stalled. Reported per arm, since each is truncated to its own shortest and
    # arms are free to differ from each other.
    for arm in sorted(arm_curves):
        lengths = {len(c) for c in arm_curves[arm]}
        if len(lengths) > 1:
            print(f"! {arm}: runs of unequal length ({min(lengths)}-{max(lengths)} "
                  f"steps), truncated to {min(lengths)}")
    ax.set_ylabel(f"Best {objective_keys[0]}" if single else "Hypervolume",
                  fontsize=FONT_LABEL)
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
    ax.set_ylabel(f"Best {objective_keys[0]}" if single else "Hypervolume",
                  fontsize=FONT_LABEL)

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
