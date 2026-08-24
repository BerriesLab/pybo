import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map
from pybo_gui.modules.bayesian_campaign_analysis._constraints import parse_constraints, is_feasible, ConstraintError
from pybo_gui.modules.bayesian_campaign_analysis._labels import (
    DASHES, arm_label, base_label, is_initial, styler)
from pybo_gui.modules.bayesian_campaign_analysis._uncertainty import total_sd, mean_sd
from pybo_gui.modules.bayesian_campaign_analysis._aggregate import (
    BAND_MODES, mean_band, arm_legend_label, attainment_grid, step_interpolate)
from pybo_gui.modules.bayesian_campaign_analysis._colorscale import diverging_norm, mark_center
from pybo_gui.modules.bayesian_campaign_analysis._reference import draw_reference
from pybo_gui.modules.bayesian_campaign_analysis._legend import place_legend

parser = argparse.ArgumentParser()
parser.add_argument("--x", required=True, help="Result key for x axis (objective)")
parser.add_argument("--y", required=True, help="Result key for y axis (objective)")
parser.add_argument("--z", default="",    help="Result key for color coding (color only, not used in Pareto)")
parser.add_argument("--zcenter", type=float, default=None,
                    help="Value the diverging colormap is centered on (defaults to the "
                         "midpoint of the data's own range)")
parser.add_argument("--xlabel", default="", help="Override x-axis label (LaTeX accepted via $...$)")
parser.add_argument("--ylabel", default="", help="Override y-axis label (LaTeX accepted via $...$)")
parser.add_argument("--zlabel", default="", help="Override colorbar label (LaTeX accepted via $...$)")
parser.add_argument("--grouped",      action="store_true", default=False,
                    help="Aggregate per group_id")
parser.add_argument("--errorbar", choices=["sem", "std", "minmax"], default="sem",
                    help="Error-bar mode in grouped view. sem = uncertainty of the "
                         "group's mean, std = spread of one measurement, minmax = mean "
                         "to the group's min and max. sem and std both fold in the "
                         "measurement variance the run recorded, when it recorded one.")
parser.add_argument("--show-numbers", action="store_true", default=False)
parser.add_argument("--aggregate-runs", action="store_true", default=False,
                    help="Replace the per-run fronts with one mean front per arm, read "
                         "onto a shared grid, plus a band. 2-D only. The grid spans "
                         "every run's range; where a run hasn't reached a given point "
                         "it drops out of that point's mean instead of the whole arm "
                         "being skipped, so the band widens - or the line thins to a "
                         "single run - toward the edges.")
parser.add_argument("--band", default="ci95", choices=BAND_MODES,
                    help="What the band around the aggregated front shows (default: "
                         "%(default)s).")
parser.add_argument("--maximize", action="append", default=[],
                    help="Result key of an objective to maximize (repeatable). "
                         "Its axis is negated for the Pareto-front computation "
                         "only; plotted values stay real. Default: minimize.")
parser.add_argument("--ground-truth", default="", dest="ground_truth",
                    help="Path to the run's objective.py. When given, the true "
                         "objective is drawn under the observations.")
parser.add_argument("--gt-method", choices=["random", "grid"], default="random",
                    dest="gt_method",
                    help="How the ground truth covers the space: random samples, or a "
                         "uniform grid.")
parser.add_argument("--gt-samples", type=int, default=4096, dest="gt_samples",
                    help="Samples drawn when --gt-method is random.")
parser.add_argument("--gt-spacing", type=float, default=0.05, dest="gt_spacing",
                    help="Step on every axis when --gt-method is grid, as a fraction "
                         "of that axis' range: 0.05 is 21 points per parameter.")
parser.add_argument("--gt-front", choices=["always", "never"], default="always",
                    dest="gt_front",
                    help="Whether the ground truth's own Pareto front is drawn along "
                         "with its cloud (default: always). Only ever there on an "
                         "unconstrained problem - a constrained one has no single front "
                         "to draw, so this changes nothing.")
parser.add_argument("--gt-noisy", action="store_true", default=False, dest="gt_noisy",
                    help="Draw the ground truth the way a run would have observed it, "
                         "noise and all, instead of the noiseless value underneath.")
parser.add_argument("--constraint", action="append", default=[],
                    help="Feasibility constraint as key:op:value (repeatable). "
                         "Only feasible experiments contribute to the Pareto front; "
                         "infeasible ones are shown dimmed.")
parser.add_argument("--front-scope", choices=["label", "initial-vs-all"], default="label",
                    dest="front_scope",
                    help="What each front is drawn over. label (default) draws one per "
                         "series, which under --label-by run or strategy is already the "
                         "whole campaign, design points included. initial-vs-all draws "
                         "two whatever the labelling: the initial design's own front, "
                         "and the front over everything - so the pair says what proposing "
                         "added to the dataset the run started from. Not applied when "
                         "averaging runs, which draws one mean front per arm instead.")
parser.add_argument("--front-line", choices=["auto", "always", "never"], default="auto",
                    dest="front_line",
                    help="Whether the non-dominated points are joined into a front line. "
                         "auto (default) draws one only on an unconstrained problem, "
                         "where every trade-off between two front points is attainable; "
                         "always and never make it the caller's decision instead, "
                         "constrained or not.")
args = parser.parse_args()
try:
    constraints = parse_constraints(args.constraint)
except ConstraintError as exc:
    print(exc)
    sys.exit(2)

# ---- CONFIG ----
MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

MARKERS      = fig_cfg["markers"]["label"]
LABEL_COLORS = fig_cfg["colors"]["label"]
SCATTER      = fig_cfg["scatter"]
FRONT        = fig_cfg["pareto_front"]

FONT_LABEL  = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]
FONT_TITLE  = fig_cfg["font"]["title"]
DPI         = fig_cfg["dpi"]

# Axis-label overrides by result key. The problem definition already names its
# objectives, so this is empty until a key needs a unit or LaTeX in the label.
PRETTY_NAMES = {}


def _label(exp):
    return (exp.get("experiment_type") or exp.get("optimizer") or "unknown").lower()


def pareto_front(points, sx=1.0, sy=1.0):
    """Non-dominated points under the given per-axis senses (sx/sy are +1 to
    minimize, -1 to maximize). Domination is evaluated in signed space; the
    returned points keep their original (real) coordinates for plotting."""
    pts = sorted(points, key=lambda p: (sx * p[0], sy * p[1]))
    front, best_y = [], float("inf")
    for x, y in pts:
        ty = sy * y
        if ty < best_y:
            front.append((x, y))
            best_y = ty
    return front


# Per-axis sense: +1 minimize, -1 maximize (front computed in signed space).
maximize = set(args.maximize)
sx = -1.0 if args.x in maximize else 1.0
sy = -1.0 if args.y in maximize else 1.0

# ---- LOAD ----
use_z = bool(args.z)

def column(exp, key):
    """A column by name, from the results or from the parameters it was measured at.

    The axis pickers offer both, and a parameter is never in results - looking only
    there left a parameter-coloured plot with no colour at all.
    """
    results = exp.get("results", {})
    if key in results:
        return results[key]
    return (exp.get("parameters") or {}).get(key)


raw_rows = []
for exp in load_experiments_from_map(MAP_PATH):
    r = exp.get("results", {})
    raw_rows.append({
        "label":    _label(exp),
        # The arm whose runs get averaged together. From optimizer and provenance,
        # not the label: under the default labelling the label names the run, which
        # separates the very runs that have to be pooled.
        "arm":      arm_label(exp, _label(exp)),
        # Kept alongside the label so pooling by arm still separates the runs when the
        # map was built by strategy, where the label no longer names them.
        "run":      exp.get("run"),
        "group_id": exp["group_id"],
        "x":        column(exp, args.x),
        "y":        column(exp, args.y),
        # What the measurement said about its own uncertainty. to_json writes a
        # <label>_var next to every value, left null when the run measured none.
        # A parameter is a setting rather than a measurement, so it has none.
        "x_var":    r.get(f"{args.x}_var"),
        "y_var":    r.get(f"{args.y}_var"),
        "z_val":    column(exp, args.z) if use_z else None,
        "feasible": is_feasible(r, constraints),
        # The user's flagged benchmark. Pulled out of the ordinary per-label series
        # entirely and drawn through _reference instead - see below.
        "reference": exp.get("reference", False),
    })

# ---- AGGREGATE (if grouped) ----
# In grouped mode, infeasible experiments are excluded from aggregation, so a
# group's mean is built only from points that individually satisfy the
# constraints; every resulting group row is therefore feasible. The infeasible
# experiments themselves are still drawn — as individual points (see below) —
# they are just never folded into a group mean.
if args.grouped:
    groups = {}
    order  = []
    for r in raw_rows:
        if r["x"] is None or r["y"] is None or not r["feasible"]:
            continue
        gid = r["group_id"]
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(r)

    rows = []
    for gid in order:
        items = groups[gid]
        xs    = [it["x"] for it in items]
        ys    = [it["y"] for it in items]
        zs    = [it["z_val"] for it in items if it["z_val"] is not None] if use_z else []
        x_mean, y_mean = float(np.mean(xs)), float(np.mean(ys))
        if args.errorbar == "minmax":
            x_err_lo = max(0.0, x_mean - float(np.min(xs)))
            x_err_hi = max(0.0, float(np.max(xs)) - x_mean)
            y_err_lo = max(0.0, y_mean - float(np.min(ys)))
            y_err_hi = max(0.0, float(np.max(ys)) - y_mean)
        else:
            # NB: don't reuse sx/sy here — those hold the Pareto-sense signs
            # used to compute the fronts further down.
            # Both modes reconcile the repeats' scatter with the recorded measurement
            # variance rather than adding them - see _uncertainty.
            estimate = mean_sd if args.errorbar == "sem" else total_sd
            sd_x = estimate(xs, [it["x_var"] for it in items])
            sd_y = estimate(ys, [it["y_var"] for it in items])
            x_err_lo = x_err_hi = sd_x
            y_err_lo = y_err_hi = sd_y
        rows.append({
            "label":    items[0]["label"],
            "arm":      items[0]["arm"],
            "run":      items[0]["run"],
            "group_id": gid,
            "x":        x_mean,
            "x_err_lo": x_err_lo,
            "x_err_hi": x_err_hi,
            "y":        y_mean,
            "y_err_lo": y_err_lo,
            "y_err_hi": y_err_hi,
            "z_val":    float(np.mean(zs)) if zs else None,
            "n":        len(items),
            "feasible": True,
            # One run per group (group_id keys on run - see build_group_map), so
            # every item in the group agrees on this.
            "reference": items[0]["reference"],
        })
    valid = rows
else:
    valid = [r for r in raw_rows if r["x"] is not None and r["y"] is not None]
    for r in valid:
        r["x_err_lo"] = r["x_err_hi"] = 0.0
        r["y_err_lo"] = r["y_err_hi"] = 0.0
        r["n"]        = 1

# ---- REFERENCE (pulled out of the ordinary per-label series) ----
# A run flagged as the user's benchmark stops being "just another run": it is drawn
# once, apart, through _reference - see below - instead of taking its usual place
# among the series it is being compared against.
reference_valid = [r for r in valid if r["feasible"] and r["reference"]]
valid = [r for r in valid if not r["reference"]]

# ---- INFEASIBLE (always individual points) ----
# Constraint-violating experiments are taken straight from raw_rows so they are
# shown one-per-experiment in both grouped and non-grouped mode — they are never
# aggregated. (In non-grouped mode these are also present in `valid`, but the
# feasible drawing loop filters them out by `r["feasible"]`.)
#
# Dropped outright when runs are averaged. Aggregating asks what an arm does, and
# answers it with one curve per arm; an infeasible point has no arm-level reading,
# so it can only be drawn as itself. Scattering every run's individual violations
# under a handful of mean fronts buries the very thing the option was chosen to
# show, and does it worst on the arm that wasted the most evaluations.
infeasible = [] if args.aggregate_runs else [
    r for r in raw_rows
    if r["x"] is not None and r["y"] is not None and not r["feasible"]]

# ---- Z COLOR SCALE ----
z_cmap = z_norm = None
if use_z:
    # Span all plotted points (feasible + infeasible): the marker already encodes
    # feasibility, so the colour shows each point's true z value. Feasible values
    # come from `valid` (group means when grouped); infeasible from the
    # individual points above.
    z_vals  = [r["z_val"] for r in valid if r["feasible"] and r["z_val"] is not None]
    z_vals += [r["z_val"] for r in infeasible if r["z_val"] is not None]
    if z_vals:
        z_cmap = cm.coolwarm
        z_norm = diverging_norm(z_vals, args.zcenter)
    else:
        use_z = False

# ---- PARETO FRONTS ----
# One front per series present, so what a front is drawn per is decided upstream by the
# map's --label-by (run, strategy, strategy+run, provenance) rather than fixed here.
FRONT_LABELS     = sorted({r["label"] for r in valid})


# Arms take a colour of their own when aggregating, so a mean front is not confused
# with any one run's. Left alone otherwise: adding names shifts the colours the styler
# assigns by position.
_arms = sorted({r["arm"] for r in valid}) if args.aggregate_runs else []
_label_color, _label_marker, _line_style, _front_style = styler(fig_cfg, list(FRONT_LABELS) + _arms)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["pareto"])

if args.ground_truth:
    # Under everything: it is the backdrop the campaign is read against, not a series.
    from pybo_gui.modules.bayesian_campaign_analysis._ground_truth import ground_truth
    gt_points, gt_front, gt_constrained = ground_truth(
        args.ground_truth, args.x, args.y,
        args.gt_method, args.gt_samples, args.gt_spacing, args.gt_noisy)
    if gt_points:
        gt_color = fig_cfg["colors"].get("ground_truth", "#8A8F98")
        ax.scatter([p[0] for p in gt_points], [p[1] for p in gt_points],
                   s=SCATTER["marker_size"] * 0.25, marker=".", facecolors=gt_color,
                   edgecolors="none", alpha=0.25, zorder=0)
        if gt_front and args.gt_front == "always":
            ax.plot([p[0] for p in gt_front], [p[1] for p in gt_front],
                    color=gt_color, linewidth=1.2, zorder=1)
            legend_handles_gt = mlines.Line2D([], [], color=gt_color, linewidth=1.2,
                                              label="Ground truth")
        else:
            # No front line drawn - a constrained problem has none, or it was turned off -
            # so the handle shows the cloud instead.
            legend_handles_gt = mlines.Line2D([], [], linestyle="None", marker=".",
                                              color=gt_color,
                                              markersize=SCATTER["marker_size"] ** 0.5,
                                              label="Ground truth")
    else:
        legend_handles_gt = None
else:
    gt_constrained = False
    legend_handles_gt = None

# Whether this problem is known to be constrained, and so whether joining front points
# by a line would claim trade-offs a constraint may forbid. Either the ground truth says
# the problem declares constraints, or the plot was given some to enforce itself.
constrained = gt_constrained or bool(constraints)

# ...unless the caller has decided for itself. "auto" is the judgement above; always and
# never answer the same question without consulting it, which is what lets a constrained
# problem show its front line and an unconstrained one hide it.
draw_front = {"auto": not constrained, "always": True, "never": False}[args.front_line]

all_labels     = sorted({r["label"] for r in valid})
legend_handles = [legend_handles_gt] if legend_handles_gt is not None else []
_last_sc       = None

# Skipped entirely when aggregating: a dozen runs' observations are what made the
# picture unreadable, which is the reason for asking to collapse them, and one legend
# entry per run buries the two that name the mean fronts.
for lbl in [] if args.aggregate_runs else all_labels:
    marker = _label_marker(lbl)
    subset = [r for r in valid if r["label"] == lbl and r["feasible"]]
    if use_z:
        subset = [r for r in subset if r["z_val"] is not None]
    if not subset:
        continue

    if args.grouped:
        if use_z:
            colors_pts = [z_cmap(z_norm(r["z_val"])) for r in subset]
        else:
            face_color = _label_color(lbl)
            colors_pts = [face_color] * len(subset)
        for r, fc in zip(subset, colors_pts):
            # Drawn whenever there is something to draw, not only when the group has
            # repeats: a single measurement with a recorded variance has a real bar.
            xerr = [[r["x_err_lo"]], [r["x_err_hi"]]] if r["x_err_hi"] else None
            yerr = [[r["y_err_lo"]], [r["y_err_hi"]]] if r["y_err_hi"] else None
            ax.errorbar(
                [r["x"]], [r["y"]],
                xerr=xerr, yerr=yerr,
                fmt=marker,
                markerfacecolor=fc,
                markeredgecolor=SCATTER["edge_color"],
                markeredgewidth=SCATTER["edge_width"],
                ecolor=fc if isinstance(fc, str) else "gray",
                markersize=SCATTER["marker_size"] ** 0.5,
                linewidth=0.8,
                capsize=3,
                alpha=SCATTER["alpha"],
                zorder=3,
            )
        if use_z:
            # invisible scatter to anchor the colorbar
            _last_sc = ax.scatter(
                [r["x"] for r in subset], [r["y"] for r in subset],
                c=[r["z_val"] for r in subset], cmap=z_cmap, norm=z_norm,
                s=0, zorder=0,
            )
    else:
        if use_z:
            _last_sc = ax.scatter(
                [r["x"] for r in subset], [r["y"] for r in subset],
                c=[r["z_val"] for r in subset], cmap=z_cmap, norm=z_norm,
                s=SCATTER["marker_size"], marker=marker,
                edgecolors=SCATTER["edge_color"],
                linewidths=SCATTER["edge_width"], alpha=SCATTER["alpha"], zorder=3,
            )
        else:
            face_color = _label_color(lbl)
            ax.scatter(
                [r["x"] for r in subset], [r["y"] for r in subset],
                s=SCATTER["marker_size"], marker=marker,
                facecolors=face_color, edgecolors=SCATTER["edge_color"],
                linewidths=SCATTER["edge_width"], alpha=SCATTER["alpha"], zorder=3,
            )

    legend_face = "gray" if use_z else _label_color(lbl)
    legend_handles.append(
        # Marker only: the front line is one style shared by every series, so it is named
        # once by its own handle below rather than advertised on each of these.
        mlines.Line2D([], [], color=_label_color(lbl),
                      linewidth=0.0, linestyle="None", marker=marker,
                      markerfacecolor=legend_face, markeredgecolor=SCATTER["edge_color"],
                      markeredgewidth=0.8, markersize=SCATTER["marker_size"] ** 0.5,
                      label=lbl.capitalize())
    )

# Infeasible points (violating a constraint): drawn as individual filled "X"
# markers (in both grouped and non-grouped mode) and excluded from the front.
# When a color code is passed they are coloured on the shared z-scale (which
# spans all points, so their true z value shows); otherwise dimmed gray.
# `infeasible` is already empty when runs are averaged, so this needs no second
# check for it - and must not have one that overwrites the list, which is what the
# z branch used to do.
infeasible_pts = infeasible
if use_z:
    infeasible_pts = [r for r in infeasible if r["z_val"] is not None]
if infeasible_pts:
    if use_z:
        ax.scatter(
            [r["x"] for r in infeasible_pts], [r["y"] for r in infeasible_pts],
            c=[r["z_val"] for r in infeasible_pts], cmap=z_cmap, norm=z_norm,
            s=SCATTER["marker_size"], marker="X",
            edgecolors="black", linewidths=0.6, alpha=0.55, zorder=1,
        )
        legend_face = "gray"
        legend_edge = "black"
    else:
        ax.scatter(
            [r["x"] for r in infeasible_pts], [r["y"] for r in infeasible_pts],
            s=SCATTER["marker_size"], marker="X",
            facecolors="lightgray", edgecolors="none", alpha=0.30, zorder=1,
        )
        legend_face = "lightgray"
        legend_edge = "none"
    legend_handles.append(
        mlines.Line2D([], [], linestyle="None", marker="X",
                      markerfacecolor=legend_face, markeredgecolor=legend_edge,
                      markersize=SCATTER["marker_size"] ** 0.5, label="Infeasible")
    )

if args.show_numbers:
    # Label feasible points (group means when grouped) and every individual
    # infeasible point, so constraint-violating experiments are tagged too.
    src  = [r for r in valid if r["feasible"] and ((not use_z) or r["z_val"] is not None)]
    src += [r for r in infeasible if (not use_z) or r["z_val"] is not None]
    for r in src:
        tag = str(r["group_id"])
        ax.annotate(tag, (r["x"], r["y"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=FONT_LEGEND - 2, color="dimgray")

# Pareto fronts — shared colour and style. The non-dominated points are always circled;
# they are only joined into a front line on an unconstrained problem, where the trade-offs
# between two of them are all attainable.
#
# One front per *base* label, not per label: a run's initial design is a series of its own
# for styling ("run3 (initial)"), but it is the same campaign, so a design point still
# undominated at the end belongs on that run's front. Grouping by base is also what keeps
# --label-by provenance honest - there "initial" and "proposed" are bases in their own
# right, so asking for them apart still draws them apart.
if args.aggregate_runs:
    # One mean front per arm instead of one per run. The runs of an arm reach their
    # fronts at x values none of the others visited, so they cannot be averaged point by
    # point: each is read onto a shared grid first. All of it happens in signed
    # (minimisation) space, so "the best y attained at or before this x" is one rule
    # whichever way an axis runs, and the result is signed back at the end.
    by_arm = {}
    for r in valid:
        if r["feasible"]:
            by_arm.setdefault(r["arm"], {}).setdefault(
                r["run"] or base_label(r["label"]), []).append(
                (sx * r["x"], sy * r["y"]))

    for arm in sorted(by_arm):
        fronts = []
        for pts in by_arm[arm].values():
            f = sorted(pareto_front(pts), key=lambda p: p[0])
            if f:
                fronts.append(([p[0] for p in f], [p[1] for p in f]))
        if not fronts:
            continue
        try:
            grid = attainment_grid(fronts)
        except ValueError as error:
            print(f"! {arm}: {error}")
            continue
        curves = [step_interpolate(fx, fy, grid) for fx, fy in fronts]
        mean, low, high = mean_band(curves, args.band)
        color = _label_color(arm)
        # Back out of signed space to plot in the objectives' own units.
        gx = sx * grid
        ax.fill_between(gx, sy * low, sy * high, color=color, alpha=0.18,
                        linewidth=0, zorder=2)
        # Drawn through the corners of the mean attainment surface rather than as the
        # staircase it strictly is. The surface is a step function - between two front
        # points a run achieved the earlier one and nothing better - but the front line
        # here is a visual guide to where an arm ended up, not a claim that the pairs
        # along it were attained, and the per-run fronts elsewhere in this plot are drawn
        # the same way. The band below keeps the full grid, where the shading carries the
        # uncertainty and a stepped edge costs nothing to read.
        #
        # Corners, not the 200 grid samples: plotting those directly would redraw the
        # staircase with sloped risers, which is the worst of both.
        corners_x, corners_y = [], []
        for gxi, myi in zip(gx, sy * mean):
            if not corners_y or not (np.isclose(myi, corners_y[-1], equal_nan=True)):
                corners_x.append(gxi)
                corners_y.append(myi)
        ax.plot(corners_x, corners_y,
                color=color, linewidth=1.4, linestyle=_line_style(arm), zorder=3)
        legend_handles.append(mlines.Line2D(
            [], [], color=color, linewidth=1.4, linestyle=_line_style(arm),
            label=arm_legend_label(arm.capitalize(), args.band, len(fronts))))

# Drawn unconditionally - independent of --aggregate-runs, which is about the
# ordinary series above, not the benchmark.
draw_reference(ax, reference_valid, sx, sy, args.band,
               fig_cfg["colors"].get("reference", "#E62020"), SCATTER["marker_size"],
               FRONT["edge_width"], legend_handles)

if args.aggregate_runs:
    front_groups = []
elif args.front_scope == "initial-vs-all":
    # The design and the campaign, not a series each: together they say what proposing
    # added to the dataset the run started from. The second is the union, design points
    # included - a front over the proposals alone draws a curve through points the design
    # had already beaten, so it is not a front of anything the campaign attained.
    front_groups = [
        ("design", [r for r in valid if r["feasible"] and is_initial(r["label"])]),
        ("overall", [r for r in valid if r["feasible"]]),
    ]
else:
    # One front per *base* label, so a run's design and its proposals - one campaign,
    # two series for styling - already share a front. Under --label-by provenance the
    # bases are "initial" and "proposed" in their own right, which is the one labelling
    # where asking for them apart draws the proposals alone.
    front_groups = [
        (base, [r for r in valid if base_label(r["label"]) == base and r["feasible"]])
        for base in sorted({base_label(lbl) for lbl in FRONT_LABELS})
    ]

front_handles = {}
for kind, members in front_groups:
    front = pareto_front([(r["x"], r["y"]) for r in members], sx, sy)
    if not front:
        continue
    # The design's own front is the lighter statement of the two: where the run started,
    # under the front it ended with.
    design = kind == "design"
    color = FRONT["design_color"] if design else FRONT["color"]
    width = FRONT["design_linewidth"] if design else FRONT["linewidth"]
    edge_width = FRONT["design_edge_width"] if design else FRONT["edge_width"]
    dashes = DASHES["initial"] if design else DASHES["proposed"]
    # Phased per series in the per-label mode, where several fronts share one style and
    # coincident ones would otherwise hide each other; fixed in the two-front mode, where
    # the dash is what tells the design from the campaign.
    style = (0, dashes) if kind in ("design", "overall") else _front_style(kind)

    xs, ys = [p[0] for p in front], [p[1] for p in front]
    if draw_front:
        ax.plot(xs, ys, color=color, linewidth=width, linestyle=style,
                zorder=2 if design else 3)
    # Circle each point with the marker of the series it actually came from, so a surviving
    # design point still reads as exploration rather than as a proposal. Edged like the
    # line, so a front point and the front it lies on read as the same statement.
    markers_at = {(r["x"], r["y"]): _label_marker(r["label"]) for r in members}
    for x, y in front:
        ax.scatter([x], [y], s=SCATTER["marker_size"], marker=markers_at.get((x, y), "o"),
                   facecolors="none", edgecolors=color, linewidths=edge_width,
                   zorder=4 if design else 5)

    # One handle per kind of front, not one per series: the fronts of several runs are
    # drawn alike, so naming that once says what it is without repeating a run's name per
    # front - which is what used to make the legend swallow the plot.
    name = "Initial design front" if design else "Pareto front"
    front_handles[name] = mlines.Line2D(
        [], [], color=color,
        linewidth=width if draw_front else 0.0,
        linestyle=(0, dashes) if draw_front else "None",
        marker="o", markerfacecolor="none", markeredgecolor=color,
        markeredgewidth=edge_width, markersize=SCATTER["marker_size"] ** 0.5,
        label=name)

legend_handles.extend(front_handles.values())

xlabel = args.xlabel if args.xlabel else PRETTY_NAMES.get(args.x, args.x)
ylabel = args.ylabel if args.ylabel else PRETTY_NAMES.get(args.y, args.y)
ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

ax.tick_params(labelsize=FONT_LABEL - 1)
ax.grid(True, **fig_cfg["grid"])
leg_cfg = fig_cfg["legend"]
# Long run names in quantity outgrow a corner at the normal font size, so past a
# handful this shrinks the legend (smaller type, more columns) to keep it inside
# the axes rather than let it spill past them.
place_legend(fig, ax, legend_handles, leg_cfg, FONT_LEGEND)

if use_z and _last_sc is not None:
    zlabel = args.zlabel if args.zlabel else PRETTY_NAMES.get(args.z, args.z)
    cbar = fig.colorbar(_last_sc, ax=ax, pad=0.02)
    cbar.set_label(zlabel, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_LABEL - 1)
    mark_center(cbar, args.zcenter)

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
