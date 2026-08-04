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
from pybo_gui.modules.bayesian_campaign_analysis._labels import styler
from pybo_gui.modules.bayesian_campaign_analysis._uncertainty import total_sd, mean_sd

parser = argparse.ArgumentParser()
parser.add_argument("--x", required=True, help="Result key for x axis (objective)")
parser.add_argument("--y", required=True, help="Result key for y axis (objective)")
parser.add_argument("--z", default="",    help="Result key for color coding (color only, not used in Pareto)")
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
                    help="Step on every axis when --gt-method is grid.")
parser.add_argument("--constraint", action="append", default=[],
                    help="Feasibility constraint as key:op:value (repeatable). "
                         "Only feasible experiments contribute to the Pareto front; "
                         "infeasible ones are shown dimmed.")
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

FONT_LABEL  = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]
FONT_TITLE  = fig_cfg["font"]["title"]
DPI         = fig_cfg["dpi"]

# Axis-label overrides by result key. The problem definition already names its
# objectives, so this is empty until a key needs a unit or LaTeX in the label.
PRETTY_NAMES = {}


def _label(exp):
    return (exp.get("experiment_type") or exp.get("technology") or "unknown").lower()


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
        })
    valid = rows
else:
    valid = [r for r in raw_rows if r["x"] is not None and r["y"] is not None]
    for r in valid:
        r["x_err_lo"] = r["x_err_hi"] = 0.0
        r["y_err_lo"] = r["y_err_hi"] = 0.0
        r["n"]        = 1

# ---- INFEASIBLE (always individual points) ----
# Constraint-violating experiments are taken straight from raw_rows so they are
# shown one-per-experiment in both grouped and non-grouped mode — they are never
# aggregated. (In non-grouped mode these are also present in `valid`, but the
# feasible drawing loop filters them out by `r["feasible"]`.)
infeasible = [r for r in raw_rows
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
        z_norm = mcolors.Normalize(vmin=min(z_vals), vmax=max(z_vals))
    else:
        use_z = False

# ---- PARETO FRONTS ----
# One front per series present, so what a front is drawn per is decided upstream by the
# map's --label-by (run, strategy, strategy+run, provenance) rather than fixed here.
FRONT_LABELS     = sorted({r["label"] for r in valid})


_label_color, _label_marker, _front_style = styler(fig_cfg, FRONT_LABELS)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["pareto"])

if args.ground_truth:
    # Under everything: it is the backdrop the campaign is read against, not a series.
    from pybo_gui.modules.bayesian_campaign_analysis._ground_truth import ground_truth
    gt_points, gt_front = ground_truth(args.ground_truth, args.x, args.y,
                                       args.gt_method, args.gt_samples, args.gt_spacing)
    if gt_points:
        gt_color = fig_cfg["colors"].get("ground_truth", "#8A8F98")
        ax.scatter([p[0] for p in gt_points], [p[1] for p in gt_points],
                   s=SCATTER["marker_size"] * 0.25, marker=".", facecolors=gt_color,
                   edgecolors="none", alpha=0.25, zorder=0)
        if gt_front:
            ax.plot([p[0] for p in gt_front], [p[1] for p in gt_front],
                    color=gt_color, linewidth=1.2, zorder=1)
            legend_handles_gt = mlines.Line2D([], [], color=gt_color, linewidth=1.2,
                                              label="Ground truth")
        else:
            legend_handles_gt = None
    else:
        legend_handles_gt = None
else:
    legend_handles_gt = None

all_labels     = sorted({r["label"] for r in valid})
legend_handles = [legend_handles_gt] if legend_handles_gt is not None else []
_last_sc       = None

for lbl in all_labels:
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
        mlines.Line2D([], [], color=_label_color(lbl), linewidth=1.0,
                      linestyle=_front_style(lbl), marker=marker,
                      markerfacecolor=legend_face, markeredgecolor=SCATTER["edge_color"],
                      markeredgewidth=0.8, markersize=SCATTER["marker_size"] ** 0.5,
                      label=lbl.capitalize())
    )

# Infeasible points (violating a constraint): drawn as individual filled "X"
# markers (in both grouped and non-grouped mode) and excluded from the front.
# When a color code is passed they are coloured on the shared z-scale (which
# spans all points, so their true z value shows); otherwise dimmed gray.
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

# Exploration Pareto fronts (sobol, lhs, manual) — shared colour and style.
for expl_lbl in FRONT_LABELS:
    pts   = [(r["x"], r["y"]) for r in valid if r["label"] == expl_lbl and r["feasible"]]
    front = pareto_front(pts, sx, sy)
    if not front:
        continue
    color  = _label_color(expl_lbl)
    marker = _label_marker(expl_lbl)
    ls     = _front_style(expl_lbl)
    xs, ys = [p[0] for p in front], [p[1] for p in front]
    ax.plot(xs, ys, color=color, linewidth=1.0, linestyle=ls, zorder=2)
    ax.scatter(xs, ys, s=SCATTER["marker_size"], marker=marker, facecolors="none", edgecolors=color, linewidths=1.0, zorder=4)
    # No handle here: the series' own entry already carries this front's colour and dash,
    # and repeating the run's name once per front is what made the legend swallow the plot.

xlabel = args.xlabel if args.xlabel else PRETTY_NAMES.get(args.x, args.x)
ylabel = args.ylabel if args.ylabel else PRETTY_NAMES.get(args.y, args.y)
ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

ax.tick_params(labelsize=FONT_LABEL - 1)
ax.grid(True, **fig_cfg["grid"])
leg_cfg = fig_cfg["legend"]
# Long run names in quantity outgrow any corner, so past a handful the legend moves under
# the axes and splits into columns instead of covering the data.
legend_below = len(legend_handles) > 4
if legend_below:
    legend = ax.legend(handles=legend_handles, fontsize=FONT_LEGEND - 1,
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=1 if len(legend_handles) <= 6 else 2,
              frameon=leg_cfg["frameon"], framealpha=leg_cfg["framealpha"])
else:
    legend = ax.legend(handles=legend_handles, fontsize=FONT_LEGEND,
                       loc="best", frameon=leg_cfg["frameon"],
                       framealpha=leg_cfg["framealpha"])

if use_z and _last_sc is not None:
    zlabel = args.zlabel if args.zlabel else PRETTY_NAMES.get(args.z, args.z)
    cbar = fig.colorbar(_last_sc, ax=ax, pad=0.02)
    cbar.set_label(zlabel, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_LABEL - 1)

fig.tight_layout(pad=fig_cfg["layout_pad"])
if legend_below:
    # tight_layout reserves nothing for a legend anchored outside the axes, so measure
    # what it actually took and give it that much of the figure.
    fig.canvas.draw()
    height = legend.get_window_extent().transformed(fig.transFigure.inverted()).height
    fig.subplots_adjust(bottom=min(0.6, height + 0.14))
plt.show(block=__name__ == "__main__")
