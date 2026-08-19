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
from pybo_gui.modules.bayesian_campaign_analysis._labels import base_label, styler
from pybo_gui.modules.bayesian_campaign_analysis._uncertainty import total_sd, mean_sd
from pybo_gui.modules.bayesian_campaign_analysis._legend import place_legend

parser = argparse.ArgumentParser(
    description="Single-objective campaign: the objective over one or two parameters.")
parser.add_argument("--objective", required=True,
                    help="Result key of the one objective the campaign optimised.")
parser.add_argument("--parameter", action="append", default=[], required=True,
                    help="Parameter key for an axis (repeatable, one or two). One "
                         "parameter draws the objective against it; two draw the "
                         "parameter plane with the objective as colour.")
parser.add_argument("--objective-label", default="", dest="objective_label",
                    help="Override the objective's axis label (LaTeX accepted via $...$)")
parser.add_argument("--parameter-label", action="append", default=[],
                    dest="parameter_label",
                    help="Override a parameter's axis label, in --parameter order.")
parser.add_argument("--maximize", action="append", default=[],
                    help="Result key of an objective to maximize (repeatable). Decides "
                         "which observation is the best one. Default: minimize.")
parser.add_argument("--grouped", action="store_true", default=False,
                    help="Aggregate per group_id")
parser.add_argument("--errorbar", choices=["sem", "std", "minmax"], default="sem",
                    help="Error-bar mode in grouped view. sem = uncertainty of the "
                         "group's mean, std = spread of one measurement, minmax = mean "
                         "to the group's min and max. sem and std both fold in the "
                         "measurement variance the run recorded, when it recorded one. "
                         "Only drawn on one parameter, where the objective has an axis.")
parser.add_argument("--show-numbers", action="store_true", default=False)
parser.add_argument("--ground-truth", default="", dest="ground_truth",
                    help="Path to the run's objective.py. When given, the true "
                         "objective is drawn under the observations.")
parser.add_argument("--gt-method", choices=["random", "grid"], default="random",
                    dest="gt_method",
                    help="How the ground truth covers the space: random samples, or a "
                         "uniform grid. Two parameters read best as a grid, which is "
                         "what a filled contour needs.")
parser.add_argument("--gt-samples", type=int, default=4096, dest="gt_samples",
                    help="Samples drawn when --gt-method is random.")
parser.add_argument("--gt-spacing", type=float, default=0.05, dest="gt_spacing",
                    help="Step on every axis when --gt-method is grid, as a fraction "
                         "of that axis' range: 0.05 is 21 points per parameter.")
parser.add_argument("--gt-noisy", action="store_true", default=False, dest="gt_noisy",
                    help="Draw the ground truth the way a run would have observed it, "
                         "noise and all, instead of the noiseless value underneath.")
parser.add_argument("--constraint", action="append", default=[],
                    help="Feasibility constraint as key:op:value (repeatable). "
                         "Only feasible experiments can be the best one; infeasible "
                         "ones are shown dimmed.")
args = parser.parse_args()
try:
    constraints = parse_constraints(args.constraint)
except ConstraintError as exc:
    print(exc)
    sys.exit(2)

# One or two parameters. Beyond that the plot would be a projection: points far apart in
# a parameter that has no axis land on top of each other, so a smooth landscape reads as
# scatter. The GUI offers the best-value evolution instead, which stays honest at any
# dimension.
par_keys = list(args.parameter)
if not 1 <= len(par_keys) <= 2:
    print(f"Give one or two parameters, not {len(par_keys)}. Above two the objective "
          f"cannot be drawn against them; plot the best value over the campaign instead.")
    sys.exit(2)

# ---- CONFIG ----
MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

SCATTER = fig_cfg["scatter"]

FONT_LABEL  = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]
DPI         = fig_cfg["dpi"]

# The landscape is a magnitude, not a divergence from a middle, so it takes a sequential
# map rather than the Pareto plot's coolwarm.
CMAP = cm.viridis

# Axis-label overrides by result key. The problem definition already names its
# objectives, so this is empty until a key needs a unit or LaTeX in the label.
PRETTY_NAMES = {}

# Larger is better once the sense is applied, so one comparison finds the best point
# whichever way the objective runs.
sense = -1.0 if args.objective in set(args.maximize) else 1.0


def _label(exp):
    return (exp.get("experiment_type") or exp.get("optimizer") or "unknown").lower()


def column(exp, key):
    """A column by name, from the results or from the parameters it was measured at."""
    results = exp.get("results", {})
    if key in results:
        return results[key]
    return (exp.get("parameters") or {}).get(key)


# ---- LOAD ----
raw_rows = []
for exp in load_experiments_from_map(MAP_PATH):
    r = exp.get("results", {})
    raw_rows.append({
        "label":    _label(exp),
        "group_id": exp["group_id"],
        "pars":     [column(exp, key) for key in par_keys],
        "obj":      column(exp, args.objective),
        # What the measurement said about its own uncertainty. to_json writes a
        # <label>_var next to every value, left null when the run measured none.
        "obj_var":  r.get(f"{args.objective}_var"),
        "feasible": is_feasible(r, constraints),
    })


def _complete(row):
    return row["obj"] is not None and all(v is not None for v in row["pars"])


# ---- AGGREGATE (if grouped) ----
# A group is the same evaluation of the same arm across replicate runs, so its members
# share their parameters and only the objective is averaged. Infeasible experiments are
# left out of the mean and drawn individually below, as in the Pareto plots.
if args.grouped:
    groups, order = {}, []
    for r in raw_rows:
        if not _complete(r) or not r["feasible"]:
            continue
        gid = r["group_id"]
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(r)

    rows = []
    for gid in order:
        items = groups[gid]
        objs = [it["obj"] for it in items]
        obj_mean = float(np.mean(objs))
        if args.errorbar == "minmax":
            err_lo = max(0.0, obj_mean - float(np.min(objs)))
            err_hi = max(0.0, float(np.max(objs)) - obj_mean)
        else:
            # Both modes reconcile the repeats' scatter with the recorded measurement
            # variance rather than adding them - see _uncertainty.
            estimate = mean_sd if args.errorbar == "sem" else total_sd
            err_lo = err_hi = estimate(objs, [it["obj_var"] for it in items])
        rows.append({
            "label":    items[0]["label"],
            "group_id": gid,
            # Shared within the group, so the first member's are the group's.
            "pars":     list(items[0]["pars"]),
            "obj":      obj_mean,
            "err_lo":   err_lo,
            "err_hi":   err_hi,
            "n":        len(items),
            "feasible": True,
        })
    valid = rows
else:
    valid = [r for r in raw_rows if _complete(r)]
    for r in valid:
        r["err_lo"] = r["err_hi"] = 0.0
        r["n"] = 1

if not valid:
    print("No data with the requested keys.")
    sys.exit(1)

# ---- INFEASIBLE (always individual points) ----
infeasible = [r for r in raw_rows if _complete(r) and not r["feasible"]]

all_labels = sorted({r["label"] for r in valid})
_label_color, _label_marker, _ = styler(fig_cfg, all_labels)

# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["objective"])
legend_handles = []
mappable = None
two_par = len(par_keys) == 2

# ---- GROUND TRUTH ----
# Under everything: it is the backdrop the campaign is read against, not a series. On one
# parameter it is the true curve; on two it is the surface the observations sit in, which
# is also what sets the colour scale the observations are drawn on.
gt_norm = None
if args.ground_truth:
    from pybo_gui.modules.bayesian_campaign_analysis._ground_truth import objective_landscape
    gt_cols, gt_vals, gt_shape = objective_landscape(
        args.ground_truth, args.objective, par_keys,
        args.gt_method, args.gt_samples, args.gt_spacing, args.gt_noisy)
    gt_color = fig_cfg["colors"].get("ground_truth", "#8A8F98")
    if gt_vals and not two_par:
        # Sorted, so a line through the samples is the function rather than a scribble.
        curve = sorted(zip(gt_cols[0], gt_vals))
        ax.plot([p[0] for p in curve], [p[1] for p in curve],
                color=gt_color, linewidth=1.2, zorder=1)
        legend_handles.append(mlines.Line2D([], [], color=gt_color, linewidth=1.2,
                                            label="Ground truth"))
    elif gt_vals:
        finite = [v for v in gt_vals if np.isfinite(v)]
        gt_norm = mcolors.Normalize(vmin=min(finite), vmax=max(finite)) if finite else None
        if gt_shape is not None:
            # A grid keeps its shape, so the landscape is a filled contour - the same
            # reading as pybo's own Experiment2DPlotter. Infeasible samples come back as
            # NaN and contourf leaves them blank, so the plot shows where the problem
            # forbids as a hole rather than as an invented level.
            grid_x = np.asarray(gt_cols[0]).reshape(gt_shape)
            grid_y = np.asarray(gt_cols[1]).reshape(gt_shape)
            grid_z = np.ma.masked_invalid(np.asarray(gt_vals).reshape(gt_shape))
            mappable = ax.contourf(grid_x, grid_y, grid_z, levels=100, cmap=CMAP,
                                   norm=gt_norm, alpha=0.8, zorder=0)
        else:
            # A random cloud has no grid to contour, so it is scattered as it was drawn.
            mappable = ax.scatter(gt_cols[0], gt_cols[1], c=gt_vals, cmap=CMAP,
                                  norm=gt_norm, s=SCATTER["marker_size"] * 0.25,
                                  marker=".", edgecolors="none", alpha=0.45, zorder=0)

# ---- COLOUR SCALE (two parameters) ----
# On two parameters the objective has no axis of its own, so it has to be the colour -
# but only when there is no landscape under the points. Where the surface is drawn it
# already shows the objective everywhere, and colouring the observations on that same
# scale paints each one the shade of the ground it stands on, which is what made them
# disappear into it. There they keep their series colour instead, as the observations in
# pybo's own Experiment2DPlotter do.
color_by_obj = two_par and mappable is None
if color_by_obj:
    obj_vals = [r["obj"] for r in valid] + [r["obj"] for r in infeasible]
    gt_norm = mcolors.Normalize(vmin=min(obj_vals), vmax=max(obj_vals))

# ---- OBSERVATIONS ----
for lbl in all_labels:
    marker = _label_marker(lbl)
    subset = [r for r in valid if r["label"] == lbl and r["feasible"]]
    if not subset:
        continue
    face_color = _label_color(lbl)

    if two_par:
        shared = dict(s=SCATTER["marker_size"], marker=marker,
                      edgecolors=SCATTER["edge_color"], linewidths=SCATTER["edge_width"],
                      alpha=SCATTER["alpha"], zorder=3)
        xs = [r["pars"][0] for r in subset]
        ys = [r["pars"][1] for r in subset]
        if color_by_obj:
            mappable = ax.scatter(xs, ys, c=[r["obj"] for r in subset], cmap=CMAP,
                                  norm=gt_norm, **shared)
        else:
            ax.scatter(xs, ys, facecolors=face_color, **shared)
    elif args.grouped:
        for r in subset:
            # Drawn whenever there is something to draw, not only when the group has
            # repeats: a single measurement with a recorded variance has a real bar.
            yerr = [[r["err_lo"]], [r["err_hi"]]] if r["err_hi"] else None
            ax.errorbar(
                [r["pars"][0]], [r["obj"]], yerr=yerr, fmt=marker,
                markerfacecolor=face_color, markeredgecolor=SCATTER["edge_color"],
                markeredgewidth=SCATTER["edge_width"], ecolor=face_color,
                markersize=SCATTER["marker_size"] ** 0.5, linewidth=0.8, capsize=3,
                alpha=SCATTER["alpha"], zorder=3,
            )
    else:
        ax.scatter(
            [r["pars"][0] for r in subset], [r["obj"] for r in subset],
            s=SCATTER["marker_size"], marker=marker, facecolors=face_color,
            edgecolors=SCATTER["edge_color"], linewidths=SCATTER["edge_width"],
            alpha=SCATTER["alpha"], zorder=3,
        )

    legend_handles.append(
        mlines.Line2D([], [], linestyle="None", marker=marker,
                      markerfacecolor="gray" if color_by_obj else face_color,
                      markeredgecolor=SCATTER["edge_color"], markeredgewidth=0.8,
                      markersize=SCATTER["marker_size"] ** 0.5, label=lbl.capitalize())
    )

# Infeasible points (violating a constraint): individual dimmed "X" markers in both
# grouped and non-grouped mode, and never the best one.
if infeasible:
    ax.scatter(
        [r["pars"][0] for r in infeasible],
        [r["pars"][1] if two_par else r["obj"] for r in infeasible],
        s=SCATTER["marker_size"], marker="X", facecolors="lightgray",
        edgecolors="none", alpha=0.30, zorder=1,
    )
    legend_handles.append(
        mlines.Line2D([], [], linestyle="None", marker="X", markerfacecolor="lightgray",
                      markeredgecolor="none",
                      markersize=SCATTER["marker_size"] ** 0.5, label="Infeasible")
    )

if args.show_numbers:
    src = [r for r in valid if r["feasible"]] + infeasible
    for r in src:
        ax.annotate(str(r["group_id"]),
                    (r["pars"][0], r["pars"][1] if two_par else r["obj"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=FONT_LEGEND - 2, color="dimgray")

# ---- BEST OBSERVATION ----
# One objective has no front, so what a Pareto plot circles collapses to a single point:
# the best feasible observation. Circled per *base* label, as the fronts are, so a run's
# initial design and its proposals are scored as the one campaign they are.
for base in sorted({base_label(lbl) for lbl in all_labels}):
    members = [r for r in valid if base_label(r["label"]) == base and r["feasible"]]
    if not members:
        continue
    best = min(members, key=lambda r: sense * r["obj"])
    ax.scatter([best["pars"][0]], [best["pars"][1] if two_par else best["obj"]],
               s=SCATTER["marker_size"] * 1.6,
               marker=_label_marker(best["label"]), facecolors="none",
               edgecolors=_label_color(base), linewidths=1.0, zorder=4)
legend_handles.append(
    mlines.Line2D([], [], linestyle="None", marker="o", markerfacecolor="none",
                  markeredgecolor="dimgray", markeredgewidth=1.0,
                  markersize=SCATTER["marker_size"] ** 0.5, label="Best")
)

# ---- AXES ----
def _par_label(i):
    override = args.parameter_label[i] if i < len(args.parameter_label) else ""
    return override or PRETTY_NAMES.get(par_keys[i], par_keys[i])


obj_label = args.objective_label or PRETTY_NAMES.get(args.objective, args.objective)
ax.set_xlabel(_par_label(0), fontsize=FONT_LABEL)
ax.set_ylabel(_par_label(1) if two_par else obj_label, fontsize=FONT_LABEL)

ax.tick_params(labelsize=FONT_LABEL - 1)
ax.grid(True, **fig_cfg["grid"])
leg_cfg = fig_cfg["legend"]
# Long run names in quantity outgrow a corner at the normal font size, so past a
# handful this shrinks the legend (smaller type, more columns) to keep it inside
# the axes rather than let it spill past them.
place_legend(fig, ax, legend_handles, leg_cfg, FONT_LEGEND)

if two_par and mappable is not None:
    cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
    cbar.set_label(obj_label, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_LABEL - 1)

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")