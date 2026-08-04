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
from pybo_gui.modules.bayesian_campaign_analysis._labels import styler
from pybo_gui.modules.bayesian_campaign_analysis._uncertainty import total_sd, mean_sd
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map
from pybo_gui.modules.bayesian_campaign_analysis._constraints import parse_constraints, is_feasible, ConstraintError

parser = argparse.ArgumentParser()
parser.add_argument("--x", required=True, help="Result key for x axis (objective)")
parser.add_argument("--y", required=True, help="Result key for y axis (objective)")
parser.add_argument("--z", required=True, help="Result key for z axis (objective; also mapped to color)")
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
DPI         = fig_cfg["dpi"]

# Axis-label overrides by result key. The problem definition already names its
# objectives, so this is empty until a key needs a unit or LaTeX in the label.
PRETTY_NAMES = {}


def _label(exp):
    return (exp.get("experiment_type") or exp.get("technology") or "unknown").lower()


def pareto_front_3d_mask(pts, signs=(1.0, 1.0, 1.0)):
    """Return a boolean mask of non-dominated points. `signs` is the per-axis
    sense (+1 minimize, -1 maximize); domination is evaluated in signed space.
    pts: list of (x, y, z) tuples (real coordinates). O(n^2)."""
    sx, sy, sz = signs
    spts = [(sx * x, sy * y, sz * z) for x, y, z in pts]
    n = len(spts)
    mask = [True] * n
    for i in range(n):
        xi, yi, zi = spts[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj, zj = spts[j]
            if xj <= xi and yj <= yi and zj <= zi and (xj < xi or yj < yi or zj < zi):
                mask[i] = False
                break
    return mask


# ---- LOAD ----
def column(exp, key):
    """A column by name, from the results or from the parameters it was measured at.

    The axis pickers offer both, and a parameter is never in results - looking only
    there left a parameter axis empty.
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
        "z_val":    column(exp, args.z),
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
        if r["x"] is None or r["y"] is None or r["z_val"] is None or not r["feasible"]:
            continue
        gid = r["group_id"]
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(r)

    rows = []
    for gid in order:
        items = groups[gid]
        xs = [it["x"]     for it in items]
        ys = [it["y"]     for it in items]
        zs = [it["z_val"] for it in items]
        x_mean, y_mean = float(np.mean(xs)), float(np.mean(ys))
        if args.errorbar == "minmax":
            x_err_lo = x_mean - float(np.min(xs))
            x_err_hi = float(np.max(xs)) - x_mean
            y_err_lo = y_mean - float(np.min(ys))
            y_err_hi = float(np.max(ys)) - y_mean
        else:
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
            "z_val":    float(np.mean(zs)),
            "n":        len(items),
            "feasible": True,
        })
    valid = rows
else:
    valid = [r for r in raw_rows
             if r["x"] is not None and r["y"] is not None and r["z_val"] is not None]
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
              if r["x"] is not None and r["y"] is not None
              and r["z_val"] is not None and not r["feasible"]]

# ---- COLOR SCALE (z) ----
# Span all plotted points: feasible from `valid` (group means when grouped),
# infeasible from the individual points above. The marker encodes feasibility.
z_vals = [r["z_val"] for r in valid if r["feasible"]]
z_vals += [r["z_val"] for r in infeasible]
if z_vals:
    z_cmap = cm.coolwarm
    z_norm = mcolors.Normalize(vmin=min(z_vals), vmax=max(z_vals))
else:
    z_cmap = cm.coolwarm
    z_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

# ---- 3D PARETO FRONT ----
# The non-dominated set is computed over feasible points only; infeasible
# points are never Pareto-optimal.
maximize = set(args.maximize)
signs = (-1.0 if args.x in maximize else 1.0,
         -1.0 if args.y in maximize else 1.0,
         -1.0 if args.z in maximize else 1.0)
feasible_rows = [r for r in valid if r["feasible"]]
nd_mask = pareto_front_3d_mask(
    [(r["x"], r["y"], r["z_val"]) for r in feasible_rows], signs)
for r in valid:
    r["nd"] = False
for r, m in zip(feasible_rows, nd_mask):
    r["nd"] = m

# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["pareto"])

all_labels     = sorted({r["label"] for r in valid})
_color, _marker, _front_style = styler(fig_cfg, all_labels)
legend_handles = []
_last_sc       = None

for lbl in all_labels:
    marker = _marker(lbl)
    subset = [r for r in valid if r["label"] == lbl and r["feasible"]]
    if not subset:
        continue

    if args.grouped:
        for r in subset:
            fc = z_cmap(z_norm(r["z_val"]))
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
                ecolor="gray",
                markersize=SCATTER["marker_size"] ** 0.5,
                linewidth=0.8,
                capsize=3,
                alpha=SCATTER["alpha"],
                zorder=3,
            )
        _last_sc = ax.scatter(
            [r["x"] for r in subset], [r["y"] for r in subset],
            c=[r["z_val"] for r in subset], cmap=z_cmap, norm=z_norm,
            s=0, zorder=0,
        )
    else:
        _last_sc = ax.scatter(
            [r["x"] for r in subset], [r["y"] for r in subset],
            c=[r["z_val"] for r in subset], cmap=z_cmap, norm=z_norm,
            s=SCATTER["marker_size"], marker=marker,
            edgecolors=SCATTER["edge_color"],
            linewidths=SCATTER["edge_width"], alpha=SCATTER["alpha"], zorder=3,
        )

    legend_handles.append(
        mlines.Line2D([], [], linestyle="None", marker=marker,
                      markerfacecolor="gray", markeredgecolor=SCATTER["edge_color"],
                      markeredgewidth=0.8, markersize=SCATTER["marker_size"] ** 0.5, label=lbl.capitalize())
    )

# Infeasible points (violating a constraint): drawn as individual filled "X"
# markers (in both grouped and non-grouped mode) and excluded from the front.
# They are coloured on the shared z-scale (which spans all points, so their true
# z value shows).
if infeasible:
    ax.scatter(
        [r["x"] for r in infeasible], [r["y"] for r in infeasible],
        c=[r["z_val"] for r in infeasible], cmap=z_cmap, norm=z_norm,
        s=SCATTER["marker_size"], marker="X",
        edgecolors="black", linewidths=0.6, alpha=0.55, zorder=1,
    )
    legend_handles.append(
        mlines.Line2D([], [], linestyle="None", marker="X",
                      markerfacecolor="gray", markeredgecolor="black",
                      markersize=SCATTER["marker_size"] ** 0.5, label="Infeasible")
    )

# Highlight 3D non-dominated points
nd_rows = [r for r in valid if r["nd"]]
if nd_rows:
    ax.scatter(
        [r["x"] for r in nd_rows], [r["y"] for r in nd_rows],
        c=[r["z_val"] for r in nd_rows], cmap=z_cmap, norm=z_norm,
        s=SCATTER["marker_size"] * 1.6, marker="o",
        edgecolors="black", linewidths=1.6, alpha=1.0, zorder=5,
    )
    legend_handles.append(
        mlines.Line2D([], [], linestyle="None", marker="o",
                      markerfacecolor="white", markeredgecolor="black",
                      markeredgewidth=1.6, markersize=SCATTER["marker_size"] ** 0.5,
                      label="Pareto-optimal (3D)")
    )

if args.show_numbers:
    for r in (r for r in valid if r["feasible"]):
        ax.annotate(str(r["group_id"]),
                    (r["x"], r["y"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=FONT_LEGEND - 2, color="dimgray")

xlabel = args.xlabel if args.xlabel else PRETTY_NAMES.get(args.x, args.x)
ylabel = args.ylabel if args.ylabel else PRETTY_NAMES.get(args.y, args.y)
ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

ax.tick_params(labelsize=FONT_LABEL - 1)
ax.grid(True, **fig_cfg["grid"])
leg_cfg = fig_cfg["legend"]
ax.legend(handles=legend_handles, fontsize=FONT_LEGEND,
          loc=leg_cfg["loc"], frameon=leg_cfg["frameon"], framealpha=leg_cfg["framealpha"])

if _last_sc is not None:
    zlabel = args.zlabel if args.zlabel else PRETTY_NAMES.get(args.z, args.z)
    cbar = fig.colorbar(_last_sc, ax=ax, pad=0.02)
    cbar.set_label(zlabel, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_LABEL - 1)

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
