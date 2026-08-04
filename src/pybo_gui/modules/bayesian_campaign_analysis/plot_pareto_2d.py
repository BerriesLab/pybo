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

parser = argparse.ArgumentParser()
parser.add_argument("--x", required=True, help="Result key for x axis (objective)")
parser.add_argument("--y", required=True, help="Result key for y axis (objective)")
parser.add_argument("--z", default="",    help="Result key for color coding (color only, not used in Pareto)")
parser.add_argument("--xlabel", default="", help="Override x-axis label (LaTeX accepted via $...$)")
parser.add_argument("--ylabel", default="", help="Override y-axis label (LaTeX accepted via $...$)")
parser.add_argument("--zlabel", default="", help="Override colorbar label (LaTeX accepted via $...$)")
parser.add_argument("--grouped",      action="store_true", default=False,
                    help="Aggregate per group_id")
parser.add_argument("--errorbar", choices=["std", "minmax"], default="std",
                    help="Error-bar mode in grouped view (std = mean ± std, minmax = mean to min/max)")
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

raw_rows = []
for exp in load_experiments_from_map(MAP_PATH):
    r = exp.get("results", {})
    raw_rows.append({
        "label":    _label(exp),
        "group_id": exp["group_id"],
        "x":        r.get(args.x),
        "y":        r.get(args.y),
        "z_val":    r.get(args.z) if use_z else None,
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
            sd_x = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
            sd_y = float(np.std(ys, ddof=1)) if len(ys) > 1 else 0.0
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
FRONT_LINESTYLES = {"initial": ":", "sobol": "-.", "lhs": "-.", "manual": "-."}
FRONT_COLOR      = "#888888"  # fallback for a label the style gives no colour


# Labels the style names explicitly keep their colour; anything else - a run directory,
# say - takes one from the cycle by position, so every series is distinct and keeps the
# same colour across the plots of one campaign.
_COLOR_CYCLE = fig_cfg["colors"].get("cycle") or [FRONT_COLOR]
INITIAL_SUFFIX = " (initial)"


def _is_initial(lbl):
    return lbl == "initial" or lbl.endswith(INITIAL_SUFFIX)


def _base_label(lbl):
    """A design series and the proposals it belongs to share a base, so they share a
    colour and sit together in the legend; the linestyle and marker tell them apart."""
    return lbl[:-len(INITIAL_SUFFIX)] if lbl.endswith(INITIAL_SUFFIX) else lbl


def _label_color(lbl):
    base = _base_label(lbl)
    if base in LABEL_COLORS:
        return LABEL_COLORS[base]
    order = sorted({_base_label(r["label"]) for r in valid} - set(LABEL_COLORS))
    return _COLOR_CYCLE[order.index(base) % len(_COLOR_CYCLE)] if base in order else FRONT_COLOR


def _label_marker(lbl):
    """The design keeps its own marker whatever it qualifies, so it reads as exploration."""
    if _is_initial(lbl):
        return MARKERS.get("initial", "^")
    return MARKERS.get(_base_label(lbl), "o")


# Dash patterns for the two kinds of front. A phase offset per series is what keeps
# coincident fronts visible: two arms started from one seed share an initial design
# exactly, so their design fronts sit on the same points, and without the offset the one
# drawn last simply hides the other.
_DASHES = {"initial": (1, 3), "proposed": (6, 2)}


def _front_style(lbl):
    pattern = _DASHES["initial"] if _is_initial(lbl) else _DASHES["proposed"]
    # Spread the phases of the series sharing a pattern evenly over its period, so no two
    # of them land on the same offset and cancel the whole point of offsetting.
    peers = [l for l in FRONT_LABELS if _is_initial(l) == _is_initial(lbl)]
    period = sum(pattern)
    phase = period * peers.index(lbl) / len(peers) if lbl in peers else 0.0
    return phase, pattern

# ---- PLOT ----
fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["pareto"])

all_labels     = sorted({r["label"] for r in valid})
legend_handles = []
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
            xerr = [[r["x_err_lo"]], [r["x_err_hi"]]] if r["n"] > 1 else None
            yerr = [[r["y_err_lo"]], [r["y_err_hi"]]] if r["n"] > 1 else None
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
        mlines.Line2D([], [], linestyle="None", marker=marker,
                      markerfacecolor=legend_face, markeredgecolor=SCATTER["edge_color"],
                      markeredgewidth=0.8, markersize=SCATTER["marker_size"] ** 0.5, label=lbl.capitalize())
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
    legend_handles.append(
        mlines.Line2D([], [], color=color, linewidth=1.0, linestyle=ls,
                      label=f"Pareto front ({expl_lbl})")
    )

xlabel = args.xlabel if args.xlabel else PRETTY_NAMES.get(args.x, args.x)
ylabel = args.ylabel if args.ylabel else PRETTY_NAMES.get(args.y, args.y)
ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
ax.set_ylabel(ylabel, fontsize=FONT_LABEL)

ax.tick_params(labelsize=FONT_LABEL - 1)
ax.grid(True, **fig_cfg["grid"])
leg_cfg = fig_cfg["legend"]
ax.legend(handles=legend_handles, fontsize=FONT_LEGEND,
          loc="best", frameon=leg_cfg["frameon"], framealpha=leg_cfg["framealpha"])

if use_z and _last_sc is not None:
    zlabel = args.zlabel if args.zlabel else PRETTY_NAMES.get(args.z, args.z)
    cbar = fig.colorbar(_last_sc, ax=ax, pad=0.02)
    cbar.set_label(zlabel, fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_LABEL - 1)

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
