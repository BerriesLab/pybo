import argparse
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.modules.bayesian_campaign_analysis._labels import base_label, is_initial, styler
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map

parser = argparse.ArgumentParser(description="Per-parameter/result evolution plot.")
parser.add_argument("--grouped", action="store_true", default=False,
                    help="Average replicate experiments per group_id into one point.")
args = parser.parse_args()

# ---- CONFIG ----
MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

# The design is recognised by its label rather than by a fixed list, since the label
# is qualified by whatever the map was built by ("bayesian_run3 (initial)").
ITERATION_KEY      = "iteration"                 # metadata field; init = 0

MARKERS      = fig_cfg["markers"]["label"]
LABEL_COLORS = fig_cfg["colors"]["label"]
SCATTER      = fig_cfg["scatter"]

FONT_LABEL  = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]
FONT_TITLE  = fig_cfg["font"]["title"]
DPI         = fig_cfg["dpi"]

# Pretty axis labels for known keys; unknown keys fall back to the key name itself
# Axis-label overrides by result key. The problem definition already names its
# objectives, so this is empty until a key needs a unit or LaTeX in the label.
PRETTY_NAMES = {}


def _label(exp):
    return (exp.get("experiment_type") or exp.get("optimizer") or "unknown").lower()


# ---- LOAD ----
experiments = load_experiments_from_map(MAP_PATH)

def _is_numeric_key(key, source, exps):
    seen = False
    for exp in exps:
        v = exp.get(source, {}).get(key)
        if v is None:
            continue
        seen = True
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False
    return seen

param_keys  = sorted({k for exp in experiments for k in exp.get("parameters", {})
                      if _is_numeric_key(k, "parameters", experiments)})
result_keys = sorted({k for exp in experiments for k in exp.get("results", {})
                      if _is_numeric_key(k, "results", experiments)})
all_cols   = param_keys + result_keys
col_source = {k: "parameters" for k in param_keys}
col_source.update({k: "results" for k in result_keys})

rows = []
for exp in experiments:
    rows.append({
        "index":         exp["index"],
        "iteration":     exp.get(ITERATION_KEY),
        "group_id":      exp["group_id"],
        "run":           exp.get("run"),
        "label":         _label(exp),
        "parameters":    exp.get("parameters", {}),
        "results":       exp.get("results", {}),
    })

# In grouped mode, collapse replicate experiments (same group_id) to one row whose
# results are the elementwise mean; label/iteration/index come from the group's
# first (chronological) experiment, parameters are shared within the group.
if args.grouped:
    grouped, order = {}, []
    for r in rows:
        gid = r["group_id"]
        if gid not in grouped:
            grouped[gid] = []
            order.append(gid)
        grouped[gid].append(r)

    def _mean_results(items):
        out = {}
        for k in result_keys:
            vals = [it["results"].get(k) for it in items]
            vals = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            out[k] = sum(vals) / len(vals) if vals else None
        return out

    rows = [{
        "index":      grouped[gid][0]["index"],
        "iteration":  grouped[gid][0]["iteration"],
        "group_id":   gid,
        "run":        grouped[gid][0]["run"],
        "label":      grouped[gid][0]["label"],
        "parameters": grouped[gid][0]["parameters"],
        "results":    _mean_results(grouped[gid]),
    } for gid in order]

all_labels = sorted({r["label"] for r in rows})
_color, _marker, _, _ = styler(fig_cfg, all_labels)

# ---- EVOLUTION GEOMETRY ----
# x is "number of observations": the initial-design block sits on one vertical
# line at x = n_init, and each Bayesian batch sits at the cumulative observation
# count through that batch (n_init + i*q when every batch has size q). The block
# sequence [init, step_1, step_2, ...] drives the interconnection lines.
#
# Each run is its own campaign and counts its own observations, so the runs of a
# multi-run selection overlap on the x axis instead of being laid end to end. A map
# built without a run field (or from a single run) collapses to one campaign, which
# is the single-sequence view. base_label keeps a run's design with its proposals.
rows.sort(key=lambda r: r["index"])  # chronological (already sorted by start_time)

campaigns = {}
for r in rows:
    campaigns.setdefault(r["run"] or base_label(r["label"]), []).append(r)

# One [init, step_1, ...] sequence per campaign: the lines connect within a run only.
campaign_blocks = []
n_inits         = []
for items in campaigns.values():
    init_block    = [r for r in items if is_initial(r["label"])]
    bayesian_rows = [r for r in items if not is_initial(r["label"])]
    n_init        = len(init_block)
    n_inits.append(n_init)

    # Group Bayesian rows into steps by their shared iteration number; rows missing a
    # (positive) iteration fall back to q=1 — each becomes its own singleton step.
    numbered, unnumbered = {}, []
    for r in bayesian_rows:
        it = r["iteration"]
        if isinstance(it, int) and not isinstance(it, bool) and it > 0:
            numbered.setdefault(it, []).append(r)
        else:
            unnumbered.append(r)
    steps = [numbered[k] for k in sorted(numbered)] + [[r] for r in unnumbered]

    for r in init_block:
        r["x"] = n_init
    cumulative_obs = 0
    for step in steps:
        cumulative_obs += len(step)
        for r in step:
            r["x"] = n_init + cumulative_obs

    campaign_blocks.append([init_block, *steps])

# ---- PLOT ----
CONNECT = {"color": "#888888", "linewidth": fig_cfg["grid"]["linewidth"],
           "alpha": 0.4, "zorder": 1}

# Build legend handles once — same set of labels appears in every figure.
legend_handles = {}
for lbl in all_labels:
    style_key  = "initial" if is_initial(lbl) else lbl
    marker     = _marker(lbl)
    face_color = _color(lbl)
    if style_key not in legend_handles:
        legend_handles[style_key] = mlines.Line2D(
            [], [], linestyle="None", marker=marker,
            markerfacecolor=face_color, markeredgecolor=SCATTER["edge_color"],
            markeredgewidth=0.8, markersize=SCATTER["marker_size"] ** 0.5,
            label=style_key.capitalize()
        )

leg_cfg = fig_cfg["legend"]

for col in all_cols:
    source = col_source[col]
    pretty = PRETTY_NAMES.get(col, col)

    fig, ax = plt.subplots(1, 1, figsize=fig_cfg["figsize"]["evolution"])

    # Fully-connect every point in one block to every point in the next, skipping
    # points whose value for this column is missing. A campaign at a time, so no line
    # ever joins two runs.
    for blocks in campaign_blocks:
        for prev, nxt in zip(blocks, blocks[1:]):
            for a in prev:
                ya = a[source].get(col)
                if ya is None:
                    continue
                for b in nxt:
                    yb = b[source].get(col)
                    if yb is None:
                        continue
                    ax.plot([a["x"], b["x"]], [ya, yb], **CONNECT)

    for lbl in all_labels:
        style_key  = "initial" if is_initial(lbl) else lbl
        marker     = _marker(lbl)
        face_color = _color(lbl)
        subset = [r for r in rows if r["label"] == lbl and r[source].get(col) is not None]
        xs = [r["x"] for r in subset]
        ys = [r[source].get(col) for r in subset]
        ax.scatter(xs, ys,
                   s=SCATTER["marker_size"] * 0.7, marker=marker,
                   facecolors=face_color, edgecolors=SCATTER["edge_color"],
                   linewidths=SCATTER["edge_width"], alpha=SCATTER["alpha"], zorder=3)

    # One line per distinct design size: runs that share an n_init share the line.
    for x_init in sorted({n for n in n_inits if n}):
        ax.axvline(x=x_init, **fig_cfg["refline"]["delay"])
    ax.set_title(pretty, fontsize=FONT_TITLE)
    ax.set_xlabel("Number of observations", fontsize=FONT_LABEL)
    ax.set_ylabel(pretty, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_LABEL - 1)
    ax.grid(True, **fig_cfg["grid"])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(handles=list(legend_handles.values()), loc=leg_cfg["loc"],
              fontsize=FONT_LEGEND, frameon=leg_cfg["frameon"],
              framealpha=leg_cfg["framealpha"])
    fig.tight_layout(pad=fig_cfg["layout_pad"])

plt.show(block=__name__ == "__main__")
