import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map

# ---- CONFIG ----
MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

FONT_LABEL = fig_cfg["font"]["label"]
FONT_TITLE = fig_cfg["font"]["title"]
DPI = fig_cfg["dpi"]
SCATTER = fig_cfg["scatter"]

# ---- LOAD & GROUP ----
groups = {}      # group_id -> list of result dicts
group_order = [] # group_ids in first-appearance order
all_keys = []    # result keys in first-appearance order

for exp in load_experiments_from_map(MAP_PATH):
    gid = exp["group_id"]
    if gid not in groups:
        groups[gid] = []
        group_order.append(gid)
    results = exp.get("results", {})
    groups[gid].append(results)
    for k in results:
        if k not in all_keys:
            all_keys.append(k)

n_groups = len(group_order)
group_labels = [str(gid) for gid in group_order]
positions = list(range(1, n_groups + 1))
fig_width = max(4, 1.5 + 0.1 * n_groups)

# ---- ONE FIGURE PER RESULT KEY ----
for obj_key in all_keys:
    obj_label = obj_key.replace("_", " ")
    data_by_group = [
        [r[obj_key] for r in groups[gid] if isinstance(r.get(obj_key), (int, float))]
        for gid in group_order
    ]

    fig, ax = plt.subplots(figsize=(fig_width, 5))

    box_data = [v for v in data_by_group if len(v) > 1]
    box_pos = [p for p, v in zip(positions, data_by_group) if len(v) > 1]

    if box_data:
        ax.boxplot(
            box_data,
            positions=box_pos,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", color="steelblue"),
            medianprops=dict(color="navy", linewidth=1.8),
            whiskerprops=dict(color="steelblue"),
            capprops=dict(color="steelblue"),
            flierprops=dict(marker="o", markerfacecolor="steelblue", markersize=SCATTER["marker_size"] ** 0.5, linestyle="none"),
        )

    for pos, vals in zip(positions, data_by_group):
        ax.scatter([pos] * len(vals), vals, color="steelblue", s=SCATTER["marker_size"], zorder=5, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=FONT_LABEL - 1, rotation=45, ha="right")
    ax.set_xlabel("Parameter group", fontsize=FONT_LABEL)
    ax.set_ylabel(obj_label, fontsize=FONT_LABEL)

    ax.tick_params(labelsize=FONT_LABEL - 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, axis="y")
    ax.set_xlim(0.5, n_groups + 0.5)

    fig.tight_layout(pad=fig_cfg["layout_pad"])

plt.show(block=__name__ == "__main__")
