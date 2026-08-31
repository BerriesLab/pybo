import argparse
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.utils.experiment_map_loader import (
    GROUP_IDENTITY_KEYS, load_experiments_from_map)
from pybo_gui.modules.bayesian_campaign_analysis._series import (
    GROUP_KEYS, GroupKeyError, group_key, parse_keys, series_label)

parser = argparse.ArgumentParser(description="Per-result boxplot over the chosen groups.")
parser.add_argument("--group-by", action="append", default=None, dest="group_by",
                    metavar="KEY",
                    help="A key to group the selected runs by (repeatable). Records "
                         "agreeing on every key given are one box, drawn over the values "
                         "they hold. Naming all of them - the default - gives one box per "
                         "setting per run, which is what this plot always drew. "
                         f"Available: {', '.join(GROUP_KEYS)}.")
args = parser.parse_args()
try:
    keys = parse_keys(GROUP_KEYS if args.group_by is None else args.group_by)
except GroupKeyError as exc:
    print(exc)
    sys.exit(2)

# group_key, not merge_key: merge_key pins the run whatever is ticked, because the spread
# an *error bar* claims is measurement noise and only a re-measurement within a run is
# that. A box claims no such thing - it draws the distribution of whatever fell in it - so
# unticking run here has to actually pool across runs, which is the one view a
# variability study of a repeated setting is drawn for.

# ---- CONFIG ----
MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

FONT_LABEL = fig_cfg["font"]["label"]
FONT_TITLE = fig_cfg["font"]["title"]
DPI = fig_cfg["dpi"]
SCATTER = fig_cfg["scatter"]

# ---- THE RIG'S GRID ----
# Grouping by `parameters` has to compare them on the rig's own steps, or a setting
# recorded once rounded and once not stays two settings. Only the problem knows those
# steps and this script is handed none, so the snapped values are read back out of
# group_map.json, which build_group_map already wrote them to - the same grid the
# group_id was assigned on, rather than a second one that could drift from it. Absent
# that file the parameters are compared as recorded, which is what every map got before
# resolutions existed.
settings = {}
group_map_path = os.path.join(os.path.dirname(MAP_PATH), "group_map.json")
if os.path.exists(group_map_path):
    with open(group_map_path, encoding="utf-8") as file:
        settings = {g["group_id"]: tuple(sorted((k, v) for k, v in g.items()
                                                if k not in GROUP_IDENTITY_KEYS))
                    for g in json.load(file)}

# ---- LOAD & GROUP ----
groups = {}      # gkey -> list of result dicts
names = {}       # gkey -> what the ticked keys still call it
group_order = [] # gkeys in first-appearance order
all_keys = []    # result keys in first-appearance order

for exp in load_experiments_from_map(MAP_PATH):
    # The parameters part comes off the map's own setting rather than out of group_key,
    # so it is the snapped one; the rest of the keys are plain fields group_key reads.
    gkey = (group_key(exp, [k for k in keys if k != "parameters"]),
            settings.get(exp["group_id"], tuple(sorted((exp.get("parameters") or {}).items())))
            if "parameters" in keys else None)
    if gkey not in groups:
        groups[gkey] = []
        names[gkey] = series_label(exp, keys, fallback="")
        group_order.append(gkey)
    results = exp.get("results", {})
    groups[gkey].append(results)
    for k in results:
        if k not in all_keys:
            all_keys.append(k)

# A name every box shares tells them apart from nothing, so it is dropped rather than
# repeated: one run selected with `run` ticked would otherwise print that run's name on
# every tick, which is how this plot is most often drawn.
if len(set(names.values())) < 2:
    names = {gkey: "" for gkey in names}

n_groups = len(group_order)
# The ordinal is the only compact name a setting has - the parameters themselves are too
# long for a tick - so it is shown whenever `parameters` is ticked, alongside whatever the
# other keys call the box. With every key ticked it counts the same groups in the same
# order build_group_map did, so the numbers are still the group ids the map carries.
group_labels = [" ".join(part for part in
                         (str(position) if "parameters" in keys else "", names[gkey])
                         if part) or str(position)
                for position, gkey in enumerate(group_order, start=1)]
positions = list(range(1, n_groups + 1))
fig_width = max(4, 1.5 + 0.1 * n_groups)

# ---- ONE FIGURE PER RESULT KEY ----
for obj_key in all_keys:
    obj_label = obj_key.replace("_", " ")
    data_by_group = [
        [r[obj_key] for r in groups[gkey] if isinstance(r.get(obj_key), (int, float))]
        for gkey in group_order
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
    ax.set_xlabel("Group", fontsize=FONT_LABEL)
    ax.set_ylabel(obj_label, fontsize=FONT_LABEL)
    # What a box stands for is the one thing the axis cannot show, and it changes from
    # one drawing to the next, so the figure says it rather than leaving it in the tab.
    ax.set_title("grouped by " + ", ".join(keys), fontsize=FONT_TITLE)

    ax.tick_params(labelsize=FONT_LABEL - 1)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5, axis="y")
    ax.set_xlim(0.5, n_groups + 0.5)

    fig.tight_layout(pad=fig_cfg["layout_pad"])

plt.show(block=__name__ == "__main__")
