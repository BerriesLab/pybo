import argparse
import json
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg, scale_figsize
from pybo_gui.configs.settings import data_path
from pybo_gui.utils.experiment_map_loader import (
    GROUP_IDENTITY_KEYS, load_experiments_from_map)
from pybo_gui.modules.bayesian_campaign_analysis._series import (
    GROUP_KEYS, GroupKeyError, group_key, parse_keys)

parser = argparse.ArgumentParser(description="Correlation matrix over the chosen groups.")
parser.add_argument("--group-by", action="append", default=None, dest="group_by",
                    metavar="KEY",
                    help="A key to group the selected runs by (repeatable). Records "
                         "agreeing on every key given are one row, at the mean of what "
                         "they measured. Naming all of them - the default - leaves one "
                         "row per setting per run, so a setting a run measured twice is "
                         "correlated at its mean rather than as two rows. "
                         f"Available: {', '.join(GROUP_KEYS)}.")
args = parser.parse_args()
try:
    keys = parse_keys(GROUP_KEYS if args.group_by is None else args.group_by)
except GroupKeyError as exc:
    print(exc)
    sys.exit(2)

# group_key, not merge_key: merge_key pins the run whatever is ticked, and a correlation
# over one setting measured in several runs is exactly what unticking `run` asks for.

MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

# ---- THE RIG'S GRID ----
# The snapped parameters build_group_map already wrote, read back rather than re-snapped:
# only the problem knows the rig's steps and this script is handed none, and reusing the
# grid the group_id was assigned on is what stops the two drifting. Absent that file the
# parameters are compared as recorded.
settings = {}
group_map_path = os.path.join(os.path.dirname(MAP_PATH), "group_map.json")
if os.path.exists(group_map_path):
    with open(group_map_path, encoding="utf-8") as file:
        settings = {g["group_id"]: tuple(sorted((k, v) for k, v in g.items()
                                                if k not in GROUP_IDENTITY_KEYS))
                    for g in json.load(file)}

FONT_LABEL = fig_cfg["font"]["label"]
FONT_TITLE = fig_cfg["font"]["title"]
DPI        = fig_cfg["dpi"]

MIN_VALID      = 10  # minimum non-NaN values for parameters and results
MIN_VALID_TEMP = 20  # minimum non-NaN values for temperature columns

# ---- LOAD & FLATTEN ----
temp_cols = set()
rows, runs, gkeys = [], [], []
for exp in load_experiments_from_map(MAP_PATH):
    row = {}
    runs.append(exp.get("run"))
    gkeys.append((group_key(exp, [k for k in keys if k != "parameters"]),
                  settings.get(exp["group_id"],
                               tuple(sorted((exp.get("parameters") or {}).items())))
                  if "parameters" in keys else None))
    row.update(exp.get("parameters", {}))
    row.update(exp.get("results", {}))
    mt = exp.get("machine_temperature", {})
    row.update(mt)
    temp_cols.update(mt.keys())
    ext_temp = exp.get("external_temperature_c")
    if ext_temp is not None:
        row["external_temperature_c"] = ext_temp
        temp_cols.add("external_temperature_c")
    rows.append(row)

df_all = pd.DataFrame(rows).select_dtypes(include="number")

# ---- COLLAPSE TO THE CHOSEN GROUPS ----
# Records agreeing on every ticked key are one row, at the mean of what they measured.
# Correlating raw replicates and correlating the settings they average to are different
# questions - the first is dominated by measurement noise where a setting was repeated -
# and the tab's grouping is where that is answered, here as in every other plot. Every key
# ticked leaves one row per setting per run, the group_id the map already carries; a
# selection that repeated no setting is then the untouched frame this always correlated.
#
# By first-appearance order (sort=False) rather than by key, so `runs` below still lines
# up with the rows positionally - a tuple of parameter values has no meaningful sort.
codes = {}
group_of = [codes.setdefault(gkey, len(codes)) for gkey in gkeys]
if len(codes) < len(df_all):
    df_all = df_all.groupby(group_of, sort=False).mean()
    # The run a group came from, taken from its first record. Only asked for below when
    # `run` is ticked, which is the case in which every record in a group shares one.
    first_run = {}
    for code, run in zip(group_of, runs):
        first_run.setdefault(code, run)
    runs = [first_run[code] for code in dict.fromkeys(group_of)]

# One matrix over the whole selection, then one per run. Pooling runs mixes campaigns
# that searched different regions, which can show a correlation neither run has on its
# own - and hide one they share. A single-run selection needs only the first matrix.
#
# Only while `run` is ticked, though: unticking it is the instruction to pool the runs,
# and a row that averages several of them belongs to no single panel to be split into.
run_order = list(dict.fromkeys(runs))
panels = [("All runs", df_all)]
if "run" in keys and len(run_order) > 1:
    panels += [(name or "unknown",
                df_all.iloc[[i for i, r in enumerate(runs) if r == name]])
               for name in run_order]

# ---- PLOT ----
drawn = 0
for title, df in panels:
    valid_counts = df.notna().sum()
    keep = [
        c for c in df.columns
        if valid_counts[c] >= (MIN_VALID_TEMP if c in temp_cols else MIN_VALID)
    ]
    df = df[keep]

    if df.empty or df.shape[1] < 2:
        print(f"Not enough data to compute a correlation matrix for {title}.")
        continue

    corr = df.corr()
    tick_labels = [c.replace("_", " ") for c in df.columns]

    n = len(df.columns)
    size = max(6, n * 0.7)
    fig, ax = plt.subplots(figsize=scale_figsize(size, size * 0.85))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # annotate each cell with its correlation value; flip text to white on the
    # dark (strongly correlated) cells so it stays legible
    annot_size = max(6, FONT_LABEL - 2)
    for i in range(n):
        for j in range(n):
            val = corr.iat[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=annot_size,
            )

    ax.set_xticks(range(n), labels=tick_labels)
    ax.set_yticks(range(n), labels=tick_labels)

    # white grid lines between cells (was linewidths=0.5 in the seaborn call)
    ax.set_xticks([k - 0.5 for k in range(n + 1)], minor=True)
    ax.set_yticks([k - 0.5 for k in range(n + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    ax.tick_params(axis="x", labelsize=FONT_LABEL - 1, rotation=90)
    ax.tick_params(axis="y", labelsize=FONT_LABEL - 1, rotation=0)

    # What a row is cannot be read off the matrix and changes from one drawing to the
    # next, so the figure says it rather than leaving it in the tab.
    ax.set_title(f"{title} - grouped by {', '.join(keys)}", fontsize=FONT_TITLE)
    fig.tight_layout(pad=fig_cfg["layout_pad"])
    drawn += 1

if not drawn:
    sys.exit(0)

plt.show(block=__name__ == "__main__")
