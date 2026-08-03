import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg, scale_figsize
from pybo_gui.configs.settings import data_path
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map

MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

FONT_LABEL = fig_cfg["font"]["label"]
FONT_TITLE = fig_cfg["font"]["title"]
DPI        = fig_cfg["dpi"]

MIN_VALID      = 10  # minimum non-NaN values for parameters and results
MIN_VALID_TEMP = 20  # minimum non-NaN values for temperature columns

# ---- LOAD & FLATTEN ----
temp_cols = set()
rows = []
for exp in load_experiments_from_map(MAP_PATH):
    row = {}
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

df = pd.DataFrame(rows).select_dtypes(include="number")
valid_counts = df.notna().sum()
keep = [
    c for c in df.columns
    if valid_counts[c] >= (MIN_VALID_TEMP if c in temp_cols else MIN_VALID)
]
df = df[keep]

if df.empty or df.shape[1] < 2:
    print("Not enough data to compute a correlation matrix.")
    sys.exit(0)

corr = df.corr()
tick_labels = [c.replace("_", " ") for c in df.columns]

# ---- PLOT ----
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

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
