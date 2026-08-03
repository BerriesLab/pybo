import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.utils.experiment_map_loader import load_experiments_from_map

MAP_PATH   = os.path.join(data_path, "experiment_map.json")
OUTPUT_DIR = data_path

FONT_LABEL  = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]
FONT_TITLE  = fig_cfg["font"]["title"]
DPI         = fig_cfg["dpi"]
SCATTER     = fig_cfg["scatter"]

# ---- LOAD ----
all_keys = []
rows = []
for exp in load_experiments_from_map(MAP_PATH):
    ts = exp.get("start_time")
    if ts is None:
        continue
    dt      = datetime.fromtimestamp(ts)
    results = exp.get("results", {})
    for k in results:
        if k not in all_keys:
            all_keys.append(k)
    rows.append({
        "time_of_day": dt.hour + dt.minute / 60 + dt.second / 3600,
        "results":     results,
    })

# ---- ONE FIGURE PER RESULT KEY ----
for key in all_keys:
    fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["results_vs_timeofday"])

    subset = [r for r in rows if isinstance(r["results"].get(key), (int, float))]
    ax.scatter(
        [r["time_of_day"]  for r in subset],
        [r["results"][key] for r in subset],
        color="steelblue", s=SCATTER["marker_size"], zorder=3,
    )

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda h, _: f"{int(h):02d}:{int((h % 1) * 60):02d}"
    ))
    ax.set_xlim(0, 24)

    label = key.replace("_", " ")
    ax.set_xlabel("Time of day", fontsize=FONT_LABEL)
    ax.set_ylabel(label,         fontsize=FONT_LABEL)

    ax.tick_params(labelsize=FONT_LABEL - 1)
    ax.grid(True, **fig_cfg["grid"])

    fig.tight_layout(pad=fig_cfg["layout_pad"])

plt.show(block=__name__ == "__main__")
