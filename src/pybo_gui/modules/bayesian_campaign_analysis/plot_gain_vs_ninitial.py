"""How the initial design's size pays off: the gain it buys and what it costs.

Reads the gain.json that campaign_gain writes and plots, against the initial design
size n0, the two halves of the trade-off a sweep over --n-initial exists to measure:

  gamma_c  the gain over the initial design at convergence, in %
  n_c      the evaluations spent reaching it, the design included

Two panels sharing one x axis rather than one panel with two y scales: the measures
have unrelated units, and a second y axis invites reading a crossing point that means
nothing.

Both panels are needed together. gamma_c alone says a bigger design converges higher
without saying what it cost; n_c alone rewards a run that stopped early having gained
little. Read as a pair they are the exploration-cost/convergence-quality trade-off.

A run that never met the convergence test has no n_c - the budget's end stands in, which
is a lower bound, not a measurement. Those are drawn as open markers and left out of the
cost boxes, and the console says how many there were.

EXAMPLE

  PYBO_CAMPAIGN_DIR=/path/to/campaign python -m \
      pybo_gui.modules.bayesian_campaign_analysis.plot_gain_vs_ninitial --hours-per-eval 1.5
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.modules.bayesian_campaign_analysis._labels import styler

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--hours-per-eval", type=float, default=0.0,
                    help="Machining hours one evaluation costs. Adds a second scale to "
                         "the cost panel reading the same axis in hours (0 = off).")
parser.add_argument("--points", type=lambda v: v.lower() not in ("0", "false", "no"),
                    default=True,
                    help="Draw each replicate as a point beside its box (default: on). "
                         "With a handful of replicates the box alone hides how few.")
parser.add_argument("--gain-dir", default=None,
                    help="Where to read gain.json from (default: the campaign "
                         "directory). Must match campaign_gain's --out-dir.")
args = parser.parse_args()

FONT_LABEL = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]

GAIN_PATH = os.path.join(args.gain_dir or data_path, "gain.json")
if not os.path.exists(GAIN_PATH):
    print(f"No gain.json at {GAIN_PATH}. Run campaign_gain first - this plots what it "
          f"writes - and point both at the same directory.")
    sys.exit(1)
report = json.load(open(GAIN_PATH, encoding="utf-8"))

# One row per run, from every arm's runs_detail. The arm label already carries the design
# size (see _labels.arm_label), so the strategy is what is left once it is taken out -
# that, not the arm, is what makes a series here, since n0 is the x axis.
runs = []
for arm in report.get("arms", []):
    for run in arm.get("runs_detail", []):
        n_initial = run.get("n_initial")
        if n_initial is None or run.get("gamma_c") is None:
            continue
        label = str(run.get("arm") or "")
        runs.append({"n0": int(n_initial),
                     "series": label.replace(f" n{n_initial}", "", 1) or "run",
                     "gamma_c": float(run["gamma_c"]),
                     "n_c": float(run["n_c"]),
                     "converged": bool(run.get("converged"))})

if not runs:
    print("gain.json has no runs carrying both n_initial and gamma_c. A campaign whose "
          "runs record no initial design cannot be plotted against its size.")
    sys.exit(1)

sizes = sorted({r["n0"] for r in runs})
series = sorted({r["series"] for r in runs})
if len(sizes) < 2:
    print(f"! only one initial design size ({sizes[0]}) in this campaign - there is no "
          f"trend to draw. Select runs from a sweep over --n-initial.")

color, _marker, _front = styler(fig_cfg, series)

# Series are dodged around each size so their boxes never overlap. One series sits on the
# size itself; several share the spacing between adjacent sizes.
step = min(np.diff(sizes)) if len(sizes) > 1 else 1.0
width = 0.5 * step / max(len(series), 1)


def _cell(name, size, key, converged_only=False):
    return [r[key] for r in runs
            if r["series"] == name and r["n0"] == size
            and (r["converged"] or not converged_only)]


def _panel(ax, key, converged_only):
    """One measure across every size and series, as a box per cell plus its points."""
    for i, name in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        positions, cells = [], []
        for size in sizes:
            values = _cell(name, size, key, converged_only)
            if values:
                positions.append(size + offset)
                cells.append(values)
        if cells:
            box = ax.boxplot(cells, positions=positions, widths=width * 0.8,
                             patch_artist=True, manage_ticks=False, showfliers=False)
            for patch in box["boxes"]:
                patch.set(facecolor=color(name), alpha=0.25, linewidth=0.8,
                          edgecolor=color(name))
            for part in ("whiskers", "caps", "medians"):
                for line in box[part]:
                    line.set(color=color(name), linewidth=1.0)
        if not args.points:
            continue
        for size in sizes:
            for run in (r for r in runs if r["series"] == name and r["n0"] == size):
                if converged_only and not run["converged"]:
                    continue
                # A deterministic jitter, so the same campaign redraws identically.
                jitter = (hash((run["series"], size, run[key])) % 100 / 100 - 0.5)
                ax.plot(size + offset + jitter * width * 0.5, run[key],
                        marker="o", markersize=3.5, linestyle="none",
                        color=color(name),
                        # Open for a run that never converged: its n_c is the budget's
                        # end standing in for a measurement it never made.
                        markerfacecolor=color(name) if run["converged"] else "none",
                        markeredgecolor=color(name), markeredgewidth=0.8,
                        alpha=0.75, zorder=3)


figsize = fig_cfg["figsize"].get("gain_vs_ninitial",
                                 list(fig_cfg["figsize"]["hypervolume"]))
fig, (ax_gain, ax_cost) = plt.subplots(
    2, 1, sharex=True, figsize=(figsize[0], figsize[1] * 1.7))

_panel(ax_gain, "gamma_c", converged_only=False)
_panel(ax_cost, "n_c", converged_only=True)

ax_gain.set_ylabel(r"Gain at convergence $\gamma_\mathrm{c}$ (%)", fontsize=FONT_LABEL)
ax_cost.set_ylabel(r"Evaluations to convergence $n_\mathrm{c}$", fontsize=FONT_LABEL)
ax_cost.set_xlabel(r"Initial design size $n_0$", fontsize=FONT_LABEL)
ax_cost.set_xticks(sizes)
ax_cost.set_xticklabels([str(s) for s in sizes])
for ax in (ax_gain, ax_cost):
    ax.grid(True, **fig_cfg["grid"])

# The same axis read in machining hours - a unit conversion of the one measure, not a
# second scale competing with it.
if args.hours_per_eval > 0:
    hours = ax_cost.secondary_yaxis(
        "right", functions=(lambda v: v * args.hours_per_eval,
                            lambda v: v / args.hours_per_eval))
    hours.set_ylabel("Machining time (h)", fontsize=FONT_LABEL)

# One series names itself in the axis labels; several need telling apart.
if len(series) > 1:
    handles = [mlines.Line2D([], [], color=color(name), marker="o", linestyle="none",
                             markersize=5, label=name) for name in series]
    leg_cfg = fig_cfg["legend"]
    ax_gain.legend(handles=handles, fontsize=FONT_LEGEND, loc=leg_cfg["loc"],
                   frameon=leg_cfg["frameon"], framealpha=leg_cfg["framealpha"])

censored = [r for r in runs if not r["converged"]]
if censored:
    print(f"! {len(censored)} of {len(runs)} runs never met the convergence test "
          f"({report.get('convergence')}) - drawn open, and left out of the cost boxes. "
          f"Their n_c is the end of the budget, a lower bound.")
for size in sizes:
    kept = [r for r in runs if r["n0"] == size and r["converged"]]
    if not kept:
        print(f"! n0 = {size}: no run converged, so the cost panel has nothing to show "
              f"for it.")

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
