"""How the initial design's size pays off: the gain it buys and what it costs.

Reads the score campaign_gain leaves inside each run directory - one gain.json per run -
for the runs the campaign's map currently holds, which is the selection ticked in the
browser. That is what keeps the plot honest: the numbers follow the selection instead of
whichever set happened to be scored last. A run with no score yet is named and skipped,
and a set of runs scored under different settings is refused rather than mixed.

Plots, against the initial design size n0, the two halves of the trade-off a sweep over
--n-initial exists to measure:

  gamma_c  the gain over the initial design at convergence, in %
  n_c      the evaluations spent reaching it, the design included

Two panels sharing one x axis rather than one panel with two y scales: the measures
have unrelated units, and a second y axis invites reading a crossing point that means
nothing.

Both panels are needed together. gamma_c alone says a bigger design converges higher
without saying what it cost; the cost alone rewards a run that stopped early having
gained little. Read as a pair they are the exploration-cost/convergence-quality
trade-off.

WHICH COST - and a floor to know about

  n_c counts from the run's first evaluation, so it includes the initial design, and the
  convergence window is only looked for past that design (campaign_gain). Together those
  put a floor under it:

      n_c >= n0 + patience

  so a sweep over n0 makes the total-cost panel slope upward whatever the optimizer did -
  part of that rise is arithmetic, not behaviour, and it is steepest exactly where the
  sweep is widest.

  --cost total (the default) keeps n_c, and is the honest number for machine time: an
  initial point costs a measurement like any other. --cost proposals plots n_c - n0
  instead, the evaluations the optimizer itself spent, where the floor cancels - that is
  the one to read for "does a larger design need fewer proposals".

  They can disagree, and the disagreement is the finding: a design that pays for itself
  in proposals can still cost more wall-clock overall.

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
import re
import sys
from pathlib import Path

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
parser.add_argument("--run", action="append", default=[], metavar="DIR",
                    help="A run directory to read a score from. Repeatable. Given none, "
                         "the runs are taken from the campaign's experiment_map.json, "
                         "which is rebuilt from the current selection - so the plot "
                         "follows what is ticked in the browser.")
args = parser.parse_args()

FONT_LABEL = fig_cfg["font"]["label"]
FONT_LEGEND = fig_cfg["font"]["legend"]

def _selected_run_dirs() -> list:
    """The run directories to score, in the order the map lists them.

    The map is rebuilt from the ticked selection before any plot runs, so reading the
    run directories out of it is what makes this plot follow the browser rather than
    whatever set happened to be scored last.
    """
    if args.run:
        return [str(Path(d).resolve()) for d in args.run]
    map_path = os.path.join(data_path, "experiment_map.json")
    if not os.path.exists(map_path):
        print(f"No experiment_map.json in {data_path}, and no --run given, so there is "
              f"nothing to say which runs to read.")
        sys.exit(1)
    exp_map = json.load(open(map_path, encoding="utf-8"))
    seen = []
    for entry in exp_map.get("experiments", []):
        run_dir = entry.get("run_dir")
        if run_dir and run_dir not in seen:
            seen.append(run_dir)
    if not seen:
        print("The map records no run directories - rebuild it, so build_experiment_map "
              "writes run_dir, then score the campaign again.")
        sys.exit(1)
    return seen


# One row per run, read from the score campaign_gain left beside it. The arm label
# already carries the design size (see _labels.arm_label), so the strategy is what is
# left once it is taken out - that, not the arm, is what makes a series here, since n0
# is the x axis.
runs, contexts, missing = [], [], []
for run_dir in _selected_run_dirs():
    score_path = os.path.join(run_dir, "gain.json")
    if not os.path.exists(score_path):
        missing.append(run_dir)
        continue
    score = json.load(open(score_path, encoding="utf-8"))
    n_initial, gamma_c, n_c = (score.get("n_initial"), score.get("gamma_c"),
                               score.get("n_c"))
    if n_initial is None or gamma_c is None or n_c is None:
        missing.append(run_dir)
        continue
    contexts.append(json.dumps(score.get("context"), sort_keys=True))
    # By pattern, not by this run's own number: the size in the arm label is written by
    # _labels.arm_label from the map's count, and the two can differ (a run measured with
    # --repeats records the design more than once). Matching " n<digits>" strips it either
    # way, so the arms stay whole instead of splitting one series per design size.
    label = re.sub(r" n\d+", "", str(score.get("arm") or ""), count=1)
    runs.append({"n0": int(n_initial),
                 "series": label or "run",
                 "gamma_c": float(gamma_c),
                 # Both readings of the same stopping point, drawn one above the other:
                 # what the optimizer spent, and what the experiment cost in total.
                 "cost_prop": float(n_c) - n_initial,
                 "cost_total": float(n_c),
                 "converged": bool(score.get("converged"))})

if missing:
    print(f"! {len(missing)} selected run(s) carry no score yet - score the campaign "
          f"first, and they will be included. Skipped: "
          f"{', '.join(os.path.basename(d) for d in missing[:4])}"
          f"{' ...' if len(missing) > 4 else ''}")

if not runs:
    print("None of the selected runs carries a score with both n_initial and gamma_c. "
          "Score the campaign first; a run recording no initial design cannot be "
          "plotted against its size.")
    sys.exit(1)

# Scores are only comparable when they were measured the same way. The context each file
# carries says how - reference point, objectives, senses, constraints, convergence
# settings - so a set that disagrees is a set that must not share a plot.
if len(set(contexts)) > 1:
    print("! The selected runs were scored under different settings (reference point, "
          "objectives, constraints or convergence), so their numbers are not "
          "comparable. Score them together in one pass, then plot.")
    sys.exit(1)
context = json.loads(contexts[0]) if contexts else {}
if context.get("reference_source") == "selection":
    print("! These scores used a reference point derived from whichever runs were "
          "loaded when they were computed, so they hold only for that selection. "
          "Re-score with --ground-truth for numbers that travel.")

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
fig, (ax_gain, ax_prop, ax_cost) = plt.subplots(
    3, 1, sharex=True, figsize=(figsize[0], figsize[1] * 2.4))

_panel(ax_gain, "gamma_c", converged_only=False)
# Both cost panels keep only the runs that converged: n_c stands in as the budget's end
# for the others, which is a lower bound rather than a measurement.
_panel(ax_prop, "cost_prop", converged_only=True)
_panel(ax_cost, "cost_total", converged_only=True)

# Short labels on purpose: the panels are stacked in a single-column figure, where a
# sentence-long ylabel on each runs into its neighbour. The symbols carry the meaning.
ax_gain.set_ylabel(r"Gain $\gamma_\mathrm{c}$ (%)", fontsize=FONT_LABEL)
ax_prop.set_ylabel(r"$n_\mathrm{c} - n_0$ (proposals)", fontsize=FONT_LABEL)
ax_cost.set_ylabel(r"$n_\mathrm{c}$ (evaluations)", fontsize=FONT_LABEL)
ax_cost.set_xlabel(r"Initial design size $n_0$", fontsize=FONT_LABEL)
ax_cost.set_xticks(sizes)
ax_cost.set_xticklabels([str(s) for s in sizes])
for ax in (ax_gain, ax_prop, ax_cost):
    ax.grid(True, **fig_cfg["grid"])

# The same axis read in machining hours - a unit conversion of the one measure, not a
# second scale competing with it.
if args.hours_per_eval > 0:
    for ax in (ax_prop, ax_cost):
        hours = ax.secondary_yaxis(
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
          f"(patience {context.get('patience')}, tol {context.get('tol')}) - drawn open, "
          f"and left out of the cost boxes. Their n_c is the end of the budget, a lower "
          f"bound.")
for size in sizes:
    kept = [r for r in runs if r["n0"] == size and r["converged"]]
    if not kept:
        print(f"! n0 = {size}: no run converged, so the cost panel has nothing to show "
              f"for it.")

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
