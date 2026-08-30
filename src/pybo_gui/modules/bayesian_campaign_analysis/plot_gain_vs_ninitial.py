"""How each group's gain compares, and what it cost.

Reads the score campaign_gain leaves for each run in its own workspace cache entry (see
build_experiment_map.run_gain_path) - one gain.json per run, fingerprinted from that run
alone rather than from any selection - for the runs the campaign's map currently holds,
which is the selection ticked in the browser. That is what keeps the plot honest: the
numbers follow the selection instead of whichever set happened to be scored last, while a
run scored once stays readable from any later selection that includes it. A run with no
score yet is named and skipped, and a set of runs scored under different settings is
refused rather than mixed.

Plots, one box per group, the two halves of the trade-off a comparison across groups
exists to measure:

  gamma    the gain over the initial design at convergence, in %
  n_c      the evaluations spent reaching it, the design included

A group is a run's arm - strategy, initial design size and provenance together (see
_labels.arm_label), the same identity plot_hypervolume's --aggregate-runs pools a curve
by. That is deliberately broader than a sweep over --n-initial alone: two runs of the
same strategy at different design sizes are two different groups here just as they would
be under a sweep ("bayesian n10" and "bayesian n20"), but so are a real campaign and a
simulated benchmark run under the same strategy ("bayesian n20 (experimental)" against
"bayesian n20 (synthetic)") - a comparison a sweep over n_initial alone has no way to draw,
because nothing there is being varied.

Two panels sharing one x axis rather than one panel with two y scales: the measures
have unrelated units, and a second y axis invites reading a crossing point that means
nothing.

Both panels are needed together. gamma alone says a group converges higher without
saying what it cost; the cost alone rewards a run that stopped early having gained
little. Read as a pair they are the exploration-cost/convergence-quality trade-off.

WHICH COST - and a floor to know about

  n_c counts from the run's first evaluation, so it includes the initial design, and the
  convergence window is only looked for past that design (campaign_gain). Together those
  put a floor under it:

      n_c >= n0 + patience

  so two groups at different design sizes are not on equal footing in the total-cost
  panel for arithmetic reasons alone, before behaviour enters into it.

  --cost total (the default) keeps n_c, and is the honest number for machine time: an
  initial point costs a measurement like any other. --cost proposals plots n_c - n0
  instead, the evaluations the optimizer itself spent, where the floor cancels - that is
  the one to read for "does a larger design need fewer proposals".

  They can disagree, and the disagreement is the finding: a design that pays for itself
  in proposals can still cost more wall-clock overall.

A run still improving when its budget ended never converged, so it has no n_c and no
gain measured at one. It is not dropped: gamma_budget - what it had gained by the end -
stands in on the gain panel, drawn as an open marker to say it is a lower bound rather
than a measurement. The cost panels leave it out entirely, since there the missing
quantity is the x value itself. The console says how many there were.

Each group gets its own colour and marker, so the comparison the figure exists to make
survives greyscale printing and colour vision deficiency - and so its boxes still read
apart from their neighbours' even where the x-axis labels below them run long enough to
overlap.

EXAMPLE

  PYBO_CAMPAIGN_DIR=/path/to/campaign python -m \
      pybo_gui.modules.bayesian_campaign_analysis.plot_gain_vs_ninitial --hours-per-eval 1.5
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from pybo_gui.configs.figure_settings.config import fig_cfg
from pybo_gui.configs.settings import data_path
from pybo_gui.modules.bayesian_campaign_analysis._labels import styler
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import run_gain_path

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


# One row per run, read from the score campaign_gain cached for it. The arm label is the
# group itself - strategy, design size and provenance together (see _labels.arm_label) -
# so unlike the old sweep-only reading, it is kept whole rather than having its design
# size stripped back out: two groups that only differ by n_initial still belong on
# different x positions, the same way two that only differ by provenance do.
runs, contexts, missing, stale, unconverged = [], [], [], [], []
for run_dir in _selected_run_dirs():
    score_path = run_gain_path(run_dir)
    if not score_path.exists():
        missing.append(run_dir)
        continue
    score = json.load(open(score_path, encoding="utf-8"))
    # The gain at convergence, which campaign_gain wrote as "gamma_c" until "gamma" was
    # freed up for it - the paper's gamma is measured at convergence, and the column that
    # held that name is now "gamma_budget". A file written before the swap is named rather
    # than half-read: its "gamma" is the end-of-budget number, so reading it under the new
    # name would silently plot a different quantity for those runs than for the rest.
    n_initial, n_c = score.get("n_initial"), score.get("n_c")
    gamma_c = score.get("gamma") if "gamma_budget" in score else None
    if gamma_c is None and "gamma_c" in score:
        stale.append(run_dir)
        continue
    arm = score.get("arm")
    if arm is None:
        missing.append(run_dir)
        continue
    # A run still improving when its budget ran out has no n_c, and campaign_gain leaves
    # gamma and eta null for it rather than passing the budget's end off as a convergence
    # point. It is not dropped here: what it gained by the end is a real measurement and a
    # lower bound on what it would have reached, so gamma_budget stands in and the marker
    # is drawn open - the same censoring convention this plot already used. The cost panels
    # do drop it, because there its missing quantity is the x value itself.
    converged = bool(score.get("converged")) and n_c is not None
    gain = gamma_c if converged else score.get("gamma_budget")
    if gain is None:
        unconverged.append(run_dir)
        continue
    contexts.append(json.dumps(score.get("context"), sort_keys=True))
    runs.append({"arm": str(arm),
                 # For ordering the x axis only - two groups that share a design size
                 # still sort together rather than by name alone. None sorts last.
                 "n0": n_initial,
                 "gamma": float(gain),
                 # Both readings of the same stopping point, drawn one above the other:
                 # what the optimizer spent, and what the experiment cost in total. NaN
                 # without an n_c, which the converged-only cost panels then leave out.
                 "cost_prop": float(n_c) - n_initial if converged and n_initial is not None
                             else float("nan"),
                 "cost_total": float(n_c) if converged else float("nan"),
                 "converged": converged})

if missing:
    print(f"! {len(missing)} selected run(s) carry no score yet - score the campaign "
          f"first, and they will be included. Skipped: "
          f"{', '.join(os.path.basename(d) for d in missing[:4])}"
          f"{' ...' if len(missing) > 4 else ''}")

if unconverged:
    print(f"! {len(unconverged)} selected run(s) never converged - they were still "
          f"improving when the budget ended, so they have no n_c or gamma to plot. Give "
          f"them a longer budget, or loosen --tol-rel when scoring. Skipped: "
          f"{', '.join(os.path.basename(d) for d in unconverged[:4])}"
          f"{' ...' if len(unconverged) > 4 else ''}")

if stale:
    print(f"! {len(stale)} selected run(s) were scored under the older column names, "
          f"where the gain at convergence was called gamma_c. Re-score the campaign and "
          f"they will be included. Skipped: "
          f"{', '.join(os.path.basename(d) for d in stale[:4])}"
          f"{' ...' if len(stale) > 4 else ''}")

if not runs:
    print("None of the selected runs carries a current score with a gain. Score the "
          "campaign first.")
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

# Ordered by design size where a group carries one, not alphabetically - a sweep over
# --n-initial should read n5, n10, n20, not the "n10" < "n20" < "n5" a string sort gives.
# A group missing n_initial sorts after those that have it, then by name.
arm_n0 = {}
for r in runs:
    arm_n0.setdefault(r["arm"], r["n0"])
categories = sorted(arm_n0, key=lambda a: (arm_n0[a] is None, arm_n0[a], a))
if len(categories) < 2:
    print(f"! only one group ({categories[0]}) in this selection - there is no "
          f"comparison to draw. Select runs from more than one arm.")

# Colour and marker together: the groups are the thing being compared here, and a figure
# that separates them by hue alone stops separating them at all in greyscale, in print, or
# for a reader with a colour vision deficiency - useful even though the x-axis labels
# already name each group, since it is what lets a reader match a stray point back to its
# box without reading sideways.
color, marker, _line, _front = styler(fig_cfg, categories)


def _cell(name, key, converged_only=False):
    return [r[key] for r in runs
            if r["arm"] == name and (r["converged"] or not converged_only)]


def _panel(ax, key, converged_only):
    """One measure across every group, as a box per category plus its points."""
    for i, name in enumerate(categories):
        values = _cell(name, key, converged_only)
        if values:
            box = ax.boxplot([values], positions=[i], widths=0.6,
                             patch_artist=True, manage_ticks=False, showfliers=False)
            for patch in box["boxes"]:
                patch.set(facecolor=color(name), alpha=0.25, linewidth=0.8,
                          edgecolor=color(name))
            for part in ("whiskers", "caps", "medians"):
                for line in box[part]:
                    line.set(color=color(name), linewidth=1.0)
        if not args.points:
            continue
        for run in (r for r in runs if r["arm"] == name):
            if converged_only and not run["converged"]:
                continue
            # A deterministic jitter, so the same campaign redraws identically.
            jitter = (hash((run["arm"], run[key])) % 100 / 100 - 0.5)
            ax.plot(i + jitter * 0.3, run[key],
                    marker=marker(name), markersize=4.5, linestyle="none",
                    color=color(name),
                    # Open for a run that never converged: what is plotted for it is
                    # gamma_budget, a lower bound, rather than a gain at a convergence
                    # point it never reached.
                    markerfacecolor=color(name) if run["converged"] else "none",
                    markeredgecolor=color(name), markeredgewidth=0.8,
                    alpha=0.75, zorder=3)


figsize = fig_cfg["figsize"].get("gain_vs_ninitial",
                                 list(fig_cfg["figsize"]["hypervolume"]))
fig, (ax_gain, ax_prop, ax_cost) = plt.subplots(
    3, 1, sharex=True, figsize=(figsize[0], figsize[1] * 2.4))

_panel(ax_gain, "gamma", converged_only=False)
# Both cost panels keep only the runs that converged: n_c stands in as the budget's end
# for the others, which is a lower bound rather than a measurement.
_panel(ax_prop, "cost_prop", converged_only=True)
_panel(ax_cost, "cost_total", converged_only=True)

# Short labels on purpose: the panels are stacked in a single-column figure, where a
# sentence-long ylabel on each runs into its neighbour. The symbols carry the meaning.
ax_gain.set_ylabel(r"$\gamma$", fontsize=FONT_LABEL)
ax_prop.set_ylabel(r"$n_\mathrm{c} - n_0$", fontsize=FONT_LABEL)
ax_cost.set_ylabel(r"$n_\mathrm{c}$", fontsize=FONT_LABEL)
ax_cost.set_xticks(range(len(categories)))
# Rotated: a group's name carries its strategy, design size and provenance together
# (see _labels.arm_label), which routinely runs longer than a bare "n20" ever did -
# horizontal labels would overlap their neighbours long before the figure got crowded.
ax_cost.set_xticklabels([name.capitalize() for name in categories],
                        rotation=30, ha="right", fontsize=FONT_LABEL)
for ax in (ax_gain, ax_prop, ax_cost):
    ax.grid(True, **fig_cfg["grid"])
    ax.set_xlim(-0.5, len(categories) - 0.5)

# The same axis read in machining hours - a unit conversion of the one measure, not a
# second scale competing with it.
if args.hours_per_eval > 0:
    for ax in (ax_prop, ax_cost):
        hours = ax.secondary_yaxis(
            "right", functions=(lambda v: v * args.hours_per_eval,
                                lambda v: v / args.hours_per_eval))
        hours.set_ylabel("Machining time (h)", fontsize=FONT_LABEL)

censored = [r for r in runs if not r["converged"]]
if censored:
    print(f"! {len(censored)} of {len(runs)} runs were still improving when the budget "
          f"ended (patience {context.get('patience')}, tol_rel {context.get('tol_rel')}) "
          f"- drawn open, with gamma_budget standing in for a gain at a convergence point "
          f"they never reached, and left out of the cost panels entirely since they have "
          f"no n_c. Both are lower bounds: give them a longer budget to measure rather "
          f"than bound them.")
for name in categories:
    if not any(r["arm"] == name and r["converged"] for r in runs):
        print(f"! {name}: no run converged, so the cost panel has nothing to show for it.")

fig.tight_layout(pad=fig_cfg["layout_pad"])
plt.show(block=__name__ == "__main__")
