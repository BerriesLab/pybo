"""The reference-run overlay shared by the two Pareto plots.

A run (or several, pooled) the user has flagged as their benchmark is drawn apart
from the ordinary per-label series it is being compared against: always in one
fixed colour and a square marker regardless of which arm produced it, and - where
more than one run makes up the reference - as a mean front read onto a shared grid
the same way --aggregate-runs already pools an arm's runs, so "the benchmark" reads
as one line rather than a run per repeat.

A single reference point (no breadth to grid over) still has to draw something: the
square marker is unconditional, and the mean/band line is only added on top when
there is a real front to average.
"""
import matplotlib.lines as mlines

from pybo_gui.modules.bayesian_campaign_analysis._aggregate import (
    mean_band, arm_legend_label, attainment_grid, step_interpolate)

REFERENCE_MARKER = "s"


def pareto_front(points, sx=1.0, sy=1.0):
    """Non-dominated (x, y) points under the given per-axis senses (sx/sy are +1 to
    minimize, -1 to maximize). A local copy rather than a shared import: this module
    is meant to be droppable into either Pareto plot without pulling in the other's
    own helpers."""
    pts = sorted(points, key=lambda p: (sx * p[0], sy * p[1]))
    front, best_y = [], float("inf")
    for x, y in pts:
        ty = sy * y
        if ty < best_y:
            front.append((x, y))
            best_y = ty
    return front


def _front_rows(rows, sx, sy):
    """Same rule as pareto_front, but keeping the row each corner came from - the
    coordinate pair alone drops the error-bar fields grouped mode computed for it.
    """
    ordered = sorted(rows, key=lambda r: (sx * r["x"], sy * r["y"]))
    front, best_y = [], float("inf")
    for r in ordered:
        ty = sy * r["y"]
        if ty < best_y:
            front.append(r)
            best_y = ty
    return front


def draw_reference(ax, reference_rows, sx, sy, band, color, marker_size, linewidth,
                   legend_handles) -> None:
    """Draw the reference overlay onto `ax` and append its legend handle.

    `reference_rows` are feasible rows already filtered to the ones flagged
    reference, each carrying real "x", "y" and the "run" (or "label", failing that)
    they came from. Does nothing when there are none.

    `linewidth` sets both the marker's stroke and the band's step line - the
    benchmark reads as one weight throughout, matched to the Pareto front's own
    (fig_cfg["pareto_front"]) rather than the thinner weight ordinary points draw
    with, so it stays the heaviest thing on the axes next to the front itself.
    """
    if not reference_rows:
        return
    by_run = {}
    for r in reference_rows:
        by_run.setdefault(r.get("run") or r["label"], []).append(
            (sx * r["x"], sy * r["y"]))
    fronts = []
    for pts in by_run.values():
        f = sorted(pareto_front(pts), key=lambda p: p[0])
        if f:
            fronts.append(([p[0] for p in f], [p[1] for p in f]))
    if not fronts:
        return

    # The real, observed front corners - marked unconditionally, whether or not
    # there is enough breadth below to also grid and average. Each keeps its own
    # error bar where grouped mode gave it one, the same as an ordinary series'
    # points do - a benchmark's repeat spread is exactly as real as anyone else's.
    for r in _front_rows(reference_rows, sx, sy):
        xerr = [[r["x_err_lo"]], [r["x_err_hi"]]] if r.get("x_err_hi") else None
        yerr = [[r["y_err_lo"]], [r["y_err_hi"]]] if r.get("y_err_hi") else None
        if xerr or yerr:
            # No marker drawn here (fmt="none") - errorbar() has no way to keep the
            # caps thin once markeredgewidth is set for a bold marker: capthick is
            # meant to override it for the caps, but is itself silently overridden
            # right back whenever markeredgewidth is also given. Drawing the square
            # as its own scatter() below sidesteps that rather than fighting it.
            ax.errorbar([r["x"]], [r["y"]], xerr=xerr, yerr=yerr, fmt="none",
                       ecolor=color, elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)
        ax.scatter([r["x"]], [r["y"]], s=marker_size,
                  marker=REFERENCE_MARKER, facecolors="none", edgecolors=color,
                  linewidths=linewidth, zorder=5)

    drew_band = False
    try:
        grid = attainment_grid(fronts)
    except ValueError:
        # A single point, or every run's front collapsed onto one x - nothing to
        # average across. The corners above already said what there is to say.
        grid = None
    if grid is not None:
        curves = [step_interpolate(fx, fy, grid) for fx, fy in fronts]
        mean, low, high = mean_band(curves, band)
        gx = sx * grid
        if len(fronts) > 1:
            ax.fill_between(gx, sy * low, sy * high, color=color, alpha=0.18,
                            linewidth=0, zorder=2)
        ax.step(gx, sy * mean, where="post" if sx > 0 else "pre",
                color=color, linewidth=linewidth, zorder=3)
        drew_band = True

    legend_handles.append(mlines.Line2D(
        [], [], color=color if drew_band else "none",
        linewidth=linewidth if drew_band else 0.0,
        marker=REFERENCE_MARKER, markerfacecolor="none", markeredgecolor=color,
        markeredgewidth=linewidth, markersize=marker_size ** 0.5,
        label=arm_legend_label("Reference", band, len(fronts))))
