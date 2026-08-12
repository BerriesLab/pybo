"""Collapsing the runs of one arm into a single curve with a band.

A study runs the same arm many times over different seeds, so a plot of it holds one
curve per run. Drawn individually they answer "what did each run do"; collapsed they
answer "what does this arm do, and by how much does it vary" - which is the question a
bo-versus-sobol comparison actually asks. Both views are wanted, so this is an option
rather than a replacement.

This is a different axis from the *grouped* option elsewhere in these plots. That one
averages the repeats of one setting *within* a step, where the spread is measurement
noise (see _uncertainty). This one averages *across runs* at the same step, where the
spread is how differently the optimizer behaves from one seed to the next. The two
compose: a grouped, aggregated plot averages repeats into a point and then runs into a
curve.

Runs are aligned by step index and truncated to the shortest. Averaging instead over
however many runs reached each step lets n fall away toward the end, and the band then
widens because the sample shrank rather than because the arm did anything - an artefact
that reads exactly like a real loss of consistency.
"""
import numpy as np
from scipy import stats

# What the band around the mean is taken to mean. "ci95" answers "where does the arm's
# true mean lie", which is the one that says whether two arms are distinguishable;
# "std" answers "where does a single run land", which is wider and does not shrink with
# more replicates. They are routinely confused, so both are offered by name.
BAND_MODES = ("ci95", "sem", "std", "minmax")


def align(curves):
    """Every curve truncated to the shortest, as a 2-D array of shape (runs, steps).

    Raises ValueError on an empty set, so a caller cannot silently plot nothing.
    """
    curves = [list(c) for c in curves if len(c)]
    if not curves:
        raise ValueError("no runs to aggregate")
    n = min(len(c) for c in curves)
    return np.asarray([c[:n] for c in curves], dtype=float)


def mean_band(curves, mode: str = "ci95"):
    """(mean, low, high) down the runs axis, one value per step.

    A single run has no spread to report, so the band collapses onto the mean rather
    than being invented: one curve is still worth drawing, it just carries no interval.
    """
    if mode not in BAND_MODES:
        raise ValueError(f"band must be one of {', '.join(BAND_MODES)}, got {mode!r}")
    y = align(curves)
    mean = y.mean(axis=0)
    n = y.shape[0]
    if n < 2:
        return mean, mean.copy(), mean.copy()

    if mode == "minmax":
        return mean, y.min(axis=0), y.max(axis=0)

    # ddof=1: the runs are a sample of the seeds the arm could have been given, not the
    # whole of them, so the spread has to be the sample's.
    sd = y.std(axis=0, ddof=1)
    if mode == "std":
        half = sd
    elif mode == "sem":
        half = sd / np.sqrt(n)
    else:
        # Student's t, not 1.96: at the ten or so replicates a study affords, the normal
        # quantile understates the interval by around 15%.
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return mean, mean - half, mean + half


def band_label(mode: str, n: int) -> str:
    """How the band should be named in a legend, including how many runs it came from."""
    return {"ci95": f"mean, 95% CI (n={n})",
            "sem": f"mean $\\pm$ SEM (n={n})",
            "std": f"mean $\\pm$ SD (n={n})",
            "minmax": f"mean, min-max (n={n})"}[mode]


def attainment_grid(fronts, n_points: int = 200):
    """A shared x grid over the range every front covers, for averaging 2-D fronts.

    `fronts` are (x, y) sequences, each already sorted by x. A Pareto front is a curve in
    objective space rather than a series over steps, so runs cannot be averaged position
    by position: their points sit at x values none of the others visited. They are read
    onto a common grid instead.

    The grid spans the *overlap* of the runs, not their union: outside it at least one
    run has no front, and averaging there would either extrapolate it or quietly drop it
    and change what n means partway along the curve.
    """
    spans = [(min(x), max(x)) for x, _ in fronts if len(x)]
    if not spans:
        raise ValueError("no fronts to aggregate")
    lo = max(s[0] for s in spans)
    hi = min(s[1] for s in spans)
    if not hi > lo:
        raise ValueError("the fronts share no common range to average over")
    return np.linspace(lo, hi, n_points)


def step_interpolate(x, y, grid):
    """`y` read onto `grid` as a staircase: at each grid point, the y of the last front
    point at or before it.

    Deliberately not linear. A front built from finitely many evaluations is an
    attainment staircase - between two of its points the run achieved the earlier one and
    nothing better. Interpolating linearly would draw a front through points no run ever
    reached, and would flatter every arm equally without saying so.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.searchsorted(x, np.asarray(grid, dtype=float), side="right") - 1
    return y[np.clip(idx, 0, len(y) - 1)]
