"""What the user chose to group the selected runs by.

The plots used to offer two fixed switches. `--grouped` pooled records sharing
`(run, parameters)` - repeats of one setting inside one run - and `--aggregate-runs` pooled
whole runs by an arm hard-coded as `optimizer + n_initial + provenance`. Between them they
left obvious comparisons unreachable: strategies pooled over design size (a sweep over
--n-initial draws nine series and there was no way to ask for three), and the same setting
measured across several runs, which is exactly what a variability study is for and which
`build_group_map._setting_key` rules out by keying on the run.

So the two switches become a list of keys. Records that agree on every ticked key are one
group, drawn as its mean with an error bar. More keys means a finer split; every key ticked
is the un-pooled view the plots opened with.

    parameters, run, strategy, n_initial, provenance, technology, repeat

    all of them                     one point per observation, one curve per run
    all but repeat                  the old --grouped
    all but run and repeat          the old --aggregate-runs
    all but run, repeat, n_initial  strategies pooled over design size

TWO KINDS OF PLOT

  A Pareto point and a hypervolume trace do not group the same way, because the trace is
  cumulative over a campaign and the point is not.

  * A point plot groups whole records: `group_key`.
  * A trace plot builds one trace per *run* whatever is ticked - the metric only accumulates
    within one campaign - and the keys decide which runs' traces average together:
    `series_key`, which is `group_key` with the two within-step notions dropped. Those two
    still act on a trace plot, by collapsing repeats into one evaluation before the trace is
    built.

WHAT IS NOT A KEY

  `source` - the initial design against the optimizer's proposals. `_labels.base_label`
  keeps both in one trace on purpose, because they are one campaign, and tells them apart
  by marker and dash inside the series. A key would cut a campaign's own curve in half.
"""
from pybo_gui.modules.bayesian_campaign_analysis.build_group_map import _snap

# In the order the GUI shows them: what a measurement was taken at, then where it came
# from, then which of several identical ones it is.
GROUP_KEYS = ("parameters", "run", "strategy", "n_initial",
              "provenance", "technology", "repeat")

# Dropped from a trace plot's series identity. Both say which observation within a step a
# record is, which a curve has already collapsed by the time it is drawn.
_WITHIN_STEP = ("parameters", "repeat")

# How the map spells each key. `parameters` and `repeat` are not plain fields and are read
# by the functions below instead.
_FIELD = {"run": "run", "strategy": "optimizer", "n_initial": "n_initial",
          "provenance": "provenance", "technology": "technology"}

class GroupKeyError(ValueError):
    """An unusable --group-by. Its own type so a caller can exit 2 rather than traceback."""


def parse_keys(values) -> tuple:
    """The keys named on the command line, in GROUP_KEYS order.

    Order is normalised rather than taken as given: the key set is what decides grouping,
    so two invocations naming the same keys differently have to produce the same series -
    including the same series *labels*, which are built from this order.
    """
    unknown = [v for v in values if v not in GROUP_KEYS]
    if unknown:
        raise GroupKeyError(f"Unknown --group-by {unknown}. "
                        f"Available: {', '.join(GROUP_KEYS)}")
    chosen = set(values)
    return tuple(k for k in GROUP_KEYS if k in chosen)


def measurement_id(exp: dict) -> tuple:
    """What tells one individual measurement from another.

    The `repeat` key means "keep repeated measurements apart", and the thing that has to
    stay apart is the measurement, not its index within a step. Keying on the index alone
    was wrong: two evaluations in *different* steps that happen to land on the same setting
    both sit at rep00, so they merged - and a converged run re-proposing its own optimum
    does exactly that, quietly shortening its budget by an evaluation.

    So it is the step and the position within it. Nothing is added to the map for this:
    `iteration` is the step directory's name and `observation` its index in that step's
    batch, both already recorded.
    """
    return (exp.get("iteration"), exp.get("observation"))


def group_key(exp: dict, keys, resolutions: dict | None = None) -> tuple:
    """Identity of the group a record belongs to, for a plot that groups whole records.

    `resolutions` maps a parameter label to the rig's step for it, so two records land
    together exactly when the rig could not have told their settings apart - the same rule
    `build_group_map` already applies, reused rather than restated so the two cannot drift.
    """
    resolutions = resolutions or {}
    parts = []
    for key in keys:
        if key == "parameters":
            parameters = exp.get("parameters") or {}
            parts.append(("parameters", tuple(
                (label, _snap(parameters[label], resolutions.get(label)))
                for label in sorted(parameters))))
        elif key == "repeat":
            parts.append(("repeat", measurement_id(exp)))
        else:
            parts.append((key, exp.get(_FIELD[key])))
    return tuple(parts)


def series_key(exp: dict, keys) -> tuple:
    """Identity of the series a record's *run* belongs to, for a plot that draws traces."""
    return group_key(exp, [k for k in keys if k not in _WITHIN_STEP])


def series_label(exp: dict, keys, fallback: str = "") -> str:
    """What to call a series, built from the keys that still tell it apart.

    Only the keys that survive into the identity are named, so a plot pooling over design
    size says "bayesian" rather than "bayesian n10" - the label has to describe the group
    it is on, or two merged series would carry one of their members' names.

    `fallback` is what an otherwise nameless series gets, typically the record's own label:
    a run that names none of these still deserves a series of its own rather than being
    silently lumped in with everything else unnamed.
    """
    # A run names itself. Everything else a record could be keyed on is constant within
    # one run - its strategy, its design size, where it came from - so appending them to
    # the run's own name only makes the legend longer.
    if "run" in keys and exp.get("run"):
        return str(exp["run"])

    parts = []
    for key in keys:
        if key in _WITHIN_STEP or key == "run":
            continue
        value = exp.get(_FIELD[key])
        if value in (None, ""):
            continue
        parts.append(f"n{value}" if key == "n_initial" else str(value).strip().lower())
    return " ".join(parts) or fallback


def pooled(keys) -> tuple:
    """The dimensions the ticked keys collapse, for the band's legend.

    A band means different things depending on what went into it - `repeat` alone is
    measurement noise, `run` alone is how differently the optimizer behaves from seed to
    seed, and both together is a mixture of the two. None of that is recoverable from the
    picture, so the legend has to carry it.
    """
    return tuple(k for k in GROUP_KEYS if k not in set(keys))
