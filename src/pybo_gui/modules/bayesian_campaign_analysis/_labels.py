"""Shared label styling for the campaign plots.

A label decides what gets its own series, colour and Pareto front; the map builder writes
it, choosing between the run, the arm, both, or the provenance (see
build_experiment_map.LABEL_BY). Two things follow that every plot needs to agree on:

* An initial design is a series *within* the chosen dimension, marked by a suffix -
  "bayesian_run3 (initial)". A plot has to recognise that to style it as exploration and
  to pair it with the proposals it belongs to.
* A label the figure style names explicitly keeps its colour; a run directory cannot be
  named in a config in advance, so it takes one from a cycle by position.

Kept beside _constraints.py, the package's other shared private helper, rather than
copied into each script.
"""

# Marks a label as the initial design of whatever it qualifies. Mirrors
# build_experiment_map.INITIAL_SUFFIX, which is what writes it.
INITIAL_SUFFIX = " (initial)"

# Marks an aggregated arm's label with its experiment_type - mirrors arm_label's own
# "{optimizer} ({experiment_type})" format. Stripped for colour the same way
# INITIAL_SUFFIX is: a real rig's bayesian runs and a simulated study's bayesian runs
# share bayesian's colour, not a second one a named palette was never designed to
# hold - told apart instead by line style (see arm_line_style) and the legend text.
PROVENANCE_SUFFIXES = (" (experimental)", " (synthetic)")

QUALIFIER_SUFFIXES = (INITIAL_SUFFIX, *PROVENANCE_SUFFIXES)

# Dash patterns for the two kinds of front, before a per-series phase is applied.
DASHES = {"initial": (1, 3), "proposed": (6, 2)}

# Solid for a real run, dashed for a simulated one - an aggregated arm's own line
# style, keyed by the same qualifier PROVENANCE_SUFFIXES strips for colour.
ARM_LINESTYLES = {"experimental": "-", "synthetic": "--"}

FALLBACK_COLOR = "#888888"


def is_initial(label) -> bool:
    """True for the initial design, however the label is qualified."""
    return bool(label) and (label == "initial" or label.endswith(INITIAL_SUFFIX))


def arm_label(exp: dict, fallback: str) -> str:
    """What --aggregate-runs pools a run's curve into.

    Both axes together, not technology (the optimizer: bayesian/sobol/manual) alone -
    otherwise a real rig's bayesian-proposed runs and a simulated study's bayesian runs
    would average into one curve just because they share an optimizer, which is exactly
    the mix-up experiment_type (experimental/synthetic) exists to keep apart. `fallback`
    is what the caller already falls back to when neither is set - typically the
    per-observation label, since an unlabelled run still deserves a series of its own
    rather than being silently lumped in with everything else unlabelled.
    """
    tech = str(exp.get("technology") or "").strip().lower()
    prov = str(exp.get("provenance") or "").strip().lower()
    if tech and prov:
        return f"{tech} ({prov})"
    return tech or prov or fallback


def arm_line_style(label: str) -> str:
    """An aggregated arm's line style: solid for a real run, dashed for a simulated
    one, matplotlib linestyle notation. Two arms that share a colour because they
    share an optimizer (see base_label) still need to read apart on the curve itself,
    not only in the legend text, when both sit on one plot."""
    if label:
        for suffix in PROVENANCE_SUFFIXES:
            if label.endswith(suffix):
                return ARM_LINESTYLES[suffix.strip(" ()")]
    return "-"


def base_label(label):
    """The label without its initial-design or experiment_type qualifier, whichever it
    carries.

    A design series and the proposals it belongs to share a base, so they share a colour
    and sit together in a legend; marker and dash tell them apart. Likewise a real and a
    simulated run of the same arm: the qualifier arm_label appends is exactly what
    distinguishes them in the legend text, so it must not also fork the colour they
    share.
    """
    if label:
        for suffix in QUALIFIER_SUFFIXES:
            if label.endswith(suffix):
                return label[:-len(suffix)]
    return label


def styler(fig_cfg: dict, labels):
    """Colour, marker and front-style functions bound to the labels present.

    `labels` is every label in the plot, which is what lets a colour be assigned by
    position and a dash phase be spread over the series sharing a pattern - both need to
    know the whole set, not one label at a time.
    """
    named = fig_cfg["colors"]["label"]
    markers = fig_cfg["markers"]["label"]
    cycle = fig_cfg["colors"].get("cycle") or [FALLBACK_COLOR]
    labels = sorted(set(labels))
    unnamed = sorted({base_label(l) for l in labels} - set(named))

    def color(label):
        base = base_label(label)
        if base in named:
            return named[base]
        return cycle[unnamed.index(base) % len(cycle)] if base in unnamed else FALLBACK_COLOR

    def marker(label):
        """The design keeps its own marker, so it reads as exploration wherever it sits."""
        if is_initial(label):
            return markers.get("initial", "^")
        return markers.get(base_label(label), "o")

    def front_style(label):
        """A dash pattern, phased so coincident fronts stay visible.

        Two arms started from one seed share an initial design exactly, so their fronts
        sit on the same points; without a phase apiece the one drawn last hides the other.
        """
        pattern = DASHES["initial"] if is_initial(label) else DASHES["proposed"]
        peers = [l for l in labels if is_initial(l) == is_initial(label)]
        period = sum(pattern)
        phase = period * peers.index(label) / len(peers) if label in peers else 0.0
        return phase, pattern

    return color, marker, front_style