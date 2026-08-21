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
# hold - told apart instead by their own line style and marker (see styler) and by
# the legend text.
PROVENANCE_SUFFIXES = (" (experimental)", " (synthetic)")

QUALIFIER_SUFFIXES = (INITIAL_SUFFIX, *PROVENANCE_SUFFIXES)

# Line styles and markers a series takes when nothing names one for it, assigned by
# position the way colours already are. Ordered most to least distinct, so a plot with
# few series uses only the patterns that read apart at a glance.
LINE_STYLE_CYCLE = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1, 1, 1)))
MARKER_CYCLE = ("o", "s", "^", "D", "v", "P", "X", "*")

# Dash patterns for the two kinds of front, before a per-series phase is applied.
DASHES = {"initial": (1, 3), "proposed": (6, 2)}


FALLBACK_COLOR = "#888888"


def is_initial(label) -> bool:
    """True for the initial design, however the label is qualified."""
    return bool(label) and (label == "initial" or label.endswith(INITIAL_SUFFIX))


def arm_label(exp: dict, fallback: str) -> str:
    """What --aggregate-runs pools a run's curve into.

    Both axes together, not the optimizer (the arm: bayesian/sobol/random/manual) alone -
    otherwise a real rig's bayesian-proposed runs and a simulated study's bayesian runs
    would average into one curve just because they share an optimizer, which is exactly
    the mix-up experiment_type (experimental/synthetic) exists to keep apart. `fallback`
    is what the caller already falls back to when neither is set - typically the
    per-observation label, since an unlabelled run still deserves a series of its own
    rather than being silently lumped in with everything else unlabelled.

    A swept --n-initial (TrialRecord, or a sweep study's own run naming - see
    build_experiment_map) folds into the arm the same way: otherwise every replicate of
    every --n-initial size pools into one curve regardless of size, which is exactly the
    comparison a sweep over it exists to draw apart. Placed ahead of the provenance
    suffix rather than appended after it, so is_initial/base_label/style_key - which all
    match that suffix with endswith - keep working unchanged.
    """
    tech = str(exp.get("optimizer") or "").strip().lower()
    prov = str(exp.get("provenance") or "").strip().lower()
    n_initial = exp.get("n_initial")
    if tech and n_initial is not None:
        # Left out of base_label's stripped set on purpose: two different sizes of
        # the same arm need to read apart on a plot comparing them, which sharing
        # one named colour - what stripping this for base_label would do - defeats.
        tech = f"{tech} n{n_initial}"
    if tech and prov:
        return f"{tech} ({prov})"
    return tech or prov or fallback


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


def style_key(label):
    """What a series is keyed by for its line style and marker.

    Only the initial-design qualifier is stripped, so a design and the proposals it
    belongs to are one series and share a style - the marker and the front's dashes
    already tell those two apart. The provenance qualifier is kept, which is what lets a
    real run and a simulated one of the same arm read apart: they share a colour by
    design (see base_label), so a shared line and marker on top would leave them
    distinguishable by the legend text alone.
    """
    if label and label.endswith(INITIAL_SUFFIX):
        return label[:-len(INITIAL_SUFFIX)]
    return label


def styler(fig_cfg: dict, labels):
    """Colour, marker, line-style and front-style functions bound to the labels present.

    `labels` is every label in the plot, which is what lets a colour be assigned by
    position and a dash phase be spread over the series sharing a pattern - both need to
    know the whole set, not one label at a time.

    Every series gets all three of colour, line style and marker, each assigned by its
    position among the labels present. Colour alone is not identity: it is lost to
    colour-blind readers, to greyscale printing and to a figure photocopied for a
    reviewer, and two arms of one campaign are exactly the case where that matters.
    """
    named = fig_cfg["colors"]["label"]
    markers = fig_cfg["markers"]["label"]
    cycle = fig_cfg["colors"].get("cycle") or [FALLBACK_COLOR]
    # By str, so a map that carries an unlabelled series - one written before
    # build_experiment_map named those "unknown" - orders instead of raising on the
    # comparison. Such a label still gets FALLBACK_COLOR, as any unnamed one does.
    labels = sorted(set(labels), key=str)
    unnamed = sorted({base_label(l) for l in labels} - set(named))
    # One position per series, shared by its line style and marker, so the two never
    # disagree about which series they are describing.
    keys = sorted({style_key(l) for l in labels}, key=str)

    def color(label):
        base = base_label(label)
        if base in named:
            return named[base]
        return cycle[unnamed.index(base) % len(cycle)] if base in unnamed else FALLBACK_COLOR

    def marker(label):
        """The design keeps its own marker, so it reads as exploration wherever it sits."""
        if is_initial(label):
            return markers.get("initial", "^")
        named_marker = markers.get(base_label(label))
        if named_marker:
            return named_marker
        return MARKER_CYCLE[keys.index(style_key(label)) % len(MARKER_CYCLE)]

    def line_style(label):
        """The series' own dash pattern, so it survives greyscale and colour blindness.

        Solid first, so a plot of one series looks as it always did.
        """
        key = style_key(label)
        index = keys.index(key) if key in keys else 0
        return LINE_STYLE_CYCLE[index % len(LINE_STYLE_CYCLE)]

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

    return color, marker, line_style, front_style