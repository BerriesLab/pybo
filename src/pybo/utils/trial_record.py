"""Everything about a trial that isn't the optimizer's own measured state: what the
script asked for (or defaulted to), and where its loop currently is.

Kept off OptimizerBase on purpose - see its to_json's `extra` parameter, which this
merges into every step it writes. The optimizer has no way to know any of this
itself: not what objective it was actually pointed at (real rig vs a pybo trial -
"experiment_type"), not how many points a script drew for its initial design before
handing it any of them, not where in a loop a particular call falls. Those are a
trial script's own decisions, not the optimizer's.
"""

PROVENANCE = ("experimental", "synthetic")


class TrialRecord:
    def __init__(self, *, n_initial: int, seed: int, provenance: str, **settings):
        """
        `provenance` is "experimental" (measured on a real rig) or "synthetic" (a
        pybo trial) - the default written into every step, and overridable per
        step (see step_fields) for a run that mixes the two, such as one warm-
        started from a real recorded dataset (--init-data) and then continuing
        with synthetic proposals: its own initial-design steps are experimental
        even though the run as a whole is not.

        `**settings` is whatever else the trial wants recorded alongside every
        step - n_evals, q, noise, repeats, device, whatever a given tutorial's own
        flags resolved to. Kept open-ended rather than named one by one, since
        different tutorials accept different flags and none of them belong on
        this class more than any other.
        """
        if provenance not in PROVENANCE:
            raise ValueError(f"provenance must be one of {PROVENANCE}, got {provenance!r}")
        self.n_initial = n_initial
        self.seed = seed
        self.provenance = provenance
        self.settings = settings

    def step_fields(self, step_index: int, repetition: int = 0,
                    provenance: str | None = None) -> dict:
        """The run-level fields for one step's experiment.json - pass as
        OptimizerBase.to_json's `extra`.

        `provenance`, when given, overrides this instance's own for this one step
        only (see the constructor's note on a mixed-provenance run); otherwise the
        instance's default is used.

        Deliberately omits "optimizer": to_json already writes the arm from the
        optimizer's own optimizer_type, which is the one part of a step record
        that really is the optimizer's to say, not this class's to repeat or
        override.
        """
        if provenance is not None and provenance not in PROVENANCE:
            raise ValueError(f"provenance must be one of {PROVENANCE}, got {provenance!r}")
        return {
            "experiment_type": provenance or self.provenance,
            "n_initial": self.n_initial,
            "seed": self.seed,
            "step_index": step_index,
            "repetition": repetition,
            **self.settings,
        }