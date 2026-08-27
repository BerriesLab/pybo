"""Resume a trial interrupted mid-run.

Replays whatever a run_dir already has recorded, so a step already measured before a
crash is read back rather than re-proposed and re-simulated - the loop that follows
stays untouched, it just starts later. See tutorials/multi_objective/vformac/main.py
for the reference loop integration: one call to resume_run before the loop, and the
loop's own range changing from range(n_steps) to range(start_i, n_steps).

Whole-step atomic, deliberately: a partially-recorded step is not patched, it is
redone in full by the ordinary (unmodified) loop body - correct because the loop is
strictly sequential, see resume_run's own docstring. A crash mid-repetition-loop
(only reachable with --repeats > 1) therefore always costs that whole step's
already-measured repetitions, not just the missing ones - a known, deliberately
deferred gap, not a silent one.
"""
import json
from pathlib import Path

import torch

from pybo.utils.init_dataset import load_experiment_record

_CONFIG_NAME = "run_config.json"


def _write_run_config(run_dir: Path, **config) -> None:
    """Stamp the args that decide this run's step indexing/shape, once - so a later
    --resume can catch a drifted flag (--q-batch, --n-initial, ...) before it
    silently misreads the wrong step directories or feeds update_XY mismatched
    shapes. A no-op if the file already exists: the *original* run's config is
    authoritative, never overwritten by a resumed invocation's own values - this is
    called unconditionally, resumed or not, so it has to be safe both ways."""
    path = run_dir / _CONFIG_NAME
    if path.exists():
        return
    path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")


def _check_run_config(run_dir: Path, **config) -> None:
    """Raise a clear, specific SystemExit if `config` disagrees with what
    _write_run_config recorded for this run_dir - checked once, up front, rather
    than left to surface as a confusing shape mismatch deep inside a resumed step."""
    path = run_dir / _CONFIG_NAME
    if not path.exists():
        return  # nothing recorded yet - not actually resuming anything
    recorded = json.loads(path.read_text(encoding="utf-8"))
    mismatches = [f"{key} (recorded {recorded[key]!r}, now {value!r})"
                  for key, value in config.items()
                  if key in recorded and str(recorded[key]) != str(value)]
    if mismatches:
        raise SystemExit(
            f"--resume {run_dir}: this invocation disagrees with the original run on "
            + ", ".join(mismatches) +
            " - resuming with different flags would misread the wrong step "
            "directories or feed the optimizer inconsistent shapes. Rerun with the "
            "same flags the original trial used.")


def _phase(i: int, n_initial_steps: int, loaded_initial: bool, repeats: int) -> tuple[str, int]:
    """(source, reps) for loop iteration `i` - the same phase formula main.py's own
    loop computes, kept in one place so resume's forward scan and the live loop
    never drift apart."""
    if i < n_initial_steps:
        return "initial", (1 if loaded_initial else repeats)
    return "proposed", repeats


def _step_records(run_dir: Path, i: int, q: int, reps: int, source: str, objective,
                  expected_X: torch.Tensor | None) -> list[dict] | None:
    """The `reps` recorded update_XY payloads for step `i`, or None if it needs to be
    (re)done - whether nothing is there yet (the common case: a crash inside
    bo.optimize() always leaves a step's directories entirely unwritten) or a
    partial/corrupt write left some rows unusable.

    `expected_X`, when given (a non-modelling/initial step, where recomputing it is
    free), is compared against every recorded batch's own parameters. For a
    modelling step (expected_X is None) there is no cheap ground truth to compare
    against - recomputing one means re-running the bo.optimize() this exists to skip
    - so repeated reps are instead cross-checked against each other."""
    paths = [run_dir / f"step_{i:03d}_rep{r:02d}" / "experiment.json" for r in range(reps)]
    if not all(p.exists() for p in paths):
        return None
    records = []
    reference_X = expected_X
    for p in paths:
        try:
            record = load_experiment_record(p, objective)
        except (json.JSONDecodeError, ValueError, KeyError):
            return None  # a torn or corrupt write from an earlier crash - redo the step
        if record["source"] != source:
            return None  # this run's phase disagrees with what's recorded - redo
        new_X = record["new_X"]
        if reference_X is None:
            reference_X = new_X
        elif new_X.shape != reference_X.shape or not torch.allclose(
                new_X.to(reference_X), reference_X, atol=1e-6):
            return None  # recorded X drifts from what this step should hold - redo
        records.append(record)
    return records


def resume_run(run_dir: Path, objective, *, resume: bool, q: int, n_initial: int,
               n_initial_steps: int, loaded_initial: bool, repeats: int,
               X_initial: torch.Tensor, strategy: str, noise: bool,
               seed: int) -> tuple[int, list[dict]]:
    """Where a (possibly resumed) run should start, and what to replay to get there.

    Called once, unconditionally, right before the main loop - whether or not
    `resume` is set. Returns (start_i, prior_records): run the existing loop as
    `for i in range(start_i, n_steps)` unchanged, and feed every dict in
    `prior_records` to bo.update_XY(**record) before that loop starts.

    With resume=False, always returns (0, []), after stamping this run's config for
    a future --resume to check against. With resume=True, checks that config first
    (raising if a flag drifted), then scans step 0, 1, 2, ... forward, stopping at
    the first one that is not fully and validly recorded - see _step_records.
    Because the loop is strictly sequential, every step after that one is
    guaranteed not-yet-attempted too, so this one forward pass gives the same result
    a check on every loop iteration would, at zero cost in the loop itself.

    A resumed run also stamps the config if none was recorded yet - a run started
    before --resume existed has none to check the first time it is resumed, but
    this invocation's own config is adopted so a *second* resume has something to
    check against, rather than staying unprotected forever.
    """
    config = {"q": q, "n_initial": n_initial, "strategy": strategy, "repeats": repeats,
              "noise": noise, "seed": seed, "objective": type(objective).__name__}
    if not resume:
        _write_run_config(run_dir, **config)
        return 0, []

    _check_run_config(run_dir, **config)
    _write_run_config(run_dir, **config)

    prior_records: list[dict] = []
    i = 0
    while True:
        source, reps = _phase(i, n_initial_steps, loaded_initial, repeats)
        expected_X = None if source == "proposed" else X_initial[i * q:(i + 1) * q]
        records = _step_records(run_dir, i, q, reps, source, objective, expected_X)
        if records is None:
            break
        prior_records.extend(records)
        i += 1
    return i, prior_records
