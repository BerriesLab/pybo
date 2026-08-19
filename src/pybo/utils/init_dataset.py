"""Load a fixed initial design from a previously recorded run.

Lets two arms (e.g. bo vs sobol), or every replicate of a study, start from the exact
same initial dataset - so what a comparison measures is the search strategy that
follows, not which points the initial design happened to draw. Reads the same
step_*/experiment.json records a completed pybo run writes (real or simulated; see
tutorials/multi_objective/branin_currin/main.py for the format written), and keeps only
the observations recorded with source == "initial".
"""
import json
import random
from pathlib import Path

import torch


def _find_steps(root: Path) -> list[Path]:
    """Every experiment.json under root, in path order - root may be a single step, a
    run, or a whole study, the same depth the campaign analysis accepts."""
    if root.is_file():
        return [root]
    return sorted(root.glob("**/experiment.json"))


def _stack(entries: list[dict], paths: list[Path], group: str, labels: list[str]) -> torch.Tensor | None:
    """entries[i][group][label] for every label, as an (n, len(labels)) tensor - or None
    when the objective declares nothing of this kind (e.g. no trackers)."""
    if not labels:
        return None
    rows = []
    for entry, path in zip(entries, paths):
        block = entry.get(group) or {}
        missing = [label for label in labels if label not in block]
        if missing:
            raise ValueError(
                f"{path}: observation {entry.get('observation_n')} has no {group} "
                f"{missing} - every column the objective declares must be present, "
                f"under the label the objective uses, on a loaded initial row.")
        rows.append([float(block[label]) for label in labels])
    return torch.tensor(rows, dtype=torch.float64)


def _stack_var(entries: list[dict], group: str, labels: list[str]) -> torch.Tensor | None:
    """The <label>_var columns, or None if the objective declares none or any kept row
    leaves one unmeasured - a partly-known variance is not something the optimizer can use."""
    if not labels:
        return None
    rows = []
    for entry in entries:
        block = entry.get(group) or {}
        row = [block.get(f"{label}_var") for label in labels]
        if any(v is None for v in row):
            return None
        rows.append([float(v) for v in row])
    return torch.tensor(rows, dtype=torch.float64)


def load_initial_dataset(root: str | Path, objective, n_initial: int | None = None,
                         shuffle_seed: int | None = None) -> dict:
    """The rig's own initial design, read back from a previous run.

    Walks every experiment.json under `root`, keeps the observations recorded with
    source == "initial", and orders them the way the rig measured them
    (observation_n) - or, with `shuffle_seed`, in a reproducible random order instead
    (see below).

    `n_initial`, when given, keeps only the first that many - an error, not a silent
    truncation, when the run recorded fewer: a warm start shorter than asked for would
    compare arms on datasets of different sizes, which defeats the point. Left as None,
    every initial observation found is used.

    `shuffle_seed`, when given, reorders the found observations (still deterministically,
    from that seed) before `n_initial` truncates them, so a smaller subset is a random
    sample of the recorded design rather than always its first points in measurement
    order. Left as None (the default), the order - and so which points a truncation
    keeps - is the fixed one every caller has always seen, which is what an arm
    comparison that warm-starts every replicate from the identical dataset relies on.
    Pass the trial's own --seed here to vary the subset by replicate instead.

    Returns {"X", "Y_obj", "Y_obj_var", "Y_con", "Y_con_var", "Y_trk", "Y_trk_var"},
    each a tensor or, for a kind the objective declares none of (or whose variance is
    only partly known), None.
    """
    root = Path(root)
    found = []
    for path in _find_steps(root):
        record = json.loads(path.read_text(encoding="utf-8"))
        for observation in record.get("data", []):
            if observation.get("source") == "initial":
                found.append((observation, path))

    if not found:
        raise ValueError(f"No observations with source == 'initial' under {root}.")

    # The canonical order first, always - shuffling is then a reproducible reordering
    # of *that*, not of whatever order the filesystem walk happened to return.
    found.sort(key=lambda item: item[0].get("observation_n") or 0)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(found)

    if n_initial is not None:
        if n_initial > len(found):
            raise ValueError(
                f"--n-initial {n_initial} exceeds the {len(found)} initial observation"
                f"{'s' if len(found) != 1 else ''} found under {root}.")
        found = found[:n_initial]

    entries = [item[0] for item in found]
    paths = [item[1] for item in found]

    par_labels = [cfg.label for cfg in objective.par_cfg]
    obj_labels = [cfg.label for cfg in objective.obj_cfg]
    con_labels = [cfg.label for cfg in (objective.ineq_Y_con_cfg or [])]
    trk_labels = [cfg.label for cfg in (objective.trk_cfg or [])]

    return {
        "X":         _stack(entries, paths, "parameters", par_labels),
        "Y_obj":     _stack(entries, paths, "objectives", obj_labels),
        "Y_obj_var": _stack_var(entries, "objectives", obj_labels),
        "Y_con":     _stack(entries, paths, "constraints", con_labels),
        "Y_con_var": _stack_var(entries, "constraints", con_labels),
        "Y_trk":     _stack(entries, paths, "trackers", trk_labels),
        "Y_trk_var": _stack_var(entries, "trackers", trk_labels),
    }


def slice_initial_batch(initial: dict, rows: slice, device=None, dtype=None) -> dict:
    """One batch of a loaded initial dataset, as update_XY's own keyword arguments.

    `initial` is what load_initial_dataset returned; `rows` selects the batch, e.g.
    slice(i * q, (i + 1) * q). A kind the objective declares none of - Y_con on an
    unconstrained objective, any *_var when it is only partly known - is simply absent
    from the dict, the same as a caller who never mentions it to update_XY.
    """
    keys = ("Y_obj", "Y_obj_var", "Y_con", "Y_con_var", "Y_trk", "Y_trk_var")
    return {
        f"new_{key}": initial[key][rows].to(device=device, dtype=dtype)
        for key in keys if initial.get(key) is not None
    }
