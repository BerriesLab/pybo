"""The true objective landscape behind a campaign, for plotting under the observations.

A campaign shows where an optimizer looked; the ground truth shows what there was to
find. Drawing one under the other is what says whether a front is near the best available
or merely the best of what was tried - which the observations alone cannot answer.

Two ways to cover the space, because they fail differently:

* ``random`` draws N quasi-random samples. The count is what you set whatever the
  dimension, so it stays usable on a problem with many parameters, and it respects the
  input constraints because it goes through the same sampler the runs draw with.
* ``grid`` steps every axis by a fixed spacing. Even coverage and reproducible, but the
  point count is exponential in the number of parameters - hence MAX_POINTS.

This is the only part of the campaign analysis that needs the problem itself rather than
its records, so it is kept apart: nothing else here imports pybo or torch, and a plot that
is not asked for a ground truth never pays for them.
"""
import itertools

import torch

from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective

METHODS = ("random", "grid")
DEFAULT_METHOD = "random"
DEFAULT_SAMPLES = 4096
DEFAULT_SPACING = 0.05
# A grid over d parameters is exponential in d; refuse rather than hang.
MAX_POINTS = 500_000


def _random_X(objective, samples: int):
    """Quasi-random samples, drawn the way the runs themselves draw."""
    from pybo.samplers.sobol import SobolSampler
    sampler = SobolSampler(device=objective.device, dtype=objective.dtype,
                           objective=objective)
    return sampler.draw_samples(n=samples)


def _grid_X(objective, spacing: float):
    """Every point of a uniform grid stepping each axis by `spacing`."""
    if spacing <= 0:
        raise SystemExit("Ground-truth grid spacing must be positive.")
    bounds = objective.bounds
    axes = []
    for d in range(objective.dim):
        lo, hi = float(bounds[0, d]), float(bounds[1, d])
        # +spacing/2 so the upper bound is included when it lands on a step.
        axes.append(torch.arange(lo, hi + spacing / 2, spacing,
                                 device=objective.device, dtype=objective.dtype))
    total = 1
    for axis in axes:
        total *= len(axis)
    if total > MAX_POINTS:
        raise SystemExit(
            f"A grid of {spacing} over {objective.dim} parameters is {total:,} points "
            f"(limit {MAX_POINTS:,}). Use a wider spacing, or the random method.")
    return torch.stack([torch.stack(point) for point in itertools.product(*axes)])


def _feasible_mask(objective, X, Y_obj, Y_con):
    """True where a sample satisfies both the input and the output constraints."""
    mask = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
    for check, value in ((getattr(objective, "is_X_feasible", None), X),
                         (getattr(objective, "is_Y_feasible", None),
                          torch.cat([Y_obj, Y_con], dim=-1) if Y_con is not None else Y_obj)):
        if check is None:
            continue
        try:
            mask &= check(value)
        except Exception:  # noqa: BLE001 - an objective declaring no such constraints
            pass
    return mask


def _is_constrained(objective) -> bool:
    """True if the problem declares any constraint, on its inputs or on its outputs."""
    return any(getattr(objective, name, None) for name in
               ("ineq_Y_con", "lin_eq_X_con", "lin_ineq_X_con", "nonlin_ineq_X_con"))


def ground_truth(objective_path, x_key: str, y_key: str, method: str = DEFAULT_METHOD,
                 samples: int = DEFAULT_SAMPLES, spacing: float = DEFAULT_SPACING):
    """(points, front, constrained) for the two named objectives.

    `points` is every feasible sample of the true objective and `front` its non-dominated
    subset sorted on x, both as (x, y) lists, so a line through `front` reads as a front.

    A constrained problem returns an empty `front`: its feasible front can be disconnected
    - C2-DTLZ2's three exclusion disks are centred on its quarter circle and leave two arcs
    - and a line through the sampled front would bridge the gaps, drawing trade-offs the
    problem forbids. Telling a hole from ordinary spacing means measuring gaps against a
    threshold, which holds only until the sampling density changes, so the line is dropped
    outright instead. The cloud still shows where the front runs and where it stops, and
    `constrained` lets the caller drop the observations' own front line on the same terms.
    """
    objective = load_objective(objective_path)
    labels = [cfg.label for cfg in objective.obj_cfg or []]
    for key in (x_key, y_key):
        if key not in labels:
            raise SystemExit(f"{key!r} is not an objective of {objective_path}. "
                             f"Available: {', '.join(labels)}")
    ix, iy = labels.index(x_key), labels.index(y_key)

    X = _grid_X(objective, spacing) if method == "grid" else _random_X(objective, samples)

    Y_obj = objective.evaluate_true_objective(X)
    try:
        Y_con = objective.evaluate_true_constraint(X)
    except Exception:  # noqa: BLE001 - unconstrained problems do not define one
        Y_con = None

    constrained = _is_constrained(objective)

    Y = Y_obj[_feasible_mask(objective, X, Y_obj, Y_con)]
    if Y.numel() == 0:
        return [], [], constrained

    points = [(float(a), float(b)) for a, b in zip(Y[:, ix], Y[:, iy])]
    if constrained:
        return points, [], True

    # Non-dominated in maximization space, as the optimizer scores it.
    from botorch.utils.multi_objective import is_non_dominated
    signed = Y.clone()
    signed[..., objective.to_minimize] *= -1
    front = Y[is_non_dominated(signed)]

    edge = sorted((float(a), float(b)) for a, b in zip(front[:, ix], front[:, iy]))
    return points, edge, False