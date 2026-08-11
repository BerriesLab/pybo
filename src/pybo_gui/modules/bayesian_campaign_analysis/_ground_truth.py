"""The true objective landscape behind a campaign, for plotting under the observations.

A campaign shows where an optimizer looked; the ground truth shows what there was to
find. Drawing one under the other is what says whether a front is near the best available
or merely the best of what was tried - which the observations alone cannot answer.

Two ways to cover the space, because they fail differently:

* ``random`` draws N quasi-random samples. The count is what you set whatever the
  dimension, so it stays usable on a problem with many parameters, and it respects the
  input constraints because it goes through the same sampler the runs draw with.
* ``grid`` steps every axis by a fixed spacing. Even coverage and reproducible, but the
   point count is exponential in the number of parameters, and nothing caps it: a spacing
  fine enough on one axis can be billions of points over four. The count is printed
  before the grid is built, so a run that is about to be too big says so.

This is the only part of the campaign analysis that needs the problem itself rather than
its records, so it is kept apart: nothing else here imports pybo or torch, and a plot that
is not asked for a ground truth never pays for them.
"""
import torch

from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective

METHODS = ("random", "grid")
DEFAULT_METHOD = "random"
DEFAULT_SAMPLES = 4096
DEFAULT_SPACING = 0.05


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
    # Reported, not refused: how many points are too many depends on the machine and on
    # what the grid is for, so the count goes to the caller and the choice stays theirs.
    print(f"Ground-truth grid: {spacing} over {objective.dim} parameters = {total:,} points")
    # cartesian_prod rather than stacking itertools.product: same ordering, but it stays
    # in torch instead of building one small tensor per point, which is what a grid this
    # size cannot afford. A single parameter comes back flat, hence the reshape.
    return torch.cartesian_prod(*axes).reshape(total, objective.dim)


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
    """True if the problem declares an output constraint.

    Input constraints do not count. They carve the parameter box, and the image of what
    is left under a continuous objective stays in one piece - the front may move, but it
    does not acquire holes, so a line through it still joins attainable trade-offs. An
    output constraint cuts objective space itself, which is what can leave a gap that a
    line would bridge."""
    return bool(getattr(objective, "ineq_Y_con", None))


def ground_truth(objective_path, x_key: str, y_key: str, method: str = DEFAULT_METHOD,
                 samples: int = DEFAULT_SAMPLES, spacing: float = DEFAULT_SPACING):
    """(points, front, constrained) for the two named objectives.

    `points` is every feasible sample of the true objective and `front` its non-dominated
    subset sorted on x, both as (x, y) lists, so a line through `front` reads as a front.

    A problem with output constraints returns an empty `front`: its feasible front can be
    disconnected - C2-DTLZ2's three exclusion disks are centred on its quarter circle and
    leave two arcs - and a line through the sampled front would bridge the gaps, drawing
    trade-offs the problem forbids. Telling a hole from ordinary spacing means measuring
    gaps against a threshold, which holds only until the sampling density changes, so the
    line is dropped outright instead. The cloud still shows where the front runs and where
    it stops, and `constrained` lets the caller drop the observations' own front line on
    the same terms. Input constraints do not trigger any of this - see _is_constrained.
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


def objective_landscape(objective_path, obj_key: str, par_keys, method: str = DEFAULT_METHOD,
                        samples: int = DEFAULT_SAMPLES, spacing: float = DEFAULT_SPACING):
    """The true objective over one or two parameters, for drawing under a 1D campaign.

    Where `ground_truth` answers "what trade-offs were there to find", this answers "what
    did the landscape look like" - the question a single-objective campaign asks instead.
    The Pareto front is what collapses to nothing on one objective; the landscape does not,
    so it is what a 1D plot is drawn against.

    Returns (columns, values, shape):

    * `columns` is one list per parameter in `par_keys`, and `values` the true objective
      there. On one parameter the caller draws a line through it sorted by x, so an
      infeasible sample is blanked to NaN rather than dropped - dropping would let the
      line silently bridge a forbidden region whenever no sample happened to land on
      either edge of it, worse the sparser or more random the coverage. On two
      parameters a dropped point has no such consequence: a grid is blanked the same way
      because its contour needs the grid intact, and a random cloud is only ever
      scattered, which loses nothing by dropping infeasible points outright - which is
      the same thing the masked contour in pybo's own Experiment2DPlotter says.
    * `shape` is the grid's (n_x, n_y) when two parameters were covered by ``grid``, and
      None otherwise. It is what lets a caller reshape the columns into a contour; a
      random cloud has no such shape, so it can only be scattered.

    Only one or two parameters can be drawn: beyond that a landscape is a projection,
    which would show a fold in the surface as scatter in the data. `plot_objective`
    refuses that case rather than drawing it.
    """
    if not 1 <= len(par_keys) <= 2:
        raise SystemExit(f"A landscape needs one or two parameters, not {len(par_keys)}.")

    objective = load_objective(objective_path)
    obj_labels = [cfg.label for cfg in objective.obj_cfg or []]
    par_labels = [cfg.label for cfg in objective.par_cfg or []]
    if obj_key not in obj_labels:
        raise SystemExit(f"{obj_key!r} is not an objective of {objective_path}. "
                         f"Available: {', '.join(obj_labels)}")
    for key in par_keys:
        if key not in par_labels:
            raise SystemExit(f"{key!r} is not a parameter of {objective_path}. "
                             f"Available: {', '.join(par_labels)}")
    i_obj = obj_labels.index(obj_key)
    i_par = [par_labels.index(key) for key in par_keys]

    if method == "grid":
        X = _grid_X(objective, spacing)
        # _grid_X takes the cartesian product of the axes in parameter order, so the last
        # axis varies fastest and the points reshape into a grid - but only over the
        # problem's own axes. A landscape asked for two parameters of a problem that has
        # more would be a slice through the rest, which this does not claim to draw.
        counts = [len(torch.arange(float(objective.bounds[0, d]),
                                   float(objective.bounds[1, d]) + spacing / 2, spacing))
                  for d in range(objective.dim)]
        # The grid's own shape, in the problem's parameter order - not in the order the
        # axes were asked for. Every column is reshaped by it alike, so the three arrays
        # stay one mesh; which of them runs along the rows is not something a contour
        # cares about, but disagreeing about it would shuffle the surface.
        shape = tuple(counts) if objective.dim == len(par_keys) else None
    else:
        X = _random_X(objective, samples)
        shape = None

    Y_obj = objective.evaluate_true_objective(X)
    try:
        Y_con = objective.evaluate_true_constraint(X)
    except Exception:  # noqa: BLE001 - unconstrained problems do not define one
        Y_con = None

    mask = _feasible_mask(objective, X, Y_obj, Y_con)
    # Blanked (kept, marked NaN) wherever a line could be drawn through the result - a
    # grid's contour, or the one-parameter line - so it breaks exactly where the problem
    # forbids rather than being silently bridged. Otherwise (a two-parameter random
    # cloud, drawn as a scatter with no line to bridge anything) simply dropped.
    if shape is not None or len(par_keys) == 1:
        values = torch.where(mask, Y_obj[:, i_obj], torch.nan)
    else:
        X, values = X[mask], Y_obj[mask][:, i_obj]
    columns = [[float(v) for v in X[:, d]] for d in i_par]
    return columns, [float(v) for v in values], shape