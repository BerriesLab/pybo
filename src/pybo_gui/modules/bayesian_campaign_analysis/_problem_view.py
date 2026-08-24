"""The problem's own view of a campaign's records, for the metrics measured against it.

A campaign's hypervolume is computed from what its runs recorded. HV* is computed from the
problem. Any metric that subtracts one from the other - the regret, the normalized
hypervolume - is only meaningful if both were measured on the same terms, and there are two
ways the records can disagree with the problem that produced them. Both show up the same
way, as a campaign whose hypervolume exceeds the optimum it is supposed to be approaching.

NOISE

  On a simulated campaign the recorded value is the true value plus noise, while HV* is an
  optimum of the noiseless surface. A front built from noisy observations sits on the
  optimistic edge of the scatter, so it dominates more volume than the surface it was drawn
  from, and HV(n) > HV* follows - not rarely, but for most points once the noise is an
  appreciable fraction of the front's own spread.

  That is not an error to be clipped away; it is a question about what is being asked.
  "How well did this campaign do, as measured" keeps the recorded values. "How well did
  this campaign's *choices* do" reads the true objective at the parameters the optimizer
  chose - a question only a simulated campaign can answer, and the one the regret metric is
  posed for. `true_results` serves the second.

POINTS THE PROBLEM DOES NOT ALLOW

  A record is a row of numbers; nothing about it has to satisfy the problem's input
  constraints. An optimizer that proposed outside them, or a rig whose recorded setpoint
  differs from what was asked for, leaves observations in the campaign that the problem
  would never have permitted - and those points are free to be excellent, because the
  constraint that ruled them out is exactly the one keeping the achievable front where it
  is. HV* never sees them: campaign_optimum draws through the problem's own sampler, which
  rejects them. So they inflate HV(n) alone, and a handful can double it.

  `input_feasible` marks them, for callers that treat them the way an output-constraint
  violation is already treated: the evaluation still counts - it was spent - but nothing it
  attained joins the front.

Neither pass changes which points a run visited or how many evaluations it spent. Only the
score put on those choices changes.
"""


def _load(objective_path, parameter_rows):
    """(objective, usable positions, X) for the rows that can be evaluated at all.

    A row recording no parameters (a fixed technology measured as a baseline) or missing
    one of them cannot be placed in the parameter space, so it is left out here and the
    caller decides what to do about it - degrading row by row rather than failing outright
    on a partially-parameterised selection.

    Torch is imported here rather than at module scope for the reason the rest of this
    package imports it lazily: a plot that never asks the problem anything should not pay
    five seconds to find that out.
    """
    import torch

    from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective

    objective = load_objective(objective_path)
    par_labels = [cfg.label for cfg in objective.par_cfg or []]

    usable, values = [], []
    for position, parameters in enumerate(parameter_rows):
        if not parameters or any(parameters.get(label) is None for label in par_labels):
            continue
        usable.append(position)
        values.append([float(parameters[label]) for label in par_labels])

    if not usable:
        return objective, [], None
    # One tensor rather than one call per row: an objective is vectorised, and a campaign
    # of a few thousand observations would otherwise pay a full forward pass each.
    return objective, usable, torch.tensor(values, device=objective.device,
                                           dtype=objective.dtype)


def true_results(objective_path, parameter_rows, objective_keys):
    """The noiseless objective at each recorded parameter vector.

    `parameter_rows` is one dict per observation, keyed by parameter label, as the
    experiment map records them. Returns a list of the same length, each entry either a
    dict of `objective_keys` to their true values or None where that row carried no
    parameters to re-read it at.
    """
    objective, usable, X = _load(objective_path, parameter_rows)
    obj_labels = [cfg.label for cfg in objective.obj_cfg or []]
    missing = [k for k in objective_keys if k not in obj_labels]
    if missing:
        raise SystemExit(f"{objective_path} has no objective named {missing}. "
                         f"Available: {', '.join(obj_labels)}")
    index = [obj_labels.index(k) for k in objective_keys]

    out = [None] * len(parameter_rows)
    if not usable:
        return out
    Y = objective.evaluate_true_objective(X, noisy=False)
    for row, position in enumerate(usable):
        out[position] = {key: float(Y[row, column])
                         for key, column in zip(objective_keys, index)}
    return out


def input_feasible(objective_path, parameter_rows):
    """Whether each recorded parameter vector satisfies the problem's input constraints.

    Returns a list of the same length as `parameter_rows`: True, False, or None where the
    row carried no parameters to check. None is deliberately not False - "this row cannot
    be checked" and "this row breaks the constraints" call for opposite treatment, and
    silently dropping the unknowns would quietly delete every reference measurement from a
    campaign that has any.
    """
    objective, usable, X = _load(objective_path, parameter_rows)
    out = [None] * len(parameter_rows)
    if not usable:
        return out
    # The objective's own check, not a re-derivation of it: the bounds, the linear
    # inequalities and whatever else a problem decides to enforce all live behind this one
    # method, and a copy here would go stale the first time one of them changed.
    mask = objective.is_X_feasible(X=X)
    for row, position in enumerate(usable):
        out[position] = bool(mask[row])
    return out
