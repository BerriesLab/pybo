"""When a run stopped improving, and whether it stopped at all.

Extracted from campaign_gain so it can be tested directly. It is the piece every other
number in that report hangs off - gamma, m_c, eta and n_tau are all measured at the
instant this returns - and it was the piece with no test, because a module-level script
cannot be imported without running its argparse against whatever sys.argv happens to hold.
The same reasoning that put hypervolume_nd in its own module.
"""


def terminal_plateau(metric, eps: float, patience: int, n_initial: int):
    """Index where the plateau the run finished on begins, or None if it never settled.

    `metric` is the run's metric after each observation, in a space where larger is better.
    A step counts as flat when it moved the metric by less than `eps`. The plateau must run
    at least `patience` steps and must begin at or after `n_initial`.

    The index returned is 0-based into `metric`; campaign_gain reports it as a 1-based
    evaluation count.

    WHY BACKWARDS

    Searching forwards for the first `patience` flat steps cannot tell a plateau from an
    ending. A run that goes quiet for a while and then finds something is called converged
    at the quiet patch, and everything it did afterwards disappears from the score. Walking
    back from the last evaluation there is nothing to confuse: the terminal plateau is the
    only one considered, and it is by definition the last time the run improved.

    It also turns "never converged" from an absence into a statement. A run still improving
    in its final `patience` steps has no terminal plateau and gets None - which is the run
    whose budget ran out before it settled. The budget's end is not a convergence point,
    and returning it as one is what let a censored run be averaged in with finished ones.

    WHY THE INITIAL DESIGN IS EXCLUDED

    A design is drawn blind, so its metric is routinely flat for several points together. A
    plateau running back into it would report the run as having stopped improving before
    the optimizer had proposed anything.
    """
    if patience < 1:
        raise ValueError(f"patience must be at least 1, got {patience}")
    values = list(metric)
    if len(values) < 2:
        return None

    start = len(values) - 1
    while start > 0 and abs(values[start] - values[start - 1]) < eps:
        start -= 1
    # `patience` flat steps means `patience` differences, which is what the span from
    # `start` to the end counts.
    if len(values) - 1 - start < patience or start < n_initial:
        return None
    return start
