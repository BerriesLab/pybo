"""Hypervolume over an arbitrary number of objectives.

Extracted from plot_hypervolume so the gain report can use it too: importing a plot
script would run its argparse against whatever sys.argv happened to hold.
"""


def pareto_front_nd(pts):
    """Non-dominated points (minimisation) for any dimensionality. O(n^2 d)."""
    front = []
    for i, pi in enumerate(pts):
        dominated = False
        for j, pj in enumerate(pts):
            if i == j:
                continue
            if all(a <= b for a, b in zip(pj, pi)) and any(a < b for a, b in zip(pj, pi)):
                dominated = True
                break
        if not dominated:
            front.append(pi)
    return front


def hypervolume_nd(points, ref):
    """Hypervolume dominated by `points` w.r.t. reference `ref` (minimisation),
    computed by recursive Hypervolume by Slicing Objectives (HSO). `points` and
    `ref` are equal-length tuples. The recursion slices along the last axis and
    drops to a 1-D length (ref - min) at the base; dominated points are handled
    naturally, so the input need not be pre-filtered to the front.

    A point is dropped unless it beats `ref` on *every* axis. The volume it
    contributes is the box it spans with the reference corner, and that box is
    empty the moment one coordinate falls on the wrong side - so such a point
    dominates nothing and belongs out of the slicing entirely. Left in, it does
    not merely contribute zero: the slice thicknesses are taken between one
    point's coordinate and the next, so a point beyond `ref` on the axis being
    sliced stretches its neighbour's slab out past the reference and inflates
    the total (a single one can triple it).

    This only ever arises with a reference the problem declared: a corner
    derived from the observations is padded past the worst of them, so every
    point is inside the box by construction, and the two branches that pick a
    reference in campaign_gain and plot_hypervolume differ in exactly this."""
    points = [p for p in points if all(a < b for a, b in zip(p, ref))]
    if not points:
        return 0.0
    if len(ref) == 1:
        return max(0.0, ref[0] - min(p[0] for p in points))
    by_last = sorted(points, key=lambda p: p[-1])
    hv = 0.0
    for k, p in enumerate(by_last):
        lo = p[-1]
        hi = by_last[k + 1][-1] if k + 1 < len(by_last) else ref[-1]
        thickness = hi - lo
        if thickness <= 0:
            continue
        active = [q[:-1] for q in by_last[: k + 1]]
        hv += hypervolume_nd(active, ref[:-1]) * thickness
    return hv
