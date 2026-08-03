"""Pareto front of a selection of steps over three objectives.

The 2D script sweeps a sorted list to find the front; three axes have no such order, so
domination is tested pairwise in signed space here. Non-dominated observations are drawn
solid and larger, dominated ones faint, so the front reads as a surface of points rather
than a line.

    python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_pareto_3d \
        --objective tutorials/multi_objective/c2dtlz2/objective.py \
        --step studies/data/c2dtlz2/variability_study/<timestamp> \
        --x f0 --y f1 --z f2
"""
import matplotlib.pyplot as plt
import numpy as np

from pybo.plotters.style import fig_cfg, resolve
from pybo_gui.modules.bayesian_campaign_analysis.campaign_cli import build_campaign_parser, load_campaign, require_columns


def non_dominated(points: np.ndarray) -> np.ndarray:
    """Boolean mask of the non-dominated rows of `points`, in minimization space.

    A point is dominated when another is no worse on every axis and strictly better on at
    least one. Written as an O(n^2) scan because a campaign is thousands of rows at most
    and the pairwise form is the definition.
    """
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        others = points[keep]
        no_worse = np.all(others <= points[i], axis=1)
        better = np.any(others < points[i], axis=1)
        if np.any(no_worse & better):
            keep[i] = False
    return keep


def build_parser():
    parser = build_campaign_parser(description=__doc__)
    for axis in ("x", "y", "z"):
        parser.add_argument(f"--{axis}", default=None,
                            help=f"Label for the {axis} axis (default: the problem's "
                                 f"objective in that position).")
        parser.add_argument(f"--{axis}label", default=None,
                            help=f"Override the {axis} axis label.")
    return parser


def main():
    args = build_parser().parse_args()
    resolve(args.plot_style)

    problem, df, senses = load_campaign(args)
    labels = [o["label"] for o in problem["objectives"]]
    if len(labels) < 3:
        raise SystemExit(f"A 3D Pareto plot needs three objectives; this problem has "
                         f"{len(labels)}. Use pybo_gui.modules.bayesian_campaign_analysis.campaign_pareto instead.")
    keys = [args.x or labels[0], args.y or labels[1], args.z or labels[2]]
    require_columns(df, keys)

    # Signed space: negate the maximized axes so domination is "smaller is better".
    sense = np.array([1.0 if senses.get(key, True) else -1.0 for key in keys])

    feasible = df[df["feasible"]]
    infeasible = df[~df["feasible"]]
    if feasible.empty:
        raise SystemExit("Every selected observation is infeasible under those constraints.")

    values = feasible[keys].to_numpy(dtype=float)
    mask = non_dominated(values * sense)

    fig = plt.figure(figsize=fig_cfg["figsize"]["pareto_analysis"], dpi=fig_cfg["dpi"])
    ax = fig.add_subplot(projection="3d")
    # A 3D axes draws its tick labels and axis titles around the projected cube, outside
    # anything a layout engine measures, so they get clipped at the figure edge. Shrinking
    # the cube itself is the only lever that frees space for them.
    ax.set_box_aspect(None, zoom=0.82)
    colour = fig_cfg["series"][0]
    edge = fig_cfg["observation"]["feasible"]["edgecolor"]
    size = fig_cfg["observation"]["feasible"]["s"]

    if len(infeasible):
        ax.scatter(*[infeasible[key] for key in keys], s=size, facecolor="none",
                   edgecolor=edge, alpha=0.25, label="infeasible")
    ax.scatter(*values[~mask].T, s=size, facecolor=colour, edgecolor=edge, alpha=0.30,
               label="dominated")
    ax.scatter(*values[mask].T, s=fig_cfg["observation"]["pareto"]["s"],
               facecolor=colour, edgecolor="black", linewidths=1.0, alpha=0.95,
               label=f"front ({int(mask.sum())} points)")

    overrides = (args.xlabel, args.ylabel, args.zlabel)
    setters = (ax.set_xlabel, ax.set_ylabel, ax.set_zlabel)
    for key, override, setter in zip(keys, overrides, setters):
        setter(override or f"{key} ({'minimize' if senses.get(key, True) else 'maximize'})")
    ax.legend(loc="upper left")

    fig.canvas.manager.set_window_title("Campaign Pareto front (3D)")
    plt.show()


if __name__ == "__main__":
    main()
