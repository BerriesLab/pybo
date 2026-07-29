"""The observed Pareto front of a campaign, against the front its initial design alone
would have given.

The gap between the two is what the optimizer bought: the dimmed front comes from the
first n_initial_samples observations only, the solid one from everything.

    python -m studies.analysis.pareto_2d --study data/A
    python -m studies.analysis.pareto_2d --study data/A --study data/B \
        --label baseline --label more-initial --x Branin --y Currin

Observations are pooled across a study's replicates, so the front is the best the
configuration achieved anywhere, not the best of one lucky seed.
"""
import matplotlib.pyplot as plt

from pybo.plotters.base_class import save_figure
from pybo.plotters.style import fig_cfg, resolve
from studies.analysis.cli import build_pareto_parser
from studies.analysis.utils import (
    discover_trials, minimized, objective_labels, observation_frame,
)


def pareto_front(points, sx: float, sy: float):
    """Non-dominated points, given per-axis senses (+1 minimize, -1 maximize).

    sx/sy are +1 when that axis's objective is minimized and -1 when it is maximized
    (set by the caller from `minimized(trials)`). Multiplying each coordinate by its
    sense before comparing folds a maximized objective into minimize space.
    Domination is decided in that signed space; the points returned keep their
    real coordinates for plotting.
    """
    ordered = sorted(points, key=lambda p: (sx * p[0], sy * p[1]))
    front, best = [], float("inf")
    for x, y in ordered:
        if sy * y < best:
            front.append((x, y))
            best = sy * y
    return front


def main():
    args = build_pareto_parser(description=__doc__).parse_args()
    resolve(args.style, fmt=args.format)

    trials = discover_trials(args.study, args.label)
    labels = objective_labels(trials)
    if len(labels) < 2:
        raise SystemExit(f"A Pareto plot needs two objectives; this problem has {len(labels)}. "
                         f"Use studies.analysis.convergence instead.")
    x_key = args.x or labels[0]
    y_key = args.y or labels[1]
    for key in (x_key, y_key):
        if key not in labels:
            raise SystemExit(f"Unknown objective {key!r}. Available: {', '.join(labels)}")

    senses = minimized(trials)
    sx = 1.0 if senses[x_key] else -1.0
    sy = 1.0 if senses[y_key] else -1.0

    df = observation_frame(trials)
    studies = list(dict.fromkeys(df["study"]))
    colours = fig_cfg["series"]
    if len(studies) > len(colours):
        raise SystemExit(f"{len(studies)} studies but only {len(colours)} series colours. "
                         f"Add more to the active palette's `series` list, or compare fewer.")

    fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["pareto_analysis"], dpi=fig_cfg["dpi"])
    edge = fig_cfg["observation"]["feasible"]["edgecolor"]

    for colour, study in zip(colours, studies):
        sub = df[df["study"] == study]
        pts_all = list(zip(sub[x_key], sub[y_key]))
        pts_init = list(zip(sub[sub["is_initial"]][x_key], sub[sub["is_initial"]][y_key]))

        ax.scatter(sub[x_key], sub[y_key], s=fig_cfg["observation"]["feasible"]["s"],
                   facecolor=colour, edgecolor=edge,
                   linewidths=fig_cfg["ground_truth"]["pareto"]["linewidths"],
                   alpha=0.55, zorder=3, label=f"{study} observations")

        # What the initial design alone reached, for the comparison this plot exists for.
        init_front = pareto_front(pts_init, sx, sy)
        if init_front:
            ax.plot(*zip(*init_front), color=colour, linewidth=1.0, linestyle=":",
                    alpha=0.6, zorder=4, label=f"{study} front (initial design)")

        # The achieved front: a ring on each point so the point's own colour shows.
        front = pareto_front(pts_all, sx, sy)
        if front:
            fx, fy = zip(*front)
            ax.plot(fx, fy, color=colour, linewidth=1.4, linestyle="--", zorder=5,
                    label=f"{study} front (final)")
            ax.scatter(fx, fy, s=fig_cfg["observation"]["pareto"]["s"], marker="o",
                       facecolors="none", edgecolors=colour, linewidths=1.6, zorder=6)

    def axis_label(key):
        # Spelled out: "(min)" reads as minutes on a problem with time objectives.
        return f"{key} ({'minimize' if senses[key] else 'maximize'})"

    ax.set_xlabel(axis_label(x_key))
    ax.set_ylabel(axis_label(y_key))
    ax.legend()

    out = (args.output_dir or args.study[0]) / "pareto_2d"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Saved", save_figure(fig, out))
    plt.close(fig)


if __name__ == "__main__":
    main()
