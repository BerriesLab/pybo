"""Pareto front of a selection of steps, against the front their initial design gives.

Reads the per-step records named by --step and the problem definition from --objective,
so axes are chosen by label and their senses default to the objective's. Observations are
pooled across the selection: the front is the best the selected steps reached anywhere,
not the best of one run.

--z adds a colour axis: a third quantity shown on the same points, which does not take
part in the domination test.

    python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_pareto \
        --objective tutorials/multi_objective/branin_currin/objective.py \
        --step studies/data/branin_currin/variability_study/2026-08-03_15-10-29 \
        --x Branin --y Currin --constraint "Currin <= 8"
"""
import matplotlib.pyplot as plt

from pybo.plotters.style import fig_cfg, resolve
from pybo_gui.modules.bayesian_campaign_analysis.campaign_cli import build_campaign_parser, load_campaign, require_columns


def pareto_front(points, sx: float, sy: float):
    """Non-dominated points, given per-axis senses (+1 minimize, -1 maximize).

    Domination is decided in signed space; the points returned keep their real
    coordinates for plotting.
    """
    ordered = sorted(points, key=lambda p: (sx * p[0], sy * p[1]))
    front, best = [], float("inf")
    for x, y in ordered:
        if sy * y < best:
            front.append((x, y))
            best = sy * y
    return front


def build_parser():
    parser = build_campaign_parser(description=__doc__)
    parser.add_argument("--x", default=None, help="Label for the x axis "
                                                  "(default: the problem's first objective).")
    parser.add_argument("--y", default=None, help="Label for the y axis "
                                                  "(default: the problem's second).")
    parser.add_argument("--z", default=None, help="Label to colour the points by. Colour "
                                                  "only: it does not affect the front.")
    parser.add_argument("--xlabel", default=None, help="Override the x axis label.")
    parser.add_argument("--ylabel", default=None, help="Override the y axis label.")
    parser.add_argument("--zlabel", default=None, help="Override the colourbar label.")
    parser.add_argument("--per-run", action="store_true",
                        help="Draw one series per run instead of pooling the selection.")
    return parser


def main():
    args = build_parser().parse_args()
    resolve(args.plot_style)

    problem, df, senses = load_campaign(args)
    labels = [o["label"] for o in problem["objectives"]]
    if len(labels) < 2:
        raise SystemExit(f"A Pareto plot needs two objectives; this problem has {len(labels)}.")
    x_key = args.x or labels[0]
    y_key = args.y or labels[1]
    require_columns(df, [k for k in (x_key, y_key, args.z) if k])

    sx = 1.0 if senses.get(x_key, True) else -1.0
    sy = 1.0 if senses.get(y_key, True) else -1.0

    series = sorted(df["run"].unique()) if args.per_run else ["selection"]
    colours = fig_cfg["series"]
    if len(series) > len(colours):
        raise SystemExit(f"{len(series)} runs but only {len(colours)} series colours. "
                         f"Add more to the palette, or drop --per-run.")

    fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["pareto_analysis"], dpi=fig_cfg["dpi"])
    edge = fig_cfg["observation"]["feasible"]["edgecolor"]
    size = fig_cfg["observation"]["feasible"]["s"]
    image = None

    for colour, name in zip(colours, series):
        sub = df if name == "selection" else df[df["run"] == name]
        feasible = sub[sub["feasible"]]
        infeasible = sub[~sub["feasible"]]

        # Infeasible observations are shown but never counted: seeing where the campaign
        # spent evaluations it could not use is the point of drawing them at all.
        if len(infeasible):
            ax.scatter(infeasible[x_key], infeasible[y_key], s=size, facecolor="none",
                       edgecolor=edge, alpha=0.35, zorder=2,
                       label=f"{name} infeasible" if name == series[0] else None)

        if args.z:
            image = ax.scatter(feasible[x_key], feasible[y_key], s=size, c=feasible[args.z],
                               cmap=fig_cfg["cmap"]["sequential"], edgecolor=edge,
                               alpha=0.85, zorder=3, label=f"{name} observations")
        else:
            ax.scatter(feasible[x_key], feasible[y_key], s=size, facecolor=colour,
                       edgecolor=edge, alpha=0.55, zorder=3, label=f"{name} observations")

        initial = feasible[feasible["is_initial"]]
        init_front = pareto_front(list(zip(initial[x_key], initial[y_key])), sx, sy)
        if init_front:
            ax.plot(*zip(*init_front), color=colour, linewidth=1.0, linestyle=":",
                    alpha=0.6, zorder=4, label=f"{name} front (initial design)")

        front = pareto_front(list(zip(feasible[x_key], feasible[y_key])), sx, sy)
        if front:
            fx, fy = zip(*front)
            ax.plot(fx, fy, color=colour, linewidth=1.4, linestyle="--", zorder=5,
                    label=f"{name} front (final)")
            ax.scatter(fx, fy, s=fig_cfg["observation"]["pareto"]["s"], marker="o",
                       facecolors="none", edgecolors=colour, linewidths=1.6, zorder=6)

    if image is not None:
        fig.colorbar(image, ax=ax, label=args.zlabel or args.z)

    def axis_label(key, override):
        # Spelled out: "(min)" reads as minutes on a problem with time objectives.
        if override:
            return override
        return f"{key} ({'minimize' if senses.get(key, True) else 'maximize'})"

    ax.set_xlabel(axis_label(x_key, args.xlabel))
    ax.set_ylabel(axis_label(y_key, args.ylabel))
    ax.legend()

    fig.canvas.manager.set_window_title("Campaign Pareto front")
    plt.show()


if __name__ == "__main__":
    main()
