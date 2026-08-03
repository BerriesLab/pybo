"""Hypervolume of a selection of steps, as it develops over the observations.

The per-step records hold observations but no metric history, so the hypervolume is
recomputed here from the observations and the objective's reference point, in the same
maximization space the optimizer uses (minimized objectives negated, see
OptimizerBase._compute_hypervolume). Recomputing rather than reading it back is what lets
any subset of steps - or of objectives, via --objective-label - be scored.

    python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_hypervolume \
        --objective tutorials/multi_objective/branin_currin/objective.py \
        --step studies/data/branin_currin/variability_study/2026-08-03_15-10-29
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective import Hypervolume, is_non_dominated

from pybo.plotters.style import fig_cfg, resolve
from pybo_gui.modules.bayesian_campaign_analysis.campaign_cli import build_campaign_parser, load_campaign, require_columns


def hypervolume_trace(Y: np.ndarray, ref_point: np.ndarray) -> list:
    """Cumulative hypervolume after each observation, in maximization space.

    Y and ref_point must already be signed so that larger is better on every axis.
    """
    hv = Hypervolume(torch.as_tensor(ref_point, dtype=torch.float64))
    points = torch.as_tensor(Y, dtype=torch.float64)
    trace = []
    for i in range(1, len(points) + 1):
        seen = points[:i]
        front = seen[is_non_dominated(seen)]
        # A front entirely behind the reference encloses no volume rather than a
        # negative one.
        trace.append(hv.compute(front) if len(front) else 0.0)
    return trace


def build_parser():
    parser = build_campaign_parser(description=__doc__)
    parser.add_argument("--objective-label", action="append", default=[], dest="objective_labels",
                        help="Objective to include, by label (repeatable). Defaults to every "
                             "objective the problem declares.")
    parser.add_argument("--improvement", action="store_true",
                        help="Plot the per-observation gain in hypervolume instead of the "
                             "running total.")
    return parser


def main():
    args = build_parser().parse_args()
    resolve(args.plot_style)

    problem, df, senses = load_campaign(args)
    declared = {o["label"]: o for o in problem["objectives"]}
    labels = args.objective_labels or list(declared)
    unknown = [label for label in labels if label not in declared]
    if unknown:
        raise SystemExit(f"Unknown objective(s) {', '.join(unknown)}. "
                         f"Available: {', '.join(declared)}")
    if len(labels) < 2:
        raise SystemExit(f"Hypervolume needs at least two objectives; {len(labels)} chosen.")
    require_columns(df, labels)

    reference = [declared[label]["ref_point"] for label in labels]
    if any(point is None for point in reference):
        raise SystemExit("Those objectives declare no ref_point, so there is nothing to "
                         "measure the hypervolume against.")

    # Maximization space: negate the minimized axes, and the reference with them.
    sense = np.array([-1.0 if senses.get(label, True) else 1.0 for label in labels])
    reference = np.asarray(reference, dtype=float) * sense

    runs = sorted(df["run"].unique())
    colours = fig_cfg["series"]
    if len(runs) > len(colours):
        raise SystemExit(f"{len(runs)} runs but only {len(colours)} series colours.")

    fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["convergence"], dpi=fig_cfg["dpi"])
    plotted = False

    for colour, run in zip(colours, runs):
        # Infeasible observations never enter the front, so the trace only rises on an
        # observation the constraints admit.
        sub = df[(df["run"] == run) & df["feasible"]].sort_values("observation_n")
        if sub.empty:
            continue
        plotted = True
        trace = hypervolume_trace(sub[labels].to_numpy(dtype=float) * sense, reference)
        x = np.arange(1, len(trace) + 1)
        y = np.diff(trace, prepend=0.0) if args.improvement else trace
        ax.plot(x, y, color=colour, linewidth=1.5, marker="o", markersize=3, label=run)

        n_initial = int(sub["is_initial"].sum())
        if n_initial:
            ax.axvline(n_initial, **fig_cfg["refline"]["initial_samples"], zorder=0)

    if not plotted:
        raise SystemExit("Every selected observation is infeasible under those constraints.")

    # Only comparable against the whole declared problem: a subset of objectives has a
    # different maximum than the one the problem records.
    if (not args.improvement and problem["max_hv"] is not None
            and len(labels) == len(declared) and not args.maximize and not args.minimize):
        ax.axhline(problem["max_hv"], color="black", linestyle="--", linewidth=1.0,
                   label="max HV")

    ax.set_xlabel("Feasible observations")
    ax.set_ylabel("Hypervolume improvement" if args.improvement else "Hypervolume")
    ax.legend(loc="upper left")

    fig.canvas.manager.set_window_title(
        "Campaign hypervolume improvement" if args.improvement else "Campaign hypervolume")
    plt.show()


if __name__ == "__main__":
    main()
