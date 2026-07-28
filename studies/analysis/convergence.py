"""How the optimization metric develops, per study, averaged over replicates.

The plot the studies harness was missing: variability_study writes a results.csv that
nothing drew. Hypervolume for a multi-objective problem, best value for a single one —
whichever the runs recorded.

    python -m studies.analysis.convergence --study data/A
    python -m studies.analysis.convergence --study data/A --study data/B \
        --label baseline --label more-initial --regret

One study draws its replicates faintly under the mean; several draw one mean per study,
since overlaying every replicate of every study is unreadable.
"""
import matplotlib.pyplot as plt

from pybo.plotters.base_class import save_figure
from pybo.plotters.style import fig_cfg
from studies.analysis._common import (
    build_analysis_parser, discover_trials, metric_frame, parse_analysis_args,
)


def main():
    parser = build_analysis_parser(description=__doc__)
    parser.add_argument("--regret", action="store_true",
                        help="Plot regret against the known optimum instead of the raw "
                             "metric. Only available when the problem declares one.")
    args = parse_analysis_args(parser=parser)

    trials = discover_trials(args.study, args.label)
    df = metric_frame(trials)

    column = "regret" if args.regret else "value"
    if args.regret and df["regret"].isna().all():
        raise SystemExit("This problem declares no optimum (problem.best_value / max_hv "
                         "are null), so regret cannot be computed.")

    studies = list(dict.fromkeys(df["study"]))
    colours = fig_cfg["series"]
    if len(studies) > len(colours):
        raise SystemExit(f"{len(studies)} studies but only {len(colours)} series colours. "
                         f"Add more to the active palette's `series` list, or compare fewer.")

    fig, ax = plt.subplots(figsize=fig_cfg["figsize"]["convergence"], dpi=fig_cfg["dpi"])

    for colour, study in zip(colours, studies):
        sub = df[df["study"] == study]

        # A single study is worth showing warts and all; several would be spaghetti.
        if len(studies) == 1:
            for run, run_df in sub.groupby("run"):
                ax.plot(run_df["evaluations"], run_df[column],
                        color=colour, linewidth=0.6, alpha=0.35, zorder=1)

        # Replicates can stop at different iterations (a trial that died early still
        # contributes what it finished), so aggregate per evaluation count.
        grouped = sub.groupby("evaluations")[column]
        mean, std, n = grouped.mean(), grouped.std(ddof=1), grouped.size()
        ax.fill_between(mean.index, mean - std.fillna(0), mean + std.fillna(0),
                        color=colour, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(mean.index, mean.values, color=colour, linewidth=1.5, zorder=3,
                label=f"{study} (n={int(n.max())})")

    # Where the initial design ends and the optimizer takes over.
    n_initial = int(df["n_initial"].iloc[0])
    if n_initial and (df["n_initial"] == n_initial).all():
        ax.axvline(n_initial, **fig_cfg["refline"]["initial_samples"], zorder=0)

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Regret" if args.regret else df["metric"].iloc[0].replace("_", " ").capitalize())
    ax.legend()

    out = (args.output_dir or args.study[0]) / ("regret" if args.regret else "convergence")
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Saved", save_figure(fig, out))
    plt.close(fig)


if __name__ == "__main__":
    main()
