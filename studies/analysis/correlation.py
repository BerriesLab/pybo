"""Correlation between the parameters and the objectives a campaign explored.

Pearson correlation over the pooled observations, labelled from the problem definition
rather than array indices. Reads as "which knob moved which result", which is the
question left after a campaign finishes.

    python -m studies.analysis.correlation --study data/A

A caveat worth remembering when reading it: these are *not* observations from a designed
experiment. After the initial design the optimizer samples where it expects improvement,
so the parameters are correlated with each other by construction and a coefficient here
is a description of where the optimizer looked, not evidence of a causal effect. Use
--initial-only to restrict to the space-filling initial design, where that objection
does not apply.

With several studies a matrix is drawn per study, side by side on a shared colour scale.
"""
import matplotlib.pyplot as plt
import numpy as np

from pybo.plotters.base_class import save_figure
from pybo.plotters.style import fig_cfg, resolve
from studies.analysis.cli import build_correlation_parser
from studies.analysis.utils import (
    discover_trials, objective_labels, observation_frame, parameter_labels,
)


def main():
    args = build_correlation_parser(description=__doc__).parse_args()
    resolve(args.plot_style, fmt=args.format)

    trials = discover_trials(args.study, args.label)
    df = observation_frame(trials)
    if args.initial_only:
        df = df[df["is_initial"]]
        if df.empty:
            raise SystemExit("No initial-design observations found.")

    columns = parameter_labels(trials) + objective_labels(trials)
    columns = [c for c in columns if c in df.columns]
    studies = list(dict.fromkeys(df["study"]))

    w, h = fig_cfg["figsize"]["correlation"]
    # Constrained layout, not tight: it is the one that can place a colorbar without
    # drawing it over the matrix. save_figure leaves an existing engine alone.
    fig, axes = plt.subplots(1, len(studies), figsize=(w * len(studies), h),
                             dpi=fig_cfg["dpi"], squeeze=False, layout="constrained")

    image = None
    for ax, study in zip(axes[0], studies):
        corr = df[df["study"] == study][columns].corr(method="pearson")
        # Diverging map anchored at zero: correlation has a meaningful midpoint, so the
        # neutral colour must land on 0 rather than on the data's mean.
        image = ax.imshow(corr.to_numpy(), cmap=fig_cfg["cmap"]["diverging"],
                          vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
        ax.set_yticks(range(len(columns)), columns)
        if len(studies) > 1:
            ax.set_title(study)

        # The numbers matter more than the shade at this size, and a colour-only
        # encoding would be unreadable for a colourblind viewer.
        for i in range(len(columns)):
            for j in range(len(columns)):
                value = corr.to_numpy()[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        color="white" if abs(value) > 0.55 else "black")

    fig.colorbar(image, ax=axes[0].tolist(), label="Pearson r", shrink=0.8)

    out = (args.output_dir or args.study[0]) / "correlation"
    out.parent.mkdir(parents=True, exist_ok=True)
    print("Saved", save_figure(fig, out))
    plt.close(fig)


if __name__ == "__main__":
    main()
