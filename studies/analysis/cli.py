"""CLI plumbing shared by the campaign-analysis scripts.

`build_analysis_parser` holds only the flags every script shares; each script's own
builder (`build_convergence_parser`, `build_pareto_parser`, `build_correlation_parser`,
`build_gain_parser`) starts from that and adds only the flags it needs, so a script's
--help never lists a flag that does nothing for it. The figure flags are one step down
for the same reason: gain.py reduces a campaign to a table and has no style to pick.
"""
import argparse
from pathlib import Path

from pybo.plotters.style import DEFAULT_STYLE, list_styles


def build_analysis_parser(description: str = "") -> argparse.ArgumentParser:
    """Flags every analysis script shares: which runs to read, and where output goes."""
    # Raw: each script passes its module docstring as the description, and the default
    # formatter rewraps it into one paragraph - collapsing the formula tables and the
    # worked examples that are the reason those docstrings are worth showing.
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", action="append", required=True, type=Path,
                        help="A run or study directory (repeatable). One study compares its "
                             "replicates; several compare the studies.")
    parser.add_argument("--label", action="append", default=[],
                        help="Series name for the matching --study (repeatable). Defaults to "
                             "the directory name.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where the output is written (defaults to the first --study).")
    return parser


def build_figure_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus what a script that draws needs. --plot-style/--format mirror
    the trial CLI so a campaign figure can be produced at a journal's column width like
    any other."""
    parser = build_analysis_parser(description=description)
    parser.add_argument("--plot-style", default=DEFAULT_STYLE, choices=list_styles(),
                        help="Publisher figure style (default: %(default)s).")
    parser.add_argument("--format", default=None, choices=["png", "pdf", "svg", "eps"],
                        help="File type, overriding the style's own.")
    return parser


def build_convergence_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus convergence.py's own --regret."""
    parser = build_figure_parser(description=description)
    parser.add_argument("--regret", action="store_true",
                        help="Plot regret against the known optimum instead of the raw "
                             "metric. Only available when the problem declares one.")
    return parser


def build_pareto_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus pareto_2d.py's own --x/--y."""
    parser = build_figure_parser(description=description)
    parser.add_argument("--x", default=None, help="Objective label for the x axis "
                                                 "(default: the problem's first).")
    parser.add_argument("--y", default=None, help="Objective label for the y axis "
                                                 "(default: the problem's second).")
    return parser


def build_gain_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus gain.py's targets and convergence rule."""
    parser = build_analysis_parser(description=description)
    parser.add_argument("--tau", action="append", default=[], type=float,
                        help="Target as a fraction of the achievable gap (repeatable, "
                             "default: 0.5 0.9 0.99).")
    parser.add_argument("--optimum", type=float, default=None,
                        help="Reference optimum m*, overriding the problem's max_hv / "
                             "best_value. Needed for gamma_norm and the it columns when "
                             "the problem declares none.")
    parser.add_argument("--patience", type=int, default=10,
                        help="Iterations of improvement below --tol that mark convergence "
                             "(default: %(default)s, as BayesianOptimizer.is_converged).")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Improvement below which an iteration counts as flat "
                             "(default: %(default)s).")
    return parser


def build_correlation_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus correlation.py's own --initial-only."""
    parser = build_figure_parser(description=description)
    parser.add_argument("--initial-only", action="store_true",
                        help="Use only the initial design, whose sampling is space-filling "
                             "and so free of the optimizer's own selection bias.")
    return parser