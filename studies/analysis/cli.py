"""CLI plumbing shared by the campaign-analysis scripts.

`build_analysis_parser` holds only the flags every script shares; each script's own
builder (`build_convergence_parser`, `build_pareto_parser`, `build_correlation_parser`)
starts from that and adds only the flags it needs, so a script's --help never lists a
flag that does nothing for it.
"""
import argparse
from pathlib import Path

from pybo.plotters.style import DEFAULT_STYLE, list_styles


def build_analysis_parser(description: str = "") -> argparse.ArgumentParser:
    """Flags every analysis script shares. --style/--format mirror the trial CLI so a
    campaign figure can be produced at a journal's column width like any other."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--study", action="append", required=True, type=Path,
                        help="A run or study directory (repeatable). One study compares its "
                             "replicates; several compare the studies.")
    parser.add_argument("--label", action="append", default=[],
                        help="Series name for the matching --study (repeatable). Defaults to "
                             "the directory name.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where the figure is written (defaults to the first --study).")
    parser.add_argument("--style", default=DEFAULT_STYLE, choices=list_styles(),
                        help="Publisher figure style (default: %(default)s).")
    parser.add_argument("--format", default=None, choices=["png", "pdf", "svg", "eps"],
                        help="File type, overriding the style's own.")
    return parser


def build_convergence_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus convergence.py's own --regret."""
    parser = build_analysis_parser(description=description)
    parser.add_argument("--regret", action="store_true",
                        help="Plot regret against the known optimum instead of the raw "
                             "metric. Only available when the problem declares one.")
    return parser


def build_pareto_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus pareto_2d.py's own --x/--y."""
    parser = build_analysis_parser(description=description)
    parser.add_argument("--x", default=None, help="Objective label for the x axis "
                                                 "(default: the problem's first).")
    parser.add_argument("--y", default=None, help="Objective label for the y axis "
                                                 "(default: the problem's second).")
    return parser


def build_correlation_parser(description: str = "") -> argparse.ArgumentParser:
    """The shared base plus correlation.py's own --initial-only."""
    parser = build_analysis_parser(description=description)
    parser.add_argument("--initial-only", action="store_true",
                        help="Use only the initial design, whose sampling is space-filling "
                             "and so free of the optimizer's own selection bias.")
    return parser