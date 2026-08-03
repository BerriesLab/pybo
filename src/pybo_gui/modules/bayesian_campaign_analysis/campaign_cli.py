"""Flags and loading shared by the campaign plots.

Every campaign plot answers the same three questions before it draws: which steps, which
problem, and which observations count. This holds that in one place so the scripts differ
only in what they draw, and so the GUI has one flag vocabulary to build against.

Note what these scripts deliberately do *not* import: pybo.plotters.base_class forces the
Agg backend at import, which cannot open a window. A campaign plot is shown, not written,
so it stays off that path and lets matplotlib pick an interactive backend.
"""
import argparse
from pathlib import Path

import pandas as pd

from pybo.plotters.style import DEFAULT_STYLE, list_styles
from pybo_gui.modules.bayesian_campaign_analysis.constraints import ConstraintError, feasible_mask
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective, problem_definition
from pybo_gui.modules.bayesian_campaign_analysis.steps import find_steps, step_frame


def build_campaign_parser(description: str = "") -> argparse.ArgumentParser:
    """The flags every campaign plot shares."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", action="append", required=True, type=Path,
                        help="A step, run or study directory (repeatable). Every "
                             "experiment.json underneath it is read.")
    parser.add_argument("--objective", required=True, type=Path,
                        help="Path to the objective.py the run used, for the labels, "
                             "senses and reference point.")
    parser.add_argument("--maximize", action="append", default=[],
                        help="Label to treat as maximized, overriding the objective "
                             "(repeatable).")
    parser.add_argument("--minimize", action="append", default=[],
                        help="Label to treat as minimized, overriding the objective "
                             "(repeatable).")
    parser.add_argument("--constraint", action="append", default=[],
                        help="Feasibility constraint as an expression over the observation's "
                             "columns, e.g. \"Branin <= 50\" (repeatable). Infeasible "
                             "observations never contribute to a front or a hypervolume.")
    parser.add_argument("--plot-style", default=DEFAULT_STYLE, choices=list_styles(),
                        help="Publisher figure style (default: %(default)s).")
    return parser


def resolve_senses(problem: dict, args) -> dict:
    """label -> True when minimized, after applying the --maximize/--minimize overrides.

    The objective is the default so the scripts are usable without either flag; the
    overrides exist because the GUI states the sense explicitly for every axis.
    """
    senses = dict(problem["minimized"])
    for label in args.maximize:
        senses[label] = False
    for label in args.minimize:
        senses[label] = True
    conflicting = set(args.maximize) & set(args.minimize)
    if conflicting:
        raise SystemExit(f"{', '.join(sorted(conflicting))} given as both maximized and "
                         f"minimized.")
    return senses


def load_campaign(args) -> tuple[dict, pd.DataFrame, dict]:
    """(problem, frame, senses) for the selection, with a `feasible` column on the frame.

    Infeasible rows are kept rather than dropped: a Pareto plot still draws them, dimmed,
    and only the front excludes them.
    """
    problem = problem_definition(load_objective(args.objective))
    df = step_frame(find_steps(args.step))
    try:
        df["feasible"] = feasible_mask(df, args.constraint)
    except ConstraintError as exc:
        raise SystemExit(str(exc))
    return problem, df, resolve_senses(problem, args)


def require_columns(df: pd.DataFrame, labels) -> None:
    missing = [label for label in labels if label not in df.columns]
    if missing:
        raise SystemExit(f"The selected steps carry no {', '.join(missing)} column. "
                         f"Available: {', '.join(c for c in df.columns)}")
