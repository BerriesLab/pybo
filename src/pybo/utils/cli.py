"""
Shared argparse setup for tutorial CLIs that run a single BO trial (see
tutorials/multi_objective/branin_currin_cli/main.py for the reference usage).
Keeping the common flags here means every CLI-ified tutorial stays consistent
with the experiments/_common.py contract without repeating the same argparse
boilerplate in each tutorial's main.py.
"""
import argparse
from pathlib import Path

from pybo.utils.helpers import str2bool


def build_trial_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-opt", type=int, default=32, help="Number of optimization steps.")
    parser.add_argument("--q-batch", type=int, default=1, help="q-batch size.")
    parser.add_argument("--n-initial", type=int, default=None,
                        help="Number of initial samples (defaults to 5*(dim+1)).")
    parser.add_argument("--seed", type=int, default=2063, help="Sobol sampler seed for the initial dataset.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to ./data/<timestamp>).")
    parser.add_argument("--plot", type=str2bool, default=False, help="Whether to generate plots.")
    return parser


def parse_variability_study_args_parser():
    parser = build_trial_args_parser(description="")
    parser.add_argument("--n-replicates", type=int, default=20, help="Number of independent repeats.")
    return parser.parse_args()


def build_sweep_args_parser() -> argparse.ArgumentParser:
    parser = parse_variability_study_args_parser(description="")
    parser.add_argument("--target", required=True, help="Dotted module path to the tutorial CLI to launch.")
    return parser
