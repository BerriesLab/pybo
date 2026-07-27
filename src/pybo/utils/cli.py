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


def build_trial_args_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-evals", type=int, default=32,
                        help="Total objective evaluations per trial (the loop runs n_evals // q optimization steps).")
    parser.add_argument("--q-batch", type=int, default=1, help="q-batch size.")
    parser.add_argument("--n-initial", type=int, default=None,
                        help="Number of initial samples (defaults to 5*(dim+1)).")
    parser.add_argument("--seed", type=int, default=2063, help="Sobol sampler seed for the initial dataset.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to ./data/<timestamp>).")
    parser.add_argument("--plot", type=str2bool, default=False, help="Whether to generate plots.")
    return parser
