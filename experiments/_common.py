"""
Shared helpers for experiment scripts that repeatedly launch a tutorial's CLI.

Each trial runs as an isolated subprocess - a crashed trial (e.g. GP fitting
diverging) is logged and skipped rather than aborting the whole sweep. Every
target CLI is expected to accept --output-dir and --plot (defaulting to
False), and to write its own results.csv directly into the exact --output-dir
it was given (see tutorials/multi_objective/branin_currin_cli/main.py for the
reference implementation of this contract).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def int_list(value: str) -> list[int]:
    """argparse type for a comma-separated list of ints (a single value yields
    a one-element list), e.g. "5" -> [5], "5,10,20" -> [5, 10, 20]."""
    return [int(v) for v in value.split(",")]


def build_sweep_parser(description: str = "") -> argparse.ArgumentParser:
    """Shared flags for every experiment sweep. Each experiment adds its own
    extras (e.g. --n-replicates) and calls parse_args()."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target", required=True, help="Dotted module path to the tutorial CLI to launch.")
    parser.add_argument("--n-evals", type=int, default=32,
                        help="Total objective evaluations per trial (the loop runs n_evals // q optimization steps).")
    parser.add_argument("--q-batch", type=int, default=1, help="q-batch size per trial.")
    parser.add_argument("--n-initial", type=int_list, default=None,
                        help="Initial sample count(s) per trial, comma-separated for a sweep (e.g. 5,10,20). "
                             "Each value is a separate setting, replicated. Defaults to the target CLI's own default.")
    parser.add_argument("--base-seed", type=int, default=2063, help="First seed; trials increment from here.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to ./data/<experiment>/<timestamp>).")
    return parser


def run_trial(target: str, cli_args: dict, run_name: str, output_dir: Path) -> Path | None:
    """Run one trial of `target` (dotted module path to a tutorial's CLI
    main.py, e.g. "tutorials.multi_objective.branin_currin_cli.main") as a
    subprocess. Returns the path to its results.csv, or None if the trial
    failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-m", target, "--output-dir", str(output_dir)]
    for flag, value in cli_args.items():
        if value is not None:
            cmd += [flag, str(value)]

    print(f"\n=== {run_name} ===", flush=True)
    # PYTHONUTF8 keeps trials from crashing on Windows when the target CLI
    # prints non-cp1252 characters (e.g. the optimizer's status emojis).
    env = os.environ | {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        print(f"!!! Trial {run_name} FAILED (exit {result.returncode})", flush=True)
        return None
    return output_dir / "results.csv"


def collect_results(csv_paths: list) -> tuple:
    """Concatenate per-trial results.csv files, dropping failed (None) trials.
    Returns (combined_dataframe, n_failed)."""
    n_failed = csv_paths.count(None)
    good = [p for p in csv_paths if p is not None]
    if not good:
        return pd.DataFrame(), n_failed
    df = pd.concat([pd.read_csv(p) for p in good], ignore_index=True)
    return df, n_failed
