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
from pybo.utils.cli import build_trial_args_parser

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


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
    env = os.environ | {"PYTHONUNBUFFERED": "1"}
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
    df = pd.concat([pd.read_csv(p) for p in good], ignore_index=True)
    return df, n_failed
