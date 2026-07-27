"""
Shared helpers for study scripts that repeatedly launch a tutorial's CLI.

Each trial runs as an isolated subprocess - a crashed trial (e.g. GP fitting
diverging) is logged and skipped rather than aborting the whole sweep. Every
target CLI is expected to accept --output-dir, --plot and --verbose (the latter
two forwarded from the sweep's own flags of the same name, both defaulting to
False for a sweep), and to write a summary.json into the exact --output-dir it
was given (BayesianOptimizer.to_json does this; see
tutorials/multi_objective/branin_currin_cli/main.py for the reference
implementation of this contract). The aggregated per-iteration trace is
derived here from those summary.json files, so tutorial CLIs no longer write
their own results.csv.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pybo.utils.helpers import str2bool

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def int_list(value: str) -> list[int]:
    """argparse type for a comma-separated list of ints (a single value yields
    a one-element list), e.g. "5" -> [5], "5,10,20" -> [5, 10, 20]."""
    return [int(v) for v in value.split(",")]


def build_sweep_parser(description: str = "") -> argparse.ArgumentParser:
    """Shared flags for every experiment sweep, covering the parts every sweep
    has in common: what to run, how big each trial is, how many seeded repeats,
    and where output goes. An experiment that needs more (e.g. a list of
    acquisition functions to compare) adds its own before calling parse_args()."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--target", required=True, help="Dotted module path to the tutorial CLI to launch.")
    parser.add_argument("--n-evals", type=int, default=32,
                        help="Total objective evaluations per trial (the loop runs n_evals // q optimization steps).")
    parser.add_argument("--q-batch", type=int, default=1, help="q-batch size per trial.")
    parser.add_argument("--n-initial", type=int_list, default=None,
                        help="Initial sample count(s) per trial, comma-separated for a sweep (e.g. 5,10,20). "
                             "Each value is a separate setting, replicated. Defaults to the target CLI's own default.")
    parser.add_argument("--base-seed", type=int, default=2063, help="First seed; trials increment from here.")
    parser.add_argument("--n-replicates", type=int, default=20,
                        help="Independent repeats per setting, each with its own seed counted up from --base-seed.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to ./data/<experiment>/<timestamp>).")
    parser.add_argument("--plot", type=str2bool, default=False,
                        help="Whether each trial generates plots. Off by default: a sweep plots every step of "
                             "every trial, which dominates the runtime.")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="Whether each trial prints per-step progress. Off by default (unlike the trial CLI, "
                             "which defaults to on) so a sweep shows one tqdm bar per trial instead of interleaving "
                             "every optimizer step.")
    return parser


def run_trial(target: str, cli_args: dict, run_name: str, output_dir: Path) -> Path | None:
    """Run one trial of `target` (dotted module path to a tutorial's CLI
    main.py, e.g. "tutorials.multi_objective.branin_currin_cli.main") as a
    subprocess. Returns the path to its summary.json, or None if the trial
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
    return output_dir / "summary.json"


def collect_results(trials: list) -> tuple:
    """Aggregate per-trial summary.json files into one tidy per-iteration
    DataFrame. `trials` is a list of (summary_json_path, tags) pairs, where a
    None path marks a failed trial (skipped) and `tags` is a dict of per-trial
    identity columns (e.g. seed, n_initial, batch_size) attached to every row
    of that trial. Reads each run's metric trace, derives regret from the
    stored problem optimum, and returns (combined_dataframe, n_failed)."""
    n_failed = sum(1 for path, _ in trials if path is None)
    rows = []
    for path, tags in trials:
        if path is None:
            continue
        with open(path) as file:
            summary = json.load(file)
        metrics = summary["metrics"]
        problem = summary["problem"]
        if "hypervolume" in metrics:
            metric_name, values, optimum = "hypervolume", metrics["hypervolume"], problem.get("max_hv")
        else:
            metric_name, values, optimum = "best_value", metrics["best_values"], problem.get("best_value")
        for i, (value, elapsed) in enumerate(zip(values, metrics["elapsed_time"])):
            rows.append({
                **tags,
                "iteration": i + 1,
                metric_name: value,
                "regret": (optimum - value) if optimum is not None else None,
                "elapsed_time": elapsed,
            })
    return pd.DataFrame(rows), n_failed
