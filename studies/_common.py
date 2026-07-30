"""
Shared helpers for study scripts that repeatedly launch a tutorial's CLI.

Each trial runs as an isolated subprocess - a crashed trial (e.g. GP fitting
diverging) is logged and skipped rather than aborting the whole sweep. Every
target CLI is expected to accept --output-dir, --plot and --verbose (the latter
two forwarded from the sweep's own flags of the same name, both defaulting to
False for a sweep), and to write a summary.json into the exact --output-dir it
was given (BayesianOptimizer.to_json does this; see
tutorials/multi_objective/branin_currin/main.py for the reference
implementation of this contract). studies/analysis reads those summary.json
files directly, so a sweep has nothing further to aggregate.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument("--target", required=True,
                        help="Dotted module path to the tutorial CLI to launch.")
    parser.add_argument("--n-evals", type=int, default=32,
                        help="Proposed objective evaluations per trial, on top of the initial design "
                             "(the loop runs n_evals // q optimization steps, so a trial costs "
                             "n_initial + n_evals evaluations).")
    parser.add_argument("--q-batch", type=int, default=1,
                        help="q-batch size per trial.")
    parser.add_argument("--n-initial", type=int_list, default=None,
                        help="Initial sample count(s) per trial, comma-separated for a sweep (e.g. 5,10,20). "
                             "Each value is a separate setting, replicated. Defaults to the target CLI's own default.")
    parser.add_argument("--base-seed", type=int, default=2063,
                        help="First seed; trials increment from here.")
    parser.add_argument("--n-replicates", type=int, default=5,
                        help="Independent repeats per setting, each with its own seed counted up from --base-seed.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to ./data/<experiment>/<timestamp>).")
    parser.add_argument("--strategy", default=None, choices=["bo", "sobol"],
                        help="Search strategy for every trial: bo, or sobol for the random "
                             "baseline. Run the sweep twice, once each, with the same "
                             "--base-seed, and studies.analysis compares the two roots.")
    parser.add_argument("--device", default=None,
                        help="Torch device forwarded to every trial (auto, cpu, cuda, cuda:N). "
                             "A sweep is where GPU memory runs out, since the trials that "
                             "exhaust it are the long ones near the end.")
    parser.add_argument("--plot", type=str2bool, default=False,
                        help="Whether each trial generates plots. Off by default: a sweep plots every step of "
                             "every trial, which dominates the runtime.")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="Whether each trial prints per-step progress. Off by default (unlike the trial CLI, "
                             "which defaults to on) so a sweep shows one tqdm bar per trial instead of interleaving "
                             "every optimizer step.")
    parser.add_argument("--plot-style", default=None,
                        help="Publisher figure style forwarded to every trial. Only matters with --plot true.")
    return parser


def run_trial(target: str, cli_args: dict, run_name: str, output_dir: Path) -> tuple[Path | None, bool]:
    """Run one trial of `target` (dotted module path to a tutorial's CLI
    main.py, e.g. "tutorials.multi_objective.branin_currin.main") as a
    subprocess. Returns (summary_path, completed).

    A trial that dies partway (e.g. GP fitting diverging at step 7 of 32) still
    has every iteration up to that point in its summary.json, because the target
    CLI rewrites the running summary after each step. Those iterations are valid,
    so the path is returned with completed=False rather than discarded.
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
    completed = result.returncode == 0
    summary_path = output_dir / "summary.json"

    if not summary_path.exists():
        print(f"!!! Trial {run_name} produced no summary.json (exit {result.returncode}) "
              f"- nothing to salvage", flush=True)
        return None, completed
    if not completed:
        print(f"!!! Trial {run_name} FAILED (exit {result.returncode}) - keeping the "
              f"iterations it completed before dying", flush=True)
    return summary_path, completed
