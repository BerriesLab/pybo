"""
Single entry point for launching any experiment under experiments/.

Each experiment is generic - it targets whichever tutorial CLI you pass via
--target - so the same experiment script can be reused across benchmarks.
Anything after the experiment name is forwarded verbatim to that experiment's
own CLI. Any experiments/<name>.py file with a main() is runnable by name -
no registration needed (files starting with "_", like _common.py, are
treated as internal helpers, not experiments).

Usage:
    python -m experiments.run init_size_sensitivity --target tutorials.multi_objective.branin_currin_cli.main
    python -m experiments.run variability_study --target tutorials.multi_objective.branin_currin_cli.main --n-replicates 20
"""
import importlib
import sys
from pathlib import Path

THIS_FILE = Path(__file__).stem


def main():
    argv = sys.argv[1:]
    experiment, *rest = argv
    module = importlib.import_module(f"experiments.{experiment}")
    sys.argv = [experiment] + rest
    module.main()


if __name__ == "__main__":
    main()
