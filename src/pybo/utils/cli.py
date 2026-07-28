"""
Shared argparse setup for tutorial CLIs that run a single BO trial (see
tutorials/multi_objective/branin_currin/main.py for the reference usage).
Keeping the common flags here means every CLI-ified tutorial stays consistent
with the studies/_common.py contract without repeating the same argparse
boilerplate in each tutorial's main.py.
"""
import argparse
from datetime import datetime
from pathlib import Path

from pybo.plotters.figure_settings.config import resolve
from pybo.plotters.figure_settings.store import list_publisher_styles
from pybo.utils.helpers import str2bool


def default_output_dir(script_file: str | Path) -> Path:
    """Default run dir for a tutorial CLI: <tutorial_dir>/data/<timestamp>,
    anchored to the tutorial's own file so it is independent of the current
    working directory (mirrors how the studies anchor to Path(__file__))."""
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(script_file).resolve().parent / "data" / date_time


def unique_dir(path: str | Path) -> Path:
    """Return a run directory safe to write into without clobbering a previous
    run: `path` itself when it does not exist or is empty, otherwise the first
    free `path_NNN` (path_001, path_002, ...). The empty-dir passthrough keeps
    the studies harness working - it pre-creates an empty trial dir that the
    tutorial is expected to write into."""
    path = Path(path)
    if not path.exists() or not any(path.iterdir()):
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.name}_{i:03d}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        i += 1


def build_trial_args_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--n-evals", type=int, default=32,
                        help="Total objective evaluations per trial (the loop runs n_evals // q optimization steps).")
    parser.add_argument("--q-batch", type=int, default=1, help="q-batch size.")
    parser.add_argument("--n-initial", type=int, default=None,
                        help="Number of initial samples (defaults to 5*(dim+1)).")
    parser.add_argument("--seed", type=int, default=2063, help="Sobol sampler seed for the initial dataset.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to <tutorial_dir>/data/<timestamp>).")
    parser.add_argument("--plot", type=str2bool, default=False, help="Whether to generate plots.")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print progress.")
    parser.add_argument("--style", default=None, choices=list_publisher_styles(),
                        help="Publisher figure style for this run. Defaults to whichever style is "
                             "active in figure_settings_app/state.json.")
    return parser


def parse_trial_args(description: str = ""):
    """Parse the trial flags and apply --style before any figure is drawn.

    The style has to be applied here rather than inside main(): fig_cfg is resolved when
    the plotters are imported, which happens before argparse runs. resolve() rebuilds it
    in place, and the plotters read it at draw time, so the new values are picked up.
    """
    args = build_trial_args_parser(description=description).parse_args()
    if args.style:
        resolve(publisher=args.style)
    return args
