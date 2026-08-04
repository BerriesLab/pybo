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

import torch

from pybo.plotters.style import DEFAULT_STYLE, list_styles, resolve
from pybo.utils.helpers import str2bool


def mps_available() -> bool:
    """Whether torch can use Apple's Metal backend on this machine."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(name: str = "auto") -> torch.device:
    """The torch device a run should use, from the --device flag.

    "auto" is the old hard-coded behaviour: cuda when torch reports one, else cpu. The
    point of naming a device explicitly is "cpu": a GP fit that exhausts GPU memory
    brings the run down partway through, and falling back to system memory finishes it,
    slower, rather than not at all.

    "mps" (Apple Metal) is accepted but deliberately never chosen by "auto". Metal has no
    float64, and every tutorial runs DTYPE = torch.float64, so a run that lands there
    fails on the first tensor it builds. Auto-selecting it would turn a working cpu
    default on Apple silicon into a broken one; asking for it by name, having dropped the
    tutorial to float32, is a choice this function does not need to second-guess.

    An unavailable or misspelled device fails here, with the flag in the message, rather
    than several steps later inside a tensor op.
    """
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(name)
    except RuntimeError as error:
        raise SystemExit(f"--device {name}: {error}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {name}: torch reports no CUDA device on this machine. "
                             f"Use --device cpu, or auto to pick whatever is there.")
        count = torch.cuda.device_count()
        if device.index is not None and device.index >= count:
            raise SystemExit(f"--device {name}: this machine has {count} CUDA device"
                             f"{'' if count == 1 else 's'}, numbered 0 to {count - 1}.")
    if device.type == "mps":
        if not mps_available():
            raise SystemExit(f"--device {name}: torch reports no Metal (MPS) backend on this "
                             f"machine. It needs Apple silicon and a torch built with MPS.")
        print("! --device mps: Metal has no float64. Set DTYPE = torch.float32 in the "
              "tutorial, or this run will fail on its first tensor.")
    return device


def default_output_dir(script_file: str | Path) -> Path:
    """Default run dir for a tutorial CLI: <tutorial_dir>/data/<timestamp>,
    anchored to the tutorial's own file so it is independent of the current
    working directory (mirrors how the studies anchor to Path(__file__))."""
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(script_file).resolve().parent / "data" / date_time


def unique_dir(path: str | Path) -> Path:
    """Return a run directory safe to write into without clobbering a previous
    run: `path` itself when it does not exist or is empty, otherwise the first
    free `path_NNN` (path_001, path_002, ...).

    Always absolute. The trial loop chdirs into each step's directory so the
    plotters write there, which leaves a relative --output-dir resolving against
    a different place every step: the run would nest its own output inside
    step_000, write summary.json where nothing looks for it, and eventually die
    on the path length. Resolving here fixes it for every tutorial at once,
    since they all pass --output-dir through this function."""
    path = Path(path).resolve()
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
                        help="Proposed objective evaluations per trial, on top of the initial design "
                             "(the loop runs n_evals // q optimization steps, so a trial costs "
                             "n_initial + n_evals evaluations).")
    parser.add_argument("--q-batch", type=int, default=1, help="q-batch size.")
    parser.add_argument("--noise", type=str2bool, default=False,
                        help="Whether to measure through the objective's declared noise. "
                             "Off by default, so a trial is exact unless it asks to be "
                             "noisy. Asking an objective that declares no std for noise "
                             "is an error rather than a silent exact run, so the flag "
                             "always means what it says.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many times each proposed point is measured. Only meaningful "
                             "against an objective that declares a noise std - repeating a "
                             "deterministic one returns the same number. --n-evals still counts "
                             "proposals, so a trial costs n_evals * repeats measurements. "
                             "Acted on by the tutorials wired for it; the rest ignore it.")
    parser.add_argument("--n-initial", type=int, default=None,
                        help="Number of initial samples, measured inside the loop before the first "
                             "proposal (defaults to 5*(dim+1), rounded up to a multiple of q).")
    parser.add_argument("--seed", type=int, default=2063, help="Seeds the global torch RNG.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to <tutorial_dir>/data/<timestamp>).")
    parser.add_argument("--device", default="auto",
                        help="Torch device: auto (default: cuda when available, else cpu), "
                             "cpu, cuda, or cuda:N. Force cpu when a run collapses on GPU "
                             "memory. mps (Apple Metal) works only if the tutorial is "
                             "dropped to float32, so auto never picks it.")
    parser.add_argument("--strategy", default="bo", choices=["bo", "sobol"],
                        help="How each new point is chosen: bo (default) by the acquisition "
                             "function, or sobol by a constrained random draw - the baseline "
                             "the optimizer is measured against. The initial design is drawn "
                             "the same way for both, so with the same --seed the two arms "
                             "start from an identical dataset.")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print progress.")
    parser.add_argument("--plot-style", default=DEFAULT_STYLE, choices=list_styles(),
                        help="Publisher figure style for this run (default: %(default)s).")
    parser.add_argument("--format", default=None, choices=["png", "pdf", "svg", "eps"],
                        help="File type for saved figures, overriding the style's own "
                             "(styles default to png).")
    return parser


def parse_trial_args(description: str = ""):
    """Parse the trial flags and resolve the figure settings from --plot-style / --format.

    This is the only resolve a CLI run performs: style.py deliberately does not resolve
    on import, so the chosen style is in place before the first figure is built.
    """
    args = build_trial_args_parser(description=description).parse_args()
    resolve(args.plot_style, fmt=args.format)
    return args
