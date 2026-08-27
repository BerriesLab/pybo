"""
Shared argparse setup for tutorial CLIs that run a single BO trial (see
tutorials/multi_objective/branin_currin/main.py for the reference usage).
Keeping the common flags here means every CLI-ified tutorial stays consistent
with the studies/_common.py contract without repeating the same argparse
boilerplate in each tutorial's main.py.
"""
import argparse
import torch
from datetime import datetime
from pathlib import Path
from pybo.utils.helpers import str2bool


def mps_available() -> bool:
    """Whether torch can use Apple's Metal backend on this machine."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(name: str = "cpu") -> torch.device:
    """The torch device a run should use, from the --device flag."""
    try:
        device = torch.device(name)
    except RuntimeError as error:
        raise SystemExit(f"--device {name}: {error}")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {name}: torch reports no CUDA device on this machine. "
                             f"Use --device cpu.")
        count = torch.cuda.device_count()
        if device.index is not None and device.index >= count:
            raise SystemExit(f"--device {name}: this machine has {count} CUDA device"
                             f"{'' if count == 1 else 's'}, numbered 0 to {count - 1}.")
    if device.type == "mps":
        if not mps_available():
            raise SystemExit(f"--device {name}: torch reports no Metal (MPS) backend on this "
                             f"machine. It needs Apple silicon and a torch built with MPS.")
        print("! --device mps: Metal has no float64. Pass --dtype float32 where the "
              "tutorial threads it, or set DTYPE = torch.float32 in one that does not, "
              "or this run will fail on its first tensor.")
    return device


DTYPES = {"float32": torch.float32, "float64": torch.float64}


def resolve_dtype(name: str = "float64") -> torch.dtype:
    """The torch dtype a run should use, from the --dtype flag."""
    if name not in DTYPES:
        raise SystemExit(f"--dtype {name}: expected one of {', '.join(DTYPES)}.")
    return DTYPES[name]


def default_output_dir(script_file: str | Path) -> Path:
    """Default run dir for a tutorial CLI: <tutorial_dir>/data/<timestamp>,
    anchored to the tutorial's own file so it is independent of the current
    working directory (mirrors how the studies anchor to Path(__file__))."""
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(script_file).resolve().parent / "data" / date_time


def unique_dir(path: str | Path) -> Path:
    """Return a run directory safe to write into without clobbering a previous
    run: `path` itself when it does not exist or is empty, otherwise the first
    free `path_NNN` (path_001, path_002, ...). Always absolute."""
    path = Path(path).resolve()
    if not path.exists() or not any(path.iterdir()):
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.name}_{i:03d}")
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
        i += 1


def resolve_output_dir(args, script_file: str | Path) -> Path:
    """Where a trial should write, honouring --resume.

    With --resume, `args.output_dir` must be given (a resume with nowhere to resume
    from would otherwise silently start a fresh timestamped run) and is returned
    as-is, bypassing unique_dir's anti-clobber redirect - resuming means writing back
    into the exact directory an earlier attempt used, not a fresh sibling next to it.
    Without --resume, today's unchanged behaviour: unique_dir(args.output_dir or
    default_output_dir(script_file))."""
    if getattr(args, "resume", False):
        if args.output_dir is None:
            raise SystemExit("--resume requires --output-dir pointing at the run to continue.")
        return Path(args.output_dir).resolve()
    return unique_dir(args.output_dir or default_output_dir(script_file))


def parse_trial_args(description: str = ""):
    """Parse the trial flags."""
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
                             "proposal (defaults to 5*(dim+1), rounded up to a multiple of q). With "
                             "--init-data, keeps only the first this many recorded initial points "
                             "instead of sizing a fresh draw.")
    parser.add_argument("--init-data", type=Path, default=None,
                        help="Load the initial design from a previous run's step records instead "
                             "of drawing one with Sobol: every observation with source == "
                             "'initial' found under this path (a run, or a whole study), in the "
                             "order it was measured. --n-initial then keeps only the first that "
                             "many, and is an error if fewer were recorded.")
    parser.add_argument("--shuffle-init", type=str2bool, default=False,
                        help="With --init-data, reorder the recorded initial observations "
                             "(seeded from --seed, so still reproducible) before --n-initial "
                             "truncates them, instead of always keeping the first that many in "
                             "measurement order. Off by default, since an arm comparison that "
                             "warm-starts every replicate from the identical dataset relies on "
                             "that fixed order; turn it on to have --n-initial keep a different "
                             "random subset per --seed instead.")
    parser.add_argument("--seed", type=int, default=2063, help="Seeds the global torch RNG.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory results are written to (defaults to <tutorial_dir>/data/<timestamp>).")
    parser.add_argument("--resume", type=str2bool, default=False,
                        help="Continue a previous attempt at --output-dir instead of starting "
                             "over - skip whatever step_*/experiment.json is already recorded "
                             "there. Acted on by the tutorial's own loop when it implements step "
                             "replay (currently vformac, the reference implementation); a "
                             "tutorial that doesn't still writes into the same directory instead "
                             "of redirecting to a new one, but redoes every step.")
    parser.add_argument("--device", default="cpu", type=resolve_device, metavar="DEVICE",
                        help="Torch device: cpu (default - always available, never runs out "
                             "of memory the way a GPU can mid-sweep), cuda, or cuda:N. mps "
                             "(Apple Metal) works only in float32, so it needs --dtype "
                             "float32 and a tutorial that threads it.")
    parser.add_argument("--dtype", default="float64", type=resolve_dtype,
                        metavar="{" + ",".join(sorted(DTYPES)) + "}",
                        help="Torch dtype every tensor a trial builds is made in: float64 "
                             "(default - what keeps a GP fit's kernel matrix factorizable as "
                             "it fills up) or float32 (half the memory, and the only precision "
                             "Apple Metal has, so --device mps needs it). Acted on by the "
                             "tutorials wired for it; the rest read their own DTYPE and ignore "
                             "it.")
    parser.add_argument("--strategy", default="bo", choices=["bo", "sobol", "random"],
                        help="How each new point is chosen: bo (default) by the acquisition "
                             "function, sobol by a constrained draw from a low-discrepancy "
                             "sequence, or random by an independent uniform one - the two "
                             "baselines the optimizer is measured against, differing in "
                             "whether the draws are spread by construction. The initial "
                             "design is drawn the same way (Sobol) whichever is chosen, so "
                             "with the same --seed every arm starts from an identical "
                             "dataset - or pass --init-data to start them from a recorded "
                             "one instead.")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Whether to print progress.")
    return parser.parse_args()
