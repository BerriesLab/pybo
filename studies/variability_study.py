"""
Run-to-run variability (reproducibility) study.

Runs the same tutorial CLI several times with identical settings, varying
only the random seed, to quantify run-to-run noise in the optimization
trajectory. Works against any tutorial, e.g.:
    python -m studies.variability_study --target tutorials.multi_objective.branin_currin.main
"""
from datetime import datetime
from pathlib import Path
from pybo.utils.cli import unique_dir
from studies._common import run_trial, build_sweep_parser


def main():
    args = build_sweep_parser(description=__doc__).parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tutorial_name = args.target.split(".")[-2]
        output_dir = Path(__file__).parent / "data" / tutorial_name / "variability_study" / date_time
    # Uniquify the study root, not the individual trials. Pointing a second study at an
    # --output-dir that already holds one must yield a fresh root (mystudy_001).
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The arm the trials run, named as the records name it rather than as the flag spells
    # it, so a folder and the experiment_type inside it agree. Falling back to the trial
    # CLI's own default (pybo/utils/cli.py) means every run says which arm it was, not
    # only the ones that passed --strategy.
    strategy = {"bo": "bayesian", "sobol": "sobol"}[args.strategy or "bo"]

    n_initials = args.n_initial if args.n_initial is not None else [None]
    trials = []
    for n_initial in n_initials:
        for replicate in range(args.n_replicates):
            seed = args.base_seed + replicate
            prefix = f"ninit{n_initial}_" if n_initial is not None else ""
            run_name = f"{strategy}_{prefix}replicate{replicate}_seed{seed}"
            summary_path, completed = run_trial(
                target=args.target,
                cli_args={
                    "--n-evals": args.n_evals,
                    "--q-batch": args.q_batch,
                    "--n-initial": n_initial,
                    "--seed": seed,
                    "--strategy": args.strategy,
                    "--noise": args.noise,
                    "--repeats": args.repeats,
                    "--device": args.device,
                    "--plot": args.plot,
                    "--verbose": args.verbose,
                    "--plot-style": args.plot_style,
                },
                run_name=run_name,
                output_dir=output_dir / run_name,
            )
            trials.append((summary_path, completed))

    n_missing = sum(1 for path, _ in trials if path is None)
    n_partial = sum(1 for path, completed in trials if path is not None and not completed)
    print(f"\n{len(trials) - n_missing} of {len(trials)} runs left a summary.bin under {output_dir}")
    if n_partial:
        print(f"{n_partial} of those stopped early; the iterations they did finish are "
              f"still on disk (see the FAILED lines above).")
    if n_missing:
        print(f"{n_missing} of {len(trials)} trials left no output at all and were excluded.")


if __name__ == "__main__":
    main()
