"""
Run-to-run variability (reproducibility) study.

Runs the same tutorial CLI several times with identical settings, varying
only the random seed, to quantify run-to-run noise in the optimization
trajectory. Works against any tutorial CLI that follows the studies._common contract, e.g.:
    python -m studies.variability_study --target tutorials.multi_objective.branin_currin.main
"""
from datetime import datetime
from pathlib import Path
from pybo.utils.cli import unique_dir
from studies._common import run_trial, collect_results, build_sweep_parser


def main():
    args = build_sweep_parser(description=__doc__).parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tutorial_name = args.target.split(".")[-2]
        output_dir = Path(__file__).parent / "data" / tutorial_name / "variability_study" / date_time
    # Uniquify the study root, not the individual trials. Pointing a second study at an
    # --output-dir that already holds one must yield a fresh root (mystudy_001); without
    # this, every trial dir inside collides instead and gets bumped on its own, and
    # run_trial then reads the previous study's summary.json from the path it handed out.
    output_dir = unique_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_initials = args.n_initial if args.n_initial is not None else [None]
    trials = []
    for n_initial in n_initials:
        for replicate in range(args.n_replicates):
            seed = args.base_seed + replicate
            prefix = f"ninit{n_initial}_" if n_initial is not None else ""
            run_name = f"{prefix}replicate{replicate}_seed{seed}"
            summary_path, completed = run_trial(
                target=args.target,
                cli_args={
                    "--n-evals": args.n_evals,
                    "--q-batch": args.q_batch,
                    "--n-initial": n_initial,
                    "--seed": seed,
                    "--plot": args.plot,
                    "--verbose": args.verbose,
                    "--style": args.style,
                },
                run_name=run_name,
                output_dir=output_dir / run_name,
            )
            tags = {"n_initial": n_initial, "seed": seed, "batch_size": args.q_batch, "completed": completed}
            trials.append((summary_path, tags))

    df, n_missing = collect_results(trials)
    results_path = output_dir / "results.csv"
    df.to_csv(results_path, index=False)
    n_partial = sum(1 for path, tags in trials if path is not None and not tags["completed"])
    print(f"\nSaved {len(df)} rows from {len(trials) - n_missing} runs to {results_path}")
    if n_partial:
        print(f"{n_partial} of those stopped early; the iterations they did finish are "
              f"included and tagged completed=False.")
    if n_missing:
        print(f"{n_missing} of {len(trials)} trials left no output at all and were excluded (see log above).")


if __name__ == "__main__":
    main()
