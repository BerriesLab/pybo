"""
Run-to-run variability (reproducibility) study.

Runs the same tutorial CLI several times with identical settings, varying
only the random seed, to quantify run-to-run noise in the optimization
trajectory. Works against any tutorial CLI that follows the studies._common contract, e.g.:
    python -m studies.variability_study --target tutorials.multi_objective.branin_currin_cli.main
"""
from datetime import datetime
from pathlib import Path
from studies._common import run_trial, collect_results, build_sweep_parser


def parse_args():
    parser = build_sweep_parser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=20, help="Number of independent repeats.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tutorial_name = args.target.split(".")[-2]
        output_dir = Path(__file__).parent / "data" / tutorial_name / "variability_study" / date_time
    output_dir.mkdir(parents=True, exist_ok=True)

    n_initials = args.n_initial if args.n_initial is not None else [None]
    trials = []
    for n_initial in n_initials:
        for replicate in range(args.n_replicates):
            seed = args.base_seed + replicate
            prefix = f"ninit{n_initial}_" if n_initial is not None else ""
            run_name = f"{prefix}replicate{replicate}_seed{seed}"
            summary_path = run_trial(
                target=args.target,
                cli_args={
                    "--n-evals": args.n_evals,
                    "--q-batch": args.q_batch,
                    "--n-initial": n_initial,
                    "--seed": seed,
                },
                run_name=run_name,
                output_dir=output_dir / run_name,
            )
            tags = {"n_initial": n_initial, "seed": seed, "batch_size": args.q_batch}
            trials.append((summary_path, tags))

    df, n_failed = collect_results(trials)
    results_path = output_dir / "results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nSaved {len(df)} rows from {len(trials) - n_failed} successful runs to {results_path}")
    if n_failed:
        print(f"{n_failed} of {len(trials)} trials failed and were excluded (see log above).")


if __name__ == "__main__":
    main()
