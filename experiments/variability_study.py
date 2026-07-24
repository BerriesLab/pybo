"""
Run-to-run variability (reproducibility) study.

Runs the same tutorial CLI several times with identical settings, varying
only the random seed, to quantify run-to-run noise in the optimization
trajectory. Works against any tutorial CLI that follows the experiments._common contract, e.g.:
    python -m experiments.run variability_study --target tutorials.multi_objective.branin_currin_cli.main
"""
from datetime import datetime
from pathlib import Path
from experiments._common import run_trial, collect_results, build_sweep_parser
from pybo.utils.cli import build_trial_args_parser


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(__file__).parent / "data" / "variability_study" / date_time
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    for replicate in range(args.n_replicates):
        seed = args.base_seed + replicate
        run_name = f"replicate{replicate}_seed{seed}"
        csv_paths.append(run_trial(
            target=args.target,
            cli_args={
                "--n-opt": args.n_opt,
                "--q-batch": args.q_batch,
                "--n-initial": args.n_initial,
                "--seed": seed,
            },
            run_name=run_name,
            output_dir=output_dir / run_name,
        ))

    df, n_failed = collect_results(csv_paths)
    summary_path = output_dir / "results.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved {len(df)} rows from {len(csv_paths) - n_failed} successful runs to {summary_path}")
    if n_failed:
        print(f"{n_failed} of {len(csv_paths)} trials failed and were excluded (see log above).")


if __name__ == "__main__":
    main()
