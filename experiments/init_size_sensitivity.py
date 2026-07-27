"""
Sensitivity of Bayesian optimization convergence to the size of the initial dataset.

Fixes the number of optimization steps and sweeps the number of initial
(Sobol) samples used to seed each run, replicating each setting several times
with different seeds to average out sampling noise. Works against any
tutorial CLI that follows the experiments._common contract, e.g.:

    python -m experiments.init_size_sensitivity \\
        --target tutorials.multi_objective.branin_currin_cli.main
"""
import itertools
from datetime import datetime
from pathlib import Path

from experiments._common import run_trial, collect_results, build_sweep_parser


def parse_args():
    parser = build_sweep_parser(description=__doc__)
    parser.add_argument("--n-initial-values", type=str, default="5,10,20,40",
                         help="Comma-separated list of initial sample counts to sweep.")
    parser.add_argument("--n-replicates", type=int, default=5, help="Repeats per n_initial value.")
    return parser.parse_args()


def main():
    args = parse_args()
    n_initial_values = [int(v) for v in args.n_initial_values.split(",")]

    output_dir = args.output_dir
    if output_dir is None:
        date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path(__file__).parent / "data" / "init_size_sensitivity" / date_time
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_counter = itertools.count(args.base_seed)
    csv_paths = []

    for n_initial in n_initial_values:
        for replicate in range(args.n_replicates):
            seed = next(seed_counter)
            run_name = f"ninit{n_initial}_seed{seed}"
            csv_paths.append(run_trial(
                target=args.target,
                cli_args={
                    "--n-evals": args.n_evals,
                    "--q-batch": args.q_batch,
                    "--n-initial": n_initial,
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
