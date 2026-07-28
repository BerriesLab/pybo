# studies/

Sweep harness for benchmarking the Bayesian optimizer. Each study repeatedly
launches a tutorial's single-trial CLI as an isolated subprocess, varying one
thing (seed, initial-dataset size, ...), then aggregates the per-trial
`summary.json` files into one combined `results.csv`.

Each study is generic over its target: pass the tutorial CLI to launch via
`--target` (a dotted module path), so the same study script works across
benchmarks.

## Running

```
python -m studies.<name> --target <dotted.module.path> [study flags]
```

Examples:

```
python -m studies.variability_study \
    --target tutorials.multi_objective.branin_currin.main --n-replicates 20

python -m studies.variability_study \
    --target tutorials.multi_objective.branin_currin.main --n-initial 5,10,20,40 --n-replicates 5
```

Shared sweep flags (see `_common.build_sweep_parser`): `--target`, `--n-evals`,
`--q-batch`, `--n-initial`, `--base-seed`, `--output-dir`. `--n-initial` accepts
a single value or a comma-separated list; each value is run as a separate
setting. Each study adds its own extras (e.g. `--n-replicates`).

## Available studies

- `variability_study` — replicates a run over seeds to quantify run-to-run
  noise. Pass a list to `--n-initial` (e.g. `5,10,20`) to sweep initial-dataset
  size, replicating each value over seeds.

## Target contract

A valid `--target` is a tutorial CLI `main.py` that accepts the
`pybo.utils.cli.build_trial_args_parser` flags (`--n-evals --q-batch
--n-initial --seed --output-dir --plot`) and writes a `summary.json` into the
`--output-dir` it is given by calling `BayesianOptimizer.to_json` each
iteration. The study then derives the aggregated per-iteration `results.csv`
(including regret) from those summaries, so targets do not write their own
`results.csv`. See `tutorials/multi_objective/branin_currin/main.py` for the
reference implementation. Every tutorial under `tutorials/` is a valid target.

`_common.py` (and any other `_`-prefixed file) is an internal helper, not a
runnable study.
