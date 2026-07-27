# experiments/

Sweep harness for benchmarking the Bayesian optimizer. Each experiment
repeatedly launches a tutorial's single-trial CLI as an isolated subprocess,
varying one thing (seed, initial-dataset size, ...), then aggregates the
per-trial `results.csv` files into one combined `results.csv`.

Each experiment is generic over its target: pass the tutorial CLI to launch
via `--target` (a dotted module path), so the same experiment script works
across benchmarks.

## Running

```
python -m experiments.<name> --target <dotted.module.path> [experiment flags]
```

Examples:

```
python -m experiments.variability_study \
    --target tutorials.multi_objective.branin_currin_cli.main --n-replicates 20

python -m experiments.init_size_sensitivity \
    --target tutorials.multi_objective.branin_currin_cli.main --n-initial-values 5,10,20,40
```

Shared sweep flags (see `_common.build_sweep_parser`): `--target`, `--n-evals`,
`--q-batch`, `--n-initial`, `--base-seed`, `--output-dir`. Each experiment adds
its own extras (e.g. `--n-replicates`, `--n-initial-values`).

## Available experiments

- `variability_study` — same settings, varying only the seed, to quantify
  run-to-run noise.
- `init_size_sensitivity` — sweeps the number of initial Sobol samples used to
  seed each run, replicated over seeds.

## Target contract

A valid `--target` is a tutorial CLI `main.py` that accepts the
`pybo.utils.cli.build_trial_args_parser` flags (`--n-evals --q-batch
--n-initial --seed --output-dir --plot`) and writes its own `results.csv`
directly into the `--output-dir` it is given. See
`tutorials/multi_objective/branin_currin_cli/main.py` for the reference
implementation. Currently it is the only CLI-fied target; the rest of the
tutorials are CLI-fied once the harness is validated.

`_common.py` (and any other `_`-prefixed file) is an internal helper, not a
runnable experiment.
