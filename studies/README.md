# studies/

Sweep harness for benchmarking the Bayesian optimizer. Each study repeatedly
launches a tutorial's single-trial CLI as an isolated subprocess, varying one
thing (seed, initial-dataset size, ...), each trial writing its own `summary.bin`
and a `step_NNN/experiment.json` per step. Point the campaign GUI
(`python -m pybo_gui.main`) at the output to turn those into figures.

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
`--q-batch`, `--n-initial`, `--base-seed`, `--output-dir`, `--device`.
`--n-initial` accepts a single value or a comma-separated list; each value is
run as a separate setting. Each study adds its own extras (e.g.
`--n-replicates`).

`--device` is forwarded to every trial: pass `cpu` when a sweep dies on GPU
memory, which trades speed for finishing at all.

## Comparing against the random baseline

`--strategy sobol` runs a sweep that spends the same budget drawing constrained
random points instead of optimizing an acquisition function
(`pybo.optimizer.sobol.SobolOptimizer`). Both arms are scored by the same code,
so the difference between their traces is what the optimizer bought.

```
python -m studies.variability_study --target tutorials.multi_objective.branin_currin.main \
    --strategy bo    --n-replicates 10 --output-dir data/branin_currin
python -m studies.variability_study --target tutorials.multi_objective.branin_currin.main \
    --strategy sobol --n-replicates 10 --output-dir data/branin_currin
```

Both arms land side by side, told apart by their folder names. Open the result
in the GUI (`python -m pybo_gui.main data/branin_currin`), tick what you want to
compare and plot; or from a terminal, build the map once and point the plots at
it:

```
python -m pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map data/branin_currin
python -m pybo_gui.modules.bayesian_campaign_analysis.build_group_map data/branin_currin

PYBO_CAMPAIGN_DIR=data/branin_currin \
    python -m pybo_gui.modules.bayesian_campaign_analysis.plot_pareto_2d --x Branin --y Currin
```

Use the same `--base-seed`, `--n-initial` and `--n-evals` for both. Replicate k
of each arm then starts from an identical initial design, so the two arms share
`m_0` and their gains are differences in what the budget was spent on, not in
where it started.

## Available studies

- `variability_study` — replicates a run over seeds to quantify run-to-run
  noise. Pass a list to `--n-initial` (e.g. `5,10,20`) to sweep initial-dataset
  size, replicating each value over seeds.

## Target contract

A valid `--target` is a tutorial CLI `main.py` that accepts the
`pybo.utils.cli.build_trial_args_parser` flags (`--n-evals --q-batch
--n-initial --seed --output-dir --plot --device --strategy`) and writes into the
`--output-dir` it is given: `summary.bin` at the root each iteration
(`OptimizerBase.to_file`), and one `step_NNN/experiment.json` per step
(`OptimizerBase.to_json`). See `tutorials/multi_objective/branin_currin/main.py`
for the reference implementation. Every tutorial under `tutorials/` is a valid
target.

A run's directory is named `<strategy>_ninit<n>_replicate<k>_seed<s>`, so the
arm is readable from the folder and matches the `experiment_type` recorded in
the steps inside it.

`_common.py` (and any other `_`-prefixed file) is an internal helper, not a
runnable study.

## Building a ground-truth surrogate from a run

`src/ground_truth/` fits a fast polynomial surrogate to
the `parameters -> objectives` observed in a collected run's
`experiment.json` files — e.g. to replace an expensive real experiment with
a synthetic stand-in that a study can then sweep against cheaply. It reads
the same `experiment.json` schema these studies write, recursively, so it
works unmodified on any study's or tutorial's output:

```
python -m ground_truth.build_polynomial_gt --root-dir data/branin_currin --degree 2
```

See `src/ground_truth/README.md` for the full flag reference.
