# ground_truth/

Fits a surrogate model — polynomial regression or Gaussian process — to the
objectives, constraints and trackers observed in a collected pybo run, for use
as a ground-truth function (e.g. to replace an expensive real experiment with a
fast stand-in).

Reads every `experiment.json` found recursively under `--root-dir` (any
`step_NNN/experiment.json` or `run/step_NNN_repYY/experiment.json` layout
works) and fits `parameters -> objectives`, `parameters -> constraints` and
`parameters -> trackers` using the labels already recorded in those files, so
it works unmodified on any tutorial's output. Blocks that were never recorded,
or that were only filled in on some observations, are skipped with a note.

## Running

```
python -m ground_truth.build_polynomial_gt [flags]
python -m ground_truth.build_gp_gt [flags]
```

or via the wrapper, which just forwards everything after the model name:

```
python -m ground_truth.build_gt polynomial [flags]
python -m ground_truth.build_gt gp [flags]
```

## Examples

```
python -m ground_truth.build_gt polynomial --root-dir data/vformac --degree 2
python -m ground_truth.build_gt gp --root-dir data/vformac --kernel matern --nu 2.5
```

## Flags

Shared:
- `--root-dir` — root folder to search recursively for `experiment.json`
  files (default: `data/vformac`).
- `--positive` — off by default. Turn on only when every observed objective
  value is physically guaranteed to be non-negative (e.g. a time or a wear
  measurement). Fits `log(y)` and predicts `exp(...)`, so predictions are
  strictly positive everywhere — including outside the observed parameter
  range. Enabling it on a dataset with any zero or negative objective value
  breaks the fit (`log` of a non-positive number is `NaN`/undefined), so
  leave it off unless you know the sign is guaranteed. It applies to the
  **objectives only**: a constraint's sign *is* its feasibility boundary
  (`f(X) >= 0` is feasible), so a log fit would be undefined on every
  infeasible observation and, on all-feasible data, would produce a surrogate
  that can never predict infeasibility. Trackers have no guaranteed sign
  either. Constraints and trackers are therefore always fit in raw space.

`build_polynomial_gt`:
- `--degree` — polynomial degree (default: `2`). Higher degrees fit the
  training data more closely but overfit fast on small datasets; there is no
  built-in cross-validation here, so sweep `--degree` by hand and compare the
  printed `R2` across runs if you need to pick one.

`build_gp_gt`:
- `--kernel {rbf,matern}` — kernel family (default: `rbf`).
- `--nu` — Matern smoothness parameter, only used with `--kernel matern`
  (default: `1.5`).
- `--n-restarts-optimizer` — random restarts for the kernel hyperparameter
  optimizer (default: `5`). A GP has no "degree" to sweep: its
  hyperparameters (length scales, noise level) are fit automatically via
  marginal-likelihood optimization, printed per objective after fitting.

Run any module with `--help` for the full, current flag list.

## Output

Both scripts print one `=== <block> ===` section for each of `objectives`,
`constraints` and `trackers`, fitting one model per column found in that block
of each observation (the `<label>_var` uncertainty companions are dropped, not
fit). A block whose values were never recorded — or only recorded on some
observations, which would leave NaN gaps — prints a skip note instead. Per
fitted block you get:
- the fitted coefficients (`build_polynomial_gt`) or kernel (`build_gp_gt`)
  per column,
- in-sample `R2` (`build_polynomial_gt`) or in-sample plus cross-validated
  `R2` per column (`build_gp_gt`),
- an `All positive` sanity check on the objectives when `--positive` is set.

Nothing is written to disk — pipe or redirect stdout if you want to keep the
output.
