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
- `--group-decimals` — decimals to round the range-normalised parameters to
  when deciding which observations repeat the same setting (default: `6`).
  Only affects the pooled noise std, never the fit. Parameters are normalised
  to their own observed range first, so one value works across parameters of
  very different magnitude (volts alongside nanoseconds). Repeats normally
  record identical setpoints, making this a guard against float formatting
  rather than real binning — lower it if repeated runs drifted and land in
  groups of one.

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

`build_polynomial_gt` additionally reports the block's **pooled noise std**: the
settings measured more than once are grouped together, and their spread is
pooled into one std per column. This is *pure error* — it measures the process's
own variability without reference to the fitted surface. The fit's residuals
would instead confound that noise with the polynomial being the wrong shape.

Groups are pooled, not averaged: the total squared deviation is divided by the
total degrees of freedom, so uneven group sizes are weighted by what they
actually contribute, and settings measured only once drop out rather than
counting as a variance of zero. The printed group count and `dof` say how much
the estimate is worth — at 2 or 3 dof it is barely an estimate.

## The paste-ready method

`build_polynomial_gt` prints each block as a **whole method**, ready to paste
into an objective over whatever it holds today:

```python
    def evaluate_true_objective(self, X: Tensor, noisy: bool = False) -> Tensor:
        x0 = (X[..., 0] - 10.863) / 2.12656
        ...
        tool_wear = (37.156
                     + 8.7068 * x0
                     ...)
        if noisy:
            tool_wear = tool_wear + 8.957 * torch.randn_like(tool_wear)
        return torch.stack([tool_wear, ...], dim=-1)
```

The ground truth lives in the objective, not in a file the objective loads, and
its noise lives in the same method as the quantity it belongs to — there is no
framework attribute holding either. The `noisy` branch draws on each quantity
**before** anything is derived from it, so an objective that builds a hinge or a
deviation out of one of these inherits the noise instead of stacking a second
draw on top. Blocks with no repeated settings get a `raise` in that branch
instead of a draw, so asking a deterministic ground truth for noise fails loudly.

The columns come out **in the order the labels appear in the records**, which
need not match the objective's `obj_cfg` order — the labels are printed above
the method for exactly this reason. Reorder the stack before pasting; nothing
downstream checks it, so getting it wrong is silent.

`--out` still archives the fit as JSON, including `noise_std`, `noise_groups`
and `noise_dof` per block. Nothing reads it back: it is a record of what was
fitted, not a runtime input.

### `--paste-out`: one block per quantity

An objective holds one method per physical quantity, not one per block, so
`--paste-out FILE` writes the bodies in that shape instead — the standardisation
lines and the named expression, with no `def`, no `noisy` branch and no
`return`:

```
# trackers / Orbiting Time (degree 2, pooled noise std 0.6754)
        x0 = (X[..., 0] - 10.863) / 2.12656
        x1 = (X[..., 1] - 5.36859) / 1.43657
        x2 = (X[..., 2] - 49683.3) / 25415.8
        orbiting_time = (20.9966
                         - 4.28653 * x0
                         ...
                         - 0.217852 * x2 ** 2)
```

The standardisation is repeated in every block rather than shared, because each
one is meant to land in a method of its own and has to stand alone. The header
carries the block, the label, the degree and the pooled noise std, so the number
to put in that method's `noisy` branch travels with the coefficients.

Paste the block whose **quantity** the method computes, which is not always the
block the constraint column is named after: an objective that derives its
constraint from a measurement fits the measurement, so iformac's
`_orbiting_time` comes from `trackers / Orbiting Time`, while
`constraints / Orbiting Time Deviation` is a fit of the already-banded distance
and nothing consumes it.

Nothing is written to disk — pipe or redirect stdout if you want to keep the
output.
