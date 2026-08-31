# ground_truth/

Fits a surrogate model to the
objectives, constraints and trackers observed in an optimization run, for use
as a ground-truth function (e.g. to replace an expensive real experiment with a
fast stand-in).

Reads every `experiment.json` found recursively under `--root-dir` and fits
`parameters -> objectives`, `parameters -> constraints` and
`parameters -> trackers` using the labels already recorded in those files, so
it works unmodified on any tutorial's output.

## Running

```
python -m pybo.ground_truth.build_polynomial_gt [flags]
```

## Examples

```
python -m pybo.ground_truth.build_polynomial_gt --root-dir data/vformac --degree 2
```

## Flags

- `--root-dir` — root folder to search recursively for `experiment.json`
  files (default: `data/vformac`).
- `--positive` — off by default. Turn on only when every observed objective
  value is physically guaranteed to be non-negative. Fits `log(y)` and predicts `exp(...)`, so predictions are
  strictly positive objectives everywhere — including outside the observed parameter
  range. Enabling it on a dataset with any zero or negative objective value
  breaks the fit (`log` of a non-positive number is `NaN`/undefined), so
  leave it off unless you know the sign is guaranteed. It applies to the
  **objectives only**: a constraint's sign *is* its feasibility boundary
  (`f(X) >= 0` is feasible), so a log fit would be undefined on every
  infeasible observation and, on all-feasible data, would produce a surrogate
  that can never predict infeasibility. Trackers have no guaranteed sign
  either. Constraints and trackers are therefore always fit in raw space.
- `--degree` — polynomial degree (default: `2`). Higher degrees fit the
  training data more closely but overfit fast on small datasets; there is no
  built-in cross-validation here, so sweep `--degree` by hand and compare the
  printed `R2` across runs if you need to pick one.
- `--group-decimals` — decimals to round the parameters to, **in their own raw
  units**, when deciding which observations repeat the same setting (default:
  `0`, whole units). Only affects the pooled noise std, never the fit.

  What makes two runs the same setting is the rig's resolution: the smallest
  step it can actually take. The records do not show it, because a record holds
  the parameters the optimizer *asked for*, not the ones the rig *executed* —
  so one setting run three times can appear as three different rows. vformac's
  `step_000`/`step_001` sit at exactly `(60, 83, 23898, 32030)` while their
  third run `step_036` reads `(60.0, 82.853, 23898.36, 32029.97)`: the same
  experiment, recorded once as executed and once as requested. Rounding to the
  rig's own grid puts the three back together, and rounding finer than the rig
  leaves the triple as a pair plus an orphan — 8 degrees of freedom instead of
  14, on a campaign that measured 14.

  The default matches vformac and iformac, whose rigs take whole-unit
  setpoints. The authority on this is the objective, whose `ParCfg` carries a
  `resolution` per parameter; this flag is one number for all of them, which
  agrees with both campaigns today but cannot express a rig that resolves
  differently on different parameters. Check the group count against the
  repeats you know you ran: once it exceeds them, distinct settings are being
  pooled and their spread is counted as noise, which is exactly what pure error
  must exclude.

Run the module with `--help` for the full, current flag list.

## Output

The script prints one `=== <block> ===` section for each of `objectives`,
`constraints` and `trackers`, fitting one model per column found in that block
of each observation (the `<label>_var` uncertainty companions are dropped, not
fit). A block whose values were never recorded — or only recorded on some
observations, which would leave NaN gaps — prints a skip note instead. Per
fitted block you get:

- the fitted coefficients per column,
- in-sample `R2` per column,
- an `All positive` sanity check on the objectives when `--positive` is set.

It also reports the block's **pooled noise std**: the settings measured more
than once are grouped together, and their spread is pooled into one std per
column. This is *pure error* — it measures the process's
own variability without reference to the fitted surface. The fit's residuals
would instead confound that noise with the polynomial being the wrong shape.

Groups are pooled, not averaged: the total squared deviation is divided by the
total degrees of freedom, so uneven group sizes are weighted by what they
actually contribute, and settings measured only once drop out rather than
counting as a variance of zero. The printed group count and `dof` say how much
the estimate is worth — at 2 or 3 dof it is barely an estimate.

Per column, over the settings `g` measured `n_g >= 2` times, with `y_g,i` the
i-th observation at setting `g` and `mean(y_g)` that setting's mean:

```
                  sum_g sum_i (y_g,i - mean(y_g))^2
sigma = sqrt( ------------------------------------- )
                        sum_g (n_g - 1)
```

The denominator is the reported `dof`, the number of terms in the outer sums the
reported group count. Settings with `n_g = 1` contribute nothing to either sum.
Two settings count as the same one when their raw parameters agree to
`--group-decimals` decimals.

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
