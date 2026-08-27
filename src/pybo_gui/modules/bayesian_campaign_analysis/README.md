# bayesian_campaign_analysis/

Scripts behind the GUI's "Bayesian campaign constructor" plots and the "Ground
truth" tab: reading step records, building Pareto/hypervolume plots, scoring
campaigns, and — the subject of this file — estimating the best hypervolume a
problem allows. Every script here is a standalone `python -m` entry point;
the GUI only assembles the flags and launches it as a subprocess (see
`pybo_gui/gui/launchers.py`), so `--help` on any of them is always current.

## campaign_optimum — searching for HV*

```
python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_optimum [flags]
```

### What HV* is for

A campaign's hypervolume `HV(n)` says how much of objective space it
dominated after `n` evaluations. On its own that number is only comparable
to other campaigns — "arm A beat arm B" — never to how much there was to
dominate in the first place. `HV*`, the best hypervolume the problem allows,
turns that into an absolute scale:

- **Normalized hypervolume** `HV(n) / HV*`, reaching 1 at the optimum.
- **Regret** `HV* - HV(n)`, reaching 0 at the optimum.

Both are drawn by `plot_hypervolume --metric`, and both need `HV*` to exist
before they mean anything. It only exists where the objectives are known
analytically, so this script reads the *problem* (`--ground-truth
objective.py`), not the records of any run against it, and evaluates the
true objective noiselessly.

### Running

```
python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_optimum \
    --ground-truth tutorials/multi_objective/branin_currin/objective.py \
    --x Branin --y Currin

python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_optimum \
    --ground-truth tutorials/multi_objective/vformac/objective.py \
    --x "Material Removal Rate" --maximize "Material Removal Rate" \
    --y "Tool Wear" --samples 262144
```

### Flags

- `--ground-truth` — the run's `objective.py`. Required: an optimum is a
  property of the problem, not of a campaign's records.
- `--x` / `--y` / `--z`, or repeated `--objective` — which objective columns
  the hypervolume is measured over (2 or more). `--objective` overrides
  `--x`/`--y`/`--z` when both are given.
- `--maximize` (repeatable) — which of those are maximized rather than
  minimized. Must **agree exactly** with the objective's own declared
  `to_minimize`: the script refuses otherwise, because `HV*` and a
  campaign's `HV(n)` have to be measured in the same signed space or the
  regret is a subtraction of numbers from different scales.
- `--samples` (default `65536`) — quasi-random samples of the parameter box.
  Raise it until the convergence table's trailing gain is a fraction of a
  percent.
- `--batch` (default `4096`) — samples per batch; sets how finely the
  convergence table is reported and caps how many points are evaluated at
  once.
- `--refine` (default `6`, `0` = off) — rounds of local refinement after
  sampling. See "Refine" below.
- `--out-dir` — where `optimum.json` is written (default: beside
  `--ground-truth`).
- `--no-save` — print `HV*` but write nothing to disk. What the GUI's
  "Compute HV*" button always passes: a quick check should not leave a file
  behind in the problem's own directory as a side effect. Run the script
  from a terminal *without* this flag if you actually want `optimum.json`
  saved for the campaign tab's `HV*` label (and `Plot norm. HV` / `Plot
  regret`) to read back.

### Algorithm

**Reference point.** Taken from each objective's own declared `ObjCfg.ref_point`
and signed into minimization space — never derived from the sampled data. A
corner padded past the observations would move with how densely the space
happened to be sampled, which would make `HV*` (and every regret measured
against it) depend on an arbitrary choice rather than the problem. A problem
that declares no `ref_point` on an axis being asked for is refused.

**1. Sampling.** One `SobolSampler`, drawn from repeatedly rather than
restarted, so each batch is the *previous* batch's points plus new ones —
that is what makes the incumbent front (and its hypervolume) grow
monotonically batch over batch, and the convergence table readable as a
climb. Each batch is evaluated on the true, noiseless objective, filtered to
feasible points, projected onto the requested objectives, and folded into
the incumbent non-dominated front.

Sampling alone spreads points **uniformly over the whole parameter box**,
while the Pareto front typically occupies a small fraction of it. That
systematically undershoots: `HV*` from sampling alone is a lower bound that
only ever climbs, and it is common for it to end up *below* what an actual
Bayesian campaign reaches — an optimizer concentrates evaluations near the
front, a uniform draw does not. Left uncorrected, the regret computed
against it goes negative, which is meaningless.

**2. Refine.** Local search around the current front, meant to push the
estimate *onto* the front instead of waiting for a uniform draw to land near
it by chance:

- **Parents** are the parameters (`front_X`) behind the points currently on
  the incumbent front.
- **Children** are Gaussian perturbations of the parents —
  `children = parents + randn * span * scale` — clamped to the parameter
  bounds and filtered through the problem's own input constraints (a child
  that stepped outside the feasible region must not reach the front).
- **Budget per round is fixed, not per parent**: `repeats = batch //
  n_parents`, so a front of 6 points and one of 600 cost the same per round
  instead of the search blowing up on a large front.
- **`scale` starts at 5%** of each parameter's range and **halves every
  round** — coarse exploration first, progressively finer refinement after,
  the same shape as a pattern-search / simulated-annealing step schedule.
  With the default `--refine 6` that is six halvings, ending at ≈0.08% of
  the range.
- A rejected child (dominated by the current front) costs nothing beyond the
  evaluation.

**Front maintenance and its cap.** The incumbent front is never recomputed
from scratch — new points are merged into the existing non-dominated set
(`is_non_dominated`), which stays cheap because the set itself stays small.
On a problem whose front is a continuous curve, though, refinement keeps
producing new non-dominated neighbours indefinitely, so without a limit the
front (and the O(n²) cost of comparing it) would grow without bound. It is
therefore thinned back to `FRONT_CAP = 2000` points after every merge —
sorted along the first objective and resampled at even intervals, so the
front's full *extent* is preserved and only its *resolution* drops. The
volume this costs is of the order of one gap in the curve: a few hundredths
of a percent at this cap, smaller than the sampling error already present.

**Hypervolume itself** is computed with `botorch.utils.multi_objective
.Hypervolume`, not this package's own `hypervolume_nd` (which `campaign_gain`
and `plot_hypervolume` use on a campaign's own front of a few dozen points,
where a pure-Python O(n² d) pass is free). Refinement drives the front here
into the thousands, where that would not finish in reasonable time. The two
implementations are required to agree exactly — `tests/test_metrics.py`
checks it on random fronts in 2-D and 3-D — because `HV* - HV(n)` subtracts
numbers that must live on the same scale.

### Reading the convergence table

```
   samples               HV*        gain
      4096         1180.4412           -
      8192         1201.0733     +1.748%
     16384         1230.1201     +2.420%
       ...
    262144         1234.5601     +0.008%
```

A trailing gain that has fallen to a fraction of a percent means the
sampling estimate has settled; one still moving by whole percent means
`--samples` is too low, and every regret computed against that run's `HV*`
is understated by roughly the same shortfall.

The refinement table below it is **not** meant to be monotone: thinning a
slightly different 2000-point set on every round moves the volume by a few
hundredths of a percent either way — that wobble is the cap doing its job,
not the search losing ground. A drop of whole percent during refinement
would be neither, and should not happen.

### A shape of problem this cannot detect

Where the true front is attainable only on a measure-zero set of parameters
— the DTLZ family, whose front needs an inner term to be exactly zero — no
finite sample, however refined, ever lands on it exactly. The estimate
settles onto a plateau *short of* the true optimum, and the convergence
table cannot tell that plateau apart from real convergence: a flat trailing
gain there means only "sampling has stopped finding anything new," not
"this is `HV*`." If the problem declares a literature `max_hv` (BraninCurrin,
C2-DTLZ2, ...), the script prints the estimate against it as a cross-check —
worth checking, but not authoritative on its own, since a declared `max_hv`
often assumes a different reference-point corner than the one this run used
(see the script's own warning when the two disagree by more than 5%).

### Output — `optimum.json`

Written beside `--ground-truth` by default (or under `--out-dir`), unless
`--no-save` is given. Contents: `hv_star`, the objectives/signs/reference the
estimate was measured under, sample and front-size counts, the full
convergence and refinement tables, and the declared `max_hv` if the problem
has one. The campaign tab's `HV*` label, `Plot norm. HV` and `Plot regret`
read `hv_star` back from this file — they never compute it themselves, since
estimating it is minutes of sampling and refinement, a terminal's job (or
the Ground truth tab's "Compute HV*" button for a quick look) rather than
something a plot button should block on.
