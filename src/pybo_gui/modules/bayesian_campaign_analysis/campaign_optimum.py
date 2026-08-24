"""The best hypervolume the problem has to offer, HV*, by dense Sobol sampling.

A campaign's hypervolume says how much of objective space it dominated. It does not say
how much there was to dominate, and without that a curve can only be read against other
curves: an arm is "better than" another, never "close to" or "far from" what the problem
allows. HV* is what turns the first reading into the second, and it is what the two
metrics that need an absolute scale are built on - the normalized hypervolume HV(n)/HV*
and the regret HV* - HV(n), both drawn by plot_hypervolume --metric.

It exists only where the objectives are known analytically, so this reads the problem
itself rather than any campaign's records: --ground-truth points at the objective.py a run
used, the parameter box is covered by quasi-random sampling, and the true objective is
evaluated there noiselessly. The optimum is the problem's, not that of a noisy draw from
it, so nothing here is ever evaluated with noise on.

WHY IT IS AN APPROXIMATION, AND HOW TO TELL WHETHER IT SETTLED

  Sampling can only find fronts it landed on, so HV* from a finite draw is a lower bound
  that climbs with the sample count and never overshoots. A single number hides which part
  of that climb it was taken from, so the estimate is reported at every batch boundary
  along with the step it gained:

      samples             HV*        gain
         4096       1180.4412           -
         8192       1201.0733     +1.748%
        16384       1230.1201     +2.420%
        ...
       262144       1234.5601     +0.008%

  A trailing gain that has fallen to a fraction of a percent is what says the estimate has
  settled. One still moving by whole percent means --samples is too small, and every regret
  computed against it is too small by the same shortfall. There is no threshold to refuse
  on: how converged is converged enough depends on what the number is for, so the table
  goes to the reader and the choice stays theirs - the same reasoning that has
  _ground_truth print its grid size rather than capping it.

  Sobol is one sequence rather than a fresh draw per batch, so each row of the *sampling*
  table is the previous row's points plus new ones and the climb is monotone. A restarted
  sequence would give a table that jitters up and down and says nothing about convergence.

  The refinement table below it is not monotone, and is not meant to be. A front that is a
  continuous curve grows without bound under refinement, so it is thinned back to a cap
  (see merge), and thinning a slightly different set of points each round moves the volume
  by a few hundredths of a percent either way. A wobble at that scale is the cap, not the
  search losing ground; a fall of whole percent would be neither, and should not happen.

  One shape of problem defeats this, and the table cannot tell you so. Where the front is
  attainable only on a measure-zero set of parameters - the DTLZ family, whose front needs
  an inner term to be exactly zero - no finite sample lands on it, and the estimate settles
  to a plateau short of the true optimum rather than at it. A flat trailing gain then means
  "sampling has stopped finding anything", not "this is HV*". BraninCurrin, and any
  polynomial surrogate fitted to a rig, have fronts attainable on a positive-measure set,
  where the plateau is the optimum; if the problem is a synthetic benchmark, check the
  estimate against a published value before trusting it as one.

THE REFERENCE POINT, AND WHY IT MAY NOT COME FROM THE DATA

  HV* is only comparable with a campaign's HV(n) if both are measured from the same corner
  and in the same space. So the reference is taken from each objective's own declared
  ObjCfg.ref_point and signed into minimization space exactly as campaign_gain and
  plot_hypervolume do it, and a problem that declares none is refused rather than falling
  back to a corner padded past the sampled data: such a corner moves with the sample, which
  would make HV* depend on how densely it happened to be estimated, and every regret drawn
  against it meaningless.

EXAMPLES

  python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_optimum \
      --ground-truth tutorials/multi_objective/branin_currin/objective.py \
      --x Branin --y Currin

  python -m pybo_gui.modules.bayesian_campaign_analysis.campaign_optimum \
      --ground-truth tutorials/multi_objective/vformac/objective.py \
      --x "Material Removal Rate" --maximize "Material Removal Rate" \
      --y "Tool Wear" --samples 262144
"""
import argparse
import json
import os
import sys

from pybo_gui.configs.settings import data_path

# Most points the incumbent front is carried at. See merge() for why it needs a ceiling at
# all, and what thinning to it costs.
FRONT_CAP = 2000

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ground-truth", required=True, dest="ground_truth",
                    help="Path to the run's objective.py. Required: an optimum is a "
                         "property of the problem, and the records of a campaign do not "
                         "carry one.")
parser.add_argument("--x", default="", help="Result key for the first objective.")
parser.add_argument("--y", default="", help="Result key for the second objective.")
parser.add_argument("--z", default="", help="Result key for a third objective.")
parser.add_argument("--objective", action="append", default=[],
                    help="Result key for an objective (repeatable). Overrides --x/--y/--z.")
parser.add_argument("--maximize", action="append", default=[],
                    help="Result key of an objective to maximize (repeatable). Default: "
                         "minimize. Must match what campaign_gain and plot_hypervolume "
                         "are given, or HV* and HV(n) are measured in different spaces.")
parser.add_argument("--samples", type=int, default=65536,
                    help="Quasi-random samples of the parameter box (default: "
                         "%(default)s). Raise it until the table's trailing gain is a "
                         "fraction of a percent.")
parser.add_argument("--batch", type=int, default=4096,
                    help="Samples per batch (default: %(default)s). Sets how finely the "
                         "convergence table is reported, and caps how many points are "
                         "evaluated at once - the front is reduced after every batch, so "
                         "memory follows the batch rather than --samples.")
parser.add_argument("--refine", type=int, default=6,
                    help="Rounds of local refinement after the sampling pass (default: "
                         "%(default)s, 0 = off). Sampling spreads points evenly over the "
                         "whole box, which is the wrong place to spend them once the front "
                         "is known to within a few percent - and an optimizer, which "
                         "concentrates on the front, routinely beats a uniform draw there. "
                         "Left as sampling alone, HV* comes out below what a campaign "
                         "reached, and its regret goes negative. Each round perturbs the "
                         "current front's own parameters by a step half the size of the "
                         "last, so the estimate is pushed onto the front rather than "
                         "waiting to land on it.")
parser.add_argument("--out-dir", default=None,
                    help="Where optimum.json is written (default: the campaign "
                         "directory). Point this at the campaign root the runs were "
                         "selected from, not a scratch map directory, so the estimate "
                         "outlives the session that computed it.")
args = parser.parse_args()

objective_keys = args.objective or [k for k in (args.x, args.y, args.z) if k]
if not objective_keys:
    print("Give at least one objective, with --x/--y or --objective.")
    sys.exit(2)
if len(objective_keys) < 2:
    # A single objective has no volume: its campaign is scored by the best value reached
    # (campaign_gain's own substitution), and the best value of a known problem is declared
    # on the objective as `best_value` rather than sampled for.
    print("One objective has no hypervolume to maximise. A single-objective campaign is "
          "scored by its best value, which the problem declares as `best_value`.")
    sys.exit(2)
if args.batch < 1 or args.samples < 1:
    print("--samples and --batch must both be positive.")
    sys.exit(2)

# Minimization space, so every objective improves downwards - the space campaign_gain and
# plot_hypervolume measure a campaign's own hypervolume in.
signs = [-1.0 if k in set(args.maximize) else 1.0 for k in objective_keys]

# Paid once, here rather than at the top: importing a pybo objective pulls in torch, which
# is seconds this script cannot avoid but its importers can. Same reasoning as
# _ground_truth's lazy botorch import, and as _hypervolume being kept torch-free.
import torch
from botorch.utils.multi_objective import Hypervolume, is_non_dominated

from pybo.samplers.sobol import SobolSampler
from pybo_gui.modules.bayesian_campaign_analysis._ground_truth import _feasible_mask
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import (
    load_objective, problem_definition)

objective = load_objective(args.ground_truth)
problem = problem_definition(objective)

labels = [o["label"] for o in problem["objectives"]]
missing = [k for k in objective_keys if k not in labels]
if missing:
    print(f"! --ground-truth {args.ground_truth}: no objective named {missing}. "
          f"Available: {', '.join(labels)}")
    sys.exit(2)
index = [labels.index(k) for k in objective_keys]

# ---- REFERENCE POINT ----
# The problem's own, or nothing. See the docstring: a corner derived from the sample would
# make HV* a function of how densely it was estimated.
ref_by_label = {o["label"]: o["ref_point"] for o in problem["objectives"]}
unset = [k for k in objective_keys if ref_by_label[k] is None]
if unset:
    print(f"! --ground-truth {args.ground_truth}: no ref_point declared for {unset}. "
          f"HV* has to be measured from the same fixed corner a campaign's own "
          f"hypervolume is, so declare one on the ObjCfg rather than deriving it here.")
    sys.exit(2)
ref = tuple(s * ref_by_label[k] for s, k in zip(signs, objective_keys))

# The senses the problem itself declares, against the ones this was asked for. They have to
# agree: a campaign scored with the opposite sense on an axis is measuring a different
# volume, and comparing the two would be silent nonsense rather than an error.
declared = {o["label"]: (1.0 if o["to_minimize"] else -1.0)
            for o in problem["objectives"]}
disagree = [k for k, s in zip(objective_keys, signs) if declared[k] != s]
if disagree:
    print(f"! --maximize disagrees with the problem's own to_minimize for {disagree}. "
          f"HV* would be measured in a different space from the campaign's HV(n).")
    sys.exit(2)

# ---- SAMPLE, REDUCING THE FRONT AS IT GOES ----
# One sampler, drawn from repeatedly: SobolSampler keeps a single engine, so successive
# calls advance one sequence instead of restarting it. That is what makes each row of the
# table below a superset of the row before it, and the climb monotone. It is also the
# sampler the runs themselves draw with, so the input constraints carving the box are
# honoured here exactly as they were there.
sampler = SobolSampler(device=objective.device, dtype=objective.dtype, objective=objective)

sign = torch.tensor(signs, device=objective.device, dtype=objective.dtype)

# The incumbent non-dominated set: its objective values in minimization space, and the
# parameters that produced them. The parameters are kept because --refine searches around
# them - a front of objective values alone says where the best trade-offs are, but not
# where to look for better ones.
front = None
front_X = None
drawn = 0
feasible = 0
history = []
previous = None


def evaluate(X):
    """(X, P) for the feasible members of X, P being the objectives asked for in
    minimization space.

    The projection onto those objectives happens here, before any front is taken: a point
    non-dominated in the problem's full objective space can be dominated in the subspace
    this HV* is for, and keeping it would put a point on the front that does not belong on
    this one.
    """
    Y_obj = objective.evaluate_true_objective(X, noisy=False)
    try:
        Y_con = objective.evaluate_true_constraint(X, noisy=False)
    except Exception:  # noqa: BLE001 - unconstrained problems do not define one
        Y_con = None
    mask = _feasible_mask(objective, X, Y_obj, Y_con)
    return X[mask], Y_obj[mask][:, index] * sign


def merge(X, P):
    """Fold (X, P) into the incumbent front and reduce it again.

    Reduced on every merge rather than once over the whole cloud at the end: the incumbent
    stays at a few hundred points however many samples are drawn, so each reduction is
    small, while one pass over 10^5 points at once is a comparison matrix that does not
    fit. is_non_dominated maximises, hence the negation.
    """
    global front, front_X
    if not P.shape[0]:
        return
    keep = is_non_dominated(-P)
    X, P = X[keep], P[keep]
    front = P if front is None else torch.cat([front, P], dim=0)
    front_X = X if front_X is None else torch.cat([front_X, X], dim=0)
    keep = is_non_dominated(-front)
    front, front_X = front[keep], front_X[keep]

    # Thinned back to a cap. On a problem whose front is a continuous curve every child
    # landing near it is non-dominated, so refinement grows the front without bound - and
    # the reduction above is quadratic in its size, which is what stops the run finishing
    # rather than the hypervolume itself. Thinning evenly along the first objective keeps
    # the front's full extent and only its resolution drops, so the volume lost is of order
    # one gap in the curve: a few hundredths of a percent at this cap, against an estimate
    # whose sampling error is already larger. Sorted first, so "evenly spaced" is along the
    # front rather than in whatever order the points were merged in.
    if front.shape[0] > FRONT_CAP:
        order = torch.argsort(front[:, 0])
        picked = order[torch.linspace(0, front.shape[0] - 1, FRONT_CAP,
                                      device=front.device).long()]
        front, front_X = front[picked], front_X[picked]


ref_max = -torch.tensor(ref, device=objective.device, dtype=objective.dtype)


def current_hv():
    """The volume the incumbent front dominates, w.r.t. `ref`.

    botorch's Hypervolume rather than this package's own hypervolume_nd, which is what
    campaign_gain and plot_hypervolume measure HV(n) with. Those two run on a campaign's
    front - tens of points - where a pure-Python O(n^2 d) pass costs nothing. Refinement
    drives this front into the thousands, and there it does not finish.

    The two have to agree exactly, or HV* - HV(n) is a subtraction of numbers from
    different scales; tests/test_metrics.py asserts they do, on random fronts in 2-D and
    3-D. It is also the same class OptimizerBase._compute_hypervolume uses live, so a
    campaign's own reported hypervolume is on this scale too.

    Points not strictly better than the reference on every axis are dropped first, as
    hypervolume_nd drops them: the box such a point spans with the reference corner is
    empty, so it dominates nothing.
    """
    inside = front[(front < torch.tensor(ref, device=front.device, dtype=front.dtype))
                   .all(dim=-1)]
    if not inside.shape[0]:
        return 0.0
    return float(Hypervolume(ref_max).compute(-inside))


n_batches = -(-args.samples // args.batch)  # ceil
for _ in range(n_batches):
    want = min(args.batch, args.samples - drawn)
    if want <= 0:
        break
    X = sampler.draw_samples(n=want)
    # draw_samples returns what it found rather than what it was asked for, so the count is
    # read off the tensor. On a heavily constrained box the two differ, and reporting the
    # request would overstate how densely the space was actually covered.
    drawn += X.shape[0]
    X, P = evaluate(X)
    feasible += X.shape[0]
    merge(X, P)

    if front is None:
        history.append((drawn, None, None))
        continue
    hv = current_hv()
    gain = None if not previous else 100.0 * (hv - previous) / abs(previous)
    history.append((drawn, hv, gain))
    previous = hv

if front is None or previous is None:
    print(f"No feasible sample in {drawn} draws - nothing to measure a volume against.")
    sys.exit(1)

# ---- LOCAL REFINEMENT ----
# Sampling puts points where the box is, not where the front is. Once the front is known to
# within a few percent that is the wrong place to spend evaluations, and it is why a uniform
# draw loses to an optimizer - which is exactly the comparison HV* is used for, so losing it
# is what drives the regret negative. Each round scatters children around the current
# front's own parameters and halves the step, so the search walks onto the front instead of
# waiting to land on it. A rejected child costs nothing: it is dominated and drops out.
refined = []
if args.refine > 0:
    span = (objective.bounds[1] - objective.bounds[0]).to(front_X)
    lo, hi = objective.bounds[0].to(front_X), objective.bounds[1].to(front_X)
    scale = 0.05
    for round_number in range(args.refine):
        parents = front_X
        # A fixed budget per round rather than per parent, so a front of six points and one
        # of six hundred cost the same and the runtime stays predictable.
        repeats = max(1, args.batch // max(1, parents.shape[0]))
        children = parents.repeat(repeats, 1)
        children = torch.clamp(children + torch.randn_like(children) * span * scale, lo, hi)
        # The problem's own input constraints, applied to the children the way the sampler
        # applies them to a draw: a perturbation is free to step outside the feasible
        # region, and a child that did must not reach the front.
        children = children[objective.is_X_feasible(X=children)]
        if children.shape[0]:
            drawn += children.shape[0]
            X, P = evaluate(children)
            feasible += X.shape[0]
            merge(X, P)
        hv = current_hv()
        gain = None if not previous else 100.0 * (hv - previous) / abs(previous)
        refined.append((round_number + 1, scale, hv, gain))
        previous = hv
        scale /= 2.0

hv_star = previous

# ---- REPORT ----
senses = ", ".join(f"{k} (max)" if s < 0 else f"{k} (min)"
                   for k, s in zip(objective_keys, signs))
print(f"\nProblem: {args.ground_truth}")
print(f"Objectives: {senses}")
print(f"Reference point (minimization space): {ref}")
print(f"Sampled {drawn:,} points, {feasible:,} feasible, front of {front.shape[0]:,}.\n")

print(f"{'samples':>10}  {'HV*':>16}  {'gain':>10}")
for count, hv, gain in history:
    if hv is None:
        print(f"{count:>10,}  {'-':>16}  {'-':>10}")
    else:
        step = "-" if gain is None else f"{gain:+.3f}%"
        print(f"{count:>10,}  {hv:>16.6g}  {step:>10}")

if refined:
    print(f"\n{'refine':>10}  {'step':>16}  {'HV*':>16}  {'gain':>10}")
    for round_number, scale, hv, gain in refined:
        step = "-" if gain is None else f"{gain:+.3f}%"
        print(f"{round_number:>10}  {scale:>15.4%}  {hv:>16.6g}  {step:>10}")

trailing = (refined[-1][3] if refined else history[-1][2])
if trailing is not None and abs(trailing) > 0.1:
    print(f"\n! The last batch still moved HV* by {trailing:+.3f}%, so this is a lower "
          f"bound that has not settled. Raise --samples until the trailing gain is a "
          f"fraction of a percent; every regret measured against it is understated by "
          f"whatever is left.")

# The literature value, where the problem carries one. Free to check and worth checking: it
# is the only independent number this estimate can be held against. It is a cross-check
# rather than a target - the two are computed from different corners often enough that a
# gap says "find out which" rather than "this estimate is wrong".
if problem["max_hv"] is not None:
    declared_hv = float(problem["max_hv"])
    print(f"\nThe problem declares max_hv = {declared_hv:.6g}", end="")
    delta = (100.0 * (hv_star - declared_hv) / abs(declared_hv)) if declared_hv else None
    print(f" ({delta:+.3f}% from this estimate)." if delta is not None else ".")
    if len(objective_keys) != len(labels):
        print("  ! Declared over all of the problem's objectives, while this estimate "
              "covers only the ones asked for - the two are not the same volume.")
    elif delta is not None and abs(delta) > 5.0:
        # Sampling only ever undershoots, so a large gap either way is a sign the two
        # numbers are not measuring the same thing. A declared max_hv is usually quoted
        # against the reference point of whatever implementation it came from, and nothing
        # ties that to the ObjCfg.ref_point this was measured from - c2dtlz2 in this repo
        # declares ref_point 1.6 while its max_hv is the value at 1.1, a factor of four.
        print(f"  ! That is a wide gap. A declared max_hv carries no record of the corner "
              f"it was measured from, so check it was taken at ref_point {ref} and not "
              f"another; failing that, see the docstring on fronts that sampling cannot "
              f"reach.")

# ---- SAVE ----
# The same shape campaign_gain writes beside a run, and read back the same way: an estimate
# is only valid for the objectives, senses and reference it was computed under, so a
# consumer can tell a stale file from a current one instead of dividing by it regardless.
context = {"objectives": objective_keys,
           "signs": list(signs),
           "reference": list(ref),
           "reference_source": "ground-truth"}
payload = {"hv_star": hv_star,
           "ground_truth": os.path.abspath(args.ground_truth),
           "samples_requested": args.samples,
           "samples_drawn": drawn,
           "samples_feasible": feasible,
           "front_size": int(front.shape[0]),
           "declared_max_hv": problem["max_hv"],
           "convergence": [{"samples": c, "hv_star": h, "gain_percent": g}
                           for c, h, g in history],
           "refine_rounds": args.refine,
           "refinement": [{"round": r, "step_fraction": sc, "hv_star": h,
                           "gain_percent": g} for r, sc, h, g in refined],
           "context": context}

out_dir = args.out_dir or data_path
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "optimum.json")
with open(out_path, "w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2, allow_nan=False)
print(f"\nHV* = {hv_star:.6g}")
print("Saved", out_path)
