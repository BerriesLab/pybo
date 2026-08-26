"""The hypervolume the campaign metrics are built on, and the problem's view of a record.

The regret and the normalized hypervolume are both HV* set against HV(n), and the two are
computed by different code: HV* by campaign_optimum through botorch's Hypervolume, HV(n) by
campaign_gain and plot_hypervolume through this package's own torch-free hypervolume_nd.
Subtracting one from the other is only meaningful if they agree exactly, so that is
asserted here rather than argued for in a comment.
"""
import math

import pytest
import torch
from botorch.utils.multi_objective import Hypervolume, is_non_dominated

from pybo_gui.modules.bayesian_campaign_analysis._hypervolume import (
    hypervolume_nd, pareto_front_nd,
)

DEVICE = torch.device("cpu")
DTYPE = torch.float64


# ---- pareto_front_nd ----

def test_front_keeps_only_non_dominated():
    # (2, 2) is dominated by (1, 1); the other two trade off against each other.
    points = [(1.0, 1.0), (2.0, 2.0), (0.0, 5.0), (5.0, 0.0)]
    assert sorted(pareto_front_nd(points)) == [(0.0, 5.0), (1.0, 1.0), (5.0, 0.0)]


def test_front_keeps_duplicates_of_a_non_dominated_point():
    # Neither copy dominates the other (domination needs a strict improvement somewhere),
    # so both survive. hypervolume_nd counts the volume once either way.
    assert pareto_front_nd([(1.0, 1.0), (1.0, 1.0)]) == [(1.0, 1.0), (1.0, 1.0)]


# ---- hypervolume_nd, against volumes that can be worked out by hand ----

def test_single_point_is_a_box():
    assert hypervolume_nd([(2.0, 3.0)], (10.0, 8.0)) == pytest.approx(8.0 * 5.0)


def test_three_dimensions():
    assert hypervolume_nd([(0.0, 0.0, 0.0)], (1.0, 2.0, 4.0)) == pytest.approx(8.0)


def test_two_points_union_not_sum():
    # [1,10]x[4,10] and [4,10]x[1,10] overlap in [4,10]x[4,10]: 54 + 54 - 36.
    hv = hypervolume_nd([(1.0, 4.0), (4.0, 1.0)], (10.0, 10.0))
    assert hv == pytest.approx(54.0 + 54.0 - 36.0)


def test_dominated_point_adds_nothing():
    ref = (10.0, 10.0)
    assert (hypervolume_nd([(1.0, 1.0), (2.0, 2.0)], ref)
            == pytest.approx(hypervolume_nd([(1.0, 1.0)], ref)))


def test_empty_is_zero():
    assert hypervolume_nd([], (1.0, 1.0)) == 0.0


# ---- the reference box ----

def test_point_outside_the_reference_box_contributes_nothing():
    """Regression: a point beyond the reference on one axis used to stretch its
    neighbour's slab out past the reference and inflate the total - here from 32 to 88.

    It only ever arises with a reference the problem declared. A corner derived from the
    observations is padded past the worst of them, so nothing can fall outside it, which
    is why this went unseen until HV* gave the two references something to disagree about.
    """
    ref = (18.0, 6.0)
    alone = hypervolume_nd([(10.0, 2.0)], ref)
    assert alone == pytest.approx(8.0 * 4.0)
    # (0.4, 13) beats the reference on the first axis and loses on the second, so the box
    # it spans with the reference corner is empty.
    assert hypervolume_nd([(10.0, 2.0), (0.4, 13.0)], ref) == pytest.approx(alone)


def test_a_point_on_the_reference_contributes_nothing():
    # Zero thickness on one axis is still zero volume, so the boundary belongs outside.
    assert hypervolume_nd([(5.0, 6.0)], (10.0, 6.0)) == 0.0


def test_every_point_outside_gives_zero():
    assert hypervolume_nd([(20.0, 20.0), (30.0, 1.0)], (10.0, 10.0)) == 0.0


# ---- the claim campaign_optimum rests on ----

def _botorch_hv(front_min, ref_min):
    """The same volume via botorch, which measures in maximization space.

    This is the class OptimizerBase._compute_hypervolume uses live and the one
    campaign_optimum computes HV* with, so agreement here is what puts a campaign's
    hypervolume and its optimum on one scale.
    """
    Y = torch.tensor(front_min, dtype=DTYPE)
    ref = torch.tensor(ref_min, dtype=DTYPE)
    inside = Y[(Y < ref).all(dim=-1)]
    if not inside.shape[0]:
        return 0.0
    return float(Hypervolume(-ref).compute(-inside))


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_agrees_with_botorch_on_random_fronts(dimensions, seed):
    torch.manual_seed(seed)
    Y = torch.rand(60, dimensions, dtype=DTYPE)
    front = Y[is_non_dominated(-Y)]
    points = [tuple(float(v) for v in row) for row in front]
    ref = tuple([0.8] * dimensions)   # inside the cloud, so some points fall outside it
    assert hypervolume_nd(points, ref) == pytest.approx(_botorch_hv(points, ref), rel=1e-9)


def test_agrees_with_botorch_on_a_smooth_front():
    # A quarter circle: the shape a real two-objective front takes, where every point is
    # non-dominated and the slicing recursion does its most work.
    points = [(math.cos(t), math.sin(t)) for t in torch.linspace(0, math.pi / 2, 50)]
    ref = (1.5, 1.5)
    assert hypervolume_nd(points, ref) == pytest.approx(_botorch_hv(points, ref), rel=1e-9)


# ---- the problem's view of a record ----

@pytest.fixture
def objective_file(tmp_path):
    """A two-parameter problem on [0, 1] with a + b <= 1 carving the box, written out as an
    importable objective.py.

    _problem_view loads a problem from a path, the way the GUI hands it one, so the test has
    to give it a file rather than an instance. Its noisy branch adds 100 rather than a small
    perturbation, so a test asserting the noiseless value cannot pass by rounding.
    """
    path = tmp_path / "objective.py"
    path.write_text(
        "import torch\n"
        "from pybo.objectives.base_class import MCMultiObjectiveBase\n"
        "from pybo.objectives.variable_registry import ParCfg, ObjCfg, LinIneqXConCfg\n"
        + "".join(line + "\n" for line in [
            "",
            "",
            "class Problem(MCMultiObjectiveBase):",
            "    def __init__(self, device, dtype):",
            "        super().__init__(",
            "            device=device, dtype=dtype,",
            "            par_cfg=[ParCfg(label='a', bounds=(0.0, 1.0)),",
            "                     ParCfg(label='b', bounds=(0.0, 1.0))],",
            "            obj_cfg=[ObjCfg(label='f0', bounds=(0.0, 2.0), to_minimize=True,"
            " ref_point=2.0),",
            "                     ObjCfg(label='f1', bounds=(0.0, 2.0), to_minimize=True,"
            " ref_point=2.0)],",
            "            lin_ineq_X_con_cfg=[LinIneqXConCfg(idxs=[0, 1],"
            " coeff=[-1.0, -1.0], rhs=-1.0)],",
            "        )",
            "",
            "    def evaluate_true_objective(self, X, noisy: bool = False):",
            "        f = torch.stack([X[..., 0], X[..., 1]], dim=-1)",
            "        return f + 100.0 if noisy else f",
        ]), encoding="utf-8")
    return str(path)


def test_true_results_reads_the_noiseless_surface(objective_file):
    from pybo_gui.modules.bayesian_campaign_analysis._problem_view import true_results
    out = true_results(objective_file, [{"a": 0.25, "b": 0.75}], ["f0", "f1"])
    # Noiseless, so the +100 the noisy branch adds must not appear.
    assert out[0]["f0"] == pytest.approx(0.25)
    assert out[0]["f1"] == pytest.approx(0.75)


def test_true_results_skips_a_row_with_no_parameters(objective_file):
    from pybo_gui.modules.bayesian_campaign_analysis._problem_view import true_results
    rows = [{}, {"a": 0.1, "b": 0.2}, {"a": 0.5}]
    out = true_results(objective_file, rows, ["f0"])
    # None, not a value: a reference measurement records no parameters, and inventing one
    # for it would put a point on the front that no run ever evaluated.
    assert out[0] is None and out[2] is None
    assert out[1]["f0"] == pytest.approx(0.1)


def test_input_feasible_marks_the_points_the_problem_forbids(objective_file):
    """The constraint is coeff . x >= rhs, so coeff=[-1,-1] with rhs=-1 reads a + b <= 1
    (the spelling vformac uses for V0 + dV <= 150). Asserted as literal expectations
    rather than against is_X_feasible, which is the thing under test here."""
    from pybo_gui.modules.bayesian_campaign_analysis._problem_view import input_feasible
    rows = [{"a": 0.2, "b": 0.3},   # a + b = 0.5, inside
            {"a": 0.8, "b": 0.9},   # a + b = 1.7, outside
            {"a": 0.5, "b": 0.5},   # a + b = 1.0, exactly on the boundary, which is inside
            {}]
    assert input_feasible(objective_file, rows)[:3] == [True, False, True]
    # None rather than False for the last: "cannot be checked" and "breaks the constraints"
    # call for opposite treatment, and conflating them deletes every reference row from a
    # campaign that has any.
    assert input_feasible(objective_file, rows)[3] is None


# ---- when a run stopped improving ----

from pybo_gui.modules.bayesian_campaign_analysis._convergence import terminal_plateau


def test_plateau_is_found_at_its_start():
    # Climbs to 10 by index 4, then eight flat steps. The plateau begins at 4.
    metric = [0, 3, 6, 9, 10] + [10] * 8
    assert terminal_plateau(metric, eps=0.5, patience=4, n_initial=0) == 4


def test_a_run_still_improving_never_converged():
    # No flat stretch at the end at all - the budget ran out mid-climb.
    metric = list(range(20))
    assert terminal_plateau(metric, eps=0.5, patience=4, n_initial=0) is None


def test_a_late_climb_beats_an_early_plateau():
    """The reason the search runs backwards.

    Six flat steps, then a climb, then six more flat steps. Forwards, the first stretch
    won and everything after it vanished from the score; backwards, the plateau the run
    actually finished on is the one reported.
    """
    metric = [5] * 7 + [6, 7, 8] + [8] * 6
    assert terminal_plateau(metric, eps=0.5, patience=4, n_initial=0) == 9


def test_a_plateau_shorter_than_patience_does_not_count():
    metric = [0, 1, 2, 3, 4, 5, 5, 5]      # only two flat steps at the end
    assert terminal_plateau(metric, eps=0.5, patience=4, n_initial=0) is None
    assert terminal_plateau(metric, eps=0.5, patience=2, n_initial=0) == 5


def test_a_plateau_running_into_the_initial_design_does_not_count():
    """A design is drawn blind, so its metric is routinely flat for several points
    together. Reporting convergence there would say the run stopped improving before the
    optimizer had proposed anything."""
    metric = [4] * 12                       # flat throughout, design included
    assert terminal_plateau(metric, eps=0.5, patience=4, n_initial=5) is None


def test_eps_decides_what_counts_as_flat():
    metric = [0, 1, 2, 3] + [3.0, 3.2, 3.4, 3.6, 3.8]
    # Steps of 0.2: flat under a threshold of 0.5, still climbing under one of 0.1.
    assert terminal_plateau(metric, eps=0.5, patience=4, n_initial=0) == 3
    assert terminal_plateau(metric, eps=0.1, patience=4, n_initial=0) is None


def test_too_short_to_judge():
    assert terminal_plateau([], eps=1.0, patience=2, n_initial=0) is None
    assert terminal_plateau([1.0], eps=1.0, patience=2, n_initial=0) is None


def test_patience_must_be_positive():
    with pytest.raises(ValueError):
        terminal_plateau([1, 1, 1], eps=1.0, patience=0, n_initial=0)


# ---- what the selected runs get grouped by ----

from pybo_gui.modules.bayesian_campaign_analysis._series import (
    GROUP_KEYS, GroupKeyError, group_key, merge_key, parse_keys, pooled, series_key,
    series_label,
)


def record(run="bayesian_ninit10_replicate0_seed2063", optimizer="bayesian", n_initial=10,
           iteration="step_003_rep00", observation=0, **parameters):
    """One map record, with only the fields the grouping reads."""
    return {"run": run, "optimizer": optimizer, "n_initial": n_initial,
            "provenance": "synthetic", "technology": None,
            "iteration": iteration, "observation": observation,
            "parameters": parameters or {"V0": 80.0, "dV": 68.0}}


def test_a_setting_measured_twice_in_one_run_is_one_measurement():
    """Repeats always merge - they are one setting measured twice, and their spread is
    what the error bar is for. Different steps landing on one setting count too, which is
    what a converged run re-proposing its own optimum does."""
    a = record(iteration="step_003_rep00")
    b = record(iteration="step_040_rep00")
    assert merge_key(a, GROUP_KEYS) == merge_key(b, GROUP_KEYS)


def test_merging_never_crosses_runs():
    """Two runs measuring one setting are two measurements of it, not a repeat. Pooling
    them is a separate question, answered by leaving `run` out of the keys - and answered
    by averaging their curves, which needs the runs still to be there."""
    a = record(run="replicate0")
    b = record(run="replicate1")
    assert merge_key(a, GROUP_KEYS) != merge_key(b, GROUP_KEYS)
    without_run = [k for k in GROUP_KEYS if k != "run"]
    assert merge_key(a, without_run) != merge_key(b, without_run)


def test_a_different_setting_is_a_different_group():
    a = record(V0=80.0, dV=68.0)
    b = record(V0=81.0, dV=68.0)
    assert merge_key(a, GROUP_KEYS) != merge_key(b, GROUP_KEYS)


def test_parameters_snap_onto_the_rig_grid():
    """A repeat is routinely recorded once as the rounded setpoint and once as the
    proposal behind it, so the grid is what makes them one setting."""
    asked = record(V0=80.4, dV=68.0)
    set_to = record(V0=80.0, dV=68.0)
    assert merge_key(asked, GROUP_KEYS) != merge_key(set_to, GROUP_KEYS)
    assert (merge_key(asked, GROUP_KEYS, {"V0": 1.0})
            == merge_key(set_to, GROUP_KEYS, {"V0": 1.0}))


def test_series_key_ignores_the_setting():
    """A trace plot builds one curve per run, so what tells two curves apart cannot
    depend on which setting a record was taken at."""
    a = record(V0=80.0)
    b = record(V0=95.0)
    assert series_key(a, GROUP_KEYS) == series_key(b, GROUP_KEYS)


def test_series_key_separates_and_pools_runs_on_demand():
    a, b = record(run="replicate0"), record(run="replicate1")
    assert series_key(a, GROUP_KEYS) != series_key(b, GROUP_KEYS)
    pooled_keys = [k for k in GROUP_KEYS if k != "run"]
    assert series_key(a, pooled_keys) == series_key(b, pooled_keys)


def test_dropping_n_initial_pools_the_design_sizes():
    """Nine series (3 strategies x 3 sizes) become three."""
    keys = [k for k in GROUP_KEYS if k not in ("run", "n_initial")]
    sizes = [series_key(record(run=f"r{n}", n_initial=n), keys) for n in (10, 15, 20)]
    assert len(set(sizes)) == 1
    assert series_key(record(optimizer="sobol"), keys) != sizes[0]


def test_a_series_is_named_by_what_still_tells_it_apart():
    assert series_label(record(), GROUP_KEYS) == "bayesian_ninit10_replicate0_seed2063"
    no_run = [k for k in GROUP_KEYS if k != "run"]
    assert series_label(record(), no_run) == "bayesian n10 synthetic"
    no_size = [k for k in no_run if k != "n_initial"]
    assert series_label(record(), no_size) == "bayesian synthetic"
    assert series_label({"run": None}, (), fallback="unnamed") == "unnamed"


def test_parse_keys_preserves_order_dedupes_and_rejects_nonsense():
    # Order is the caller's own choice now, not normalised away - it decides
    # series_label's part order, and the GUI's reorderable key list depends on it
    # actually reaching the script. Grouping itself is unaffected either way: which
    # records pool together is a plain equality test on the chosen keys.
    assert parse_keys(["run", "parameters"]) == ("run", "parameters")
    assert parse_keys(["parameters", "run"]) == ("parameters", "run")
    # A repeated key collapses to its first occurrence rather than erroring.
    assert parse_keys(["run", "run", "parameters"]) == ("run", "parameters")
    with pytest.raises(GroupKeyError):
        parse_keys(["strategy", "colour"])
    # The key that used to exist and no longer does, so a stale command says so.
    with pytest.raises(GroupKeyError):
        parse_keys(["repeat"])


def test_pooled_names_what_the_band_swept_up():
    assert pooled(GROUP_KEYS) == ()
    assert pooled([k for k in GROUP_KEYS if k != "run"]) == ("run",)
