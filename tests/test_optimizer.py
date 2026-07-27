import torch
import pytest
from pybo.optimizer.optimizer import BayesianOptimizer
from pybo.objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg

DEVICE = torch.device("cpu")
DTYPE = torch.float64


class SingleMin(MCSingleObjectiveBase):
    def __init__(self):
        super().__init__(
            device=DEVICE,
            dtype=DTYPE,
            par_cfg=[ParCfg(bounds=(-1.0, 5.0))],
            obj_cfg=[ObjCfg(bounds=(0.0, 9.0), to_minimize=True, ref_point=10.0)],
        )

    def evaluate_true_objective(self, X):
        return (X - 2).pow(2)


class TwoMin(MCMultiObjectiveBase):
    def __init__(self):
        super().__init__(
            device=DEVICE,
            dtype=DTYPE,
            par_cfg=[ParCfg(bounds=(0.0, 1.0)), ParCfg(bounds=(0.0, 1.0))],
            obj_cfg=[
                ObjCfg(label="a", bounds=(0.0, 10.0), to_minimize=True, ref_point=6.0),
                ObjCfg(label="b", bounds=(0.0, 10.0), to_minimize=True, ref_point=6.0),
            ],
        )

    def evaluate_true_objective(self, X):
        return X.clone()


class NeverFeasible(MCSingleObjectiveBase):
    def __init__(self):
        super().__init__(
            device=DEVICE,
            dtype=DTYPE,
            par_cfg=[ParCfg(bounds=(-1.0, 5.0))],
            obj_cfg=[ObjCfg(bounds=(0.0, 9.0), to_minimize=True, ref_point=10.0)],
            # A constraint is satisfied when it evaluates <= 0, so a constant +1
            # makes every observation infeasible.
            ineq_Y_con_cfg=[IneqYConCfg(f=lambda Y: torch.ones_like(Y[..., :1]))],
        )

    def evaluate_true_objective(self, X):
        return (X - 2).pow(2)


def test_feasible_mask_all_feasible_without_constraints():
    obj = SingleMin()
    X = torch.tensor([[0.0], [2.0], [4.0]], dtype=DTYPE)
    Y = obj.evaluate_true_objective(X)
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y)
    bo._compute_feasible_mask(verbose=False)
    assert bo.feasible_mask.tolist() == [True, True, True]


def test_best_f_picks_feasible_minimum():
    obj = SingleMin()
    X = torch.tensor([[0.0], [2.0], [4.0]], dtype=DTYPE)
    Y = torch.tensor([[4.0], [1.0], [9.0]], dtype=DTYPE)  # minimum at index 1
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y)
    bo._compute_feasible_mask(verbose=False)
    bo._compute_acquisition_function_reference(verbose=False)
    assert bo.best_feasible_Y.item() == 1.0
    assert bo.best_feasible_X.tolist() == [2.0]
    # best_f is stored in maximization space, i.e. negated for a minimization objective
    assert bo.best_f.item() == -1.0


def test_reference_point_negated_to_max_space():
    obj = TwoMin()
    X = torch.rand(5, 2, dtype=DTYPE)
    Y = obj.evaluate_true_objective(X)
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y)
    bo._compute_feasible_mask(verbose=False)
    bo._compute_acquisition_function_reference(verbose=False)
    assert bo.ref_point.tolist() == [-6.0, -6.0]


def test_pareto_front_excludes_dominated_point():
    obj = TwoMin()
    X = torch.tensor([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]], dtype=DTYPE)
    # Minimization objectives: [5,5] is dominated by [2,2]; the other three are not dominated.
    Y = torch.tensor([[1.0, 4.0], [4.0, 1.0], [2.0, 2.0], [5.0, 5.0]], dtype=DTYPE)
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y)
    bo._compute_feasible_mask(verbose=False)
    bo._compute_acquisition_function_reference(verbose=False)
    # The Pareto front is computed as part of the metrics stage, not the reference stage
    bo._compute_metrics(verbose=False)
    front = bo.feasible_pareto_front_Y
    assert front.shape[0] == 3
    assert [5.0, 5.0] not in front.tolist()


def test_Y_obj_setter_validates_objective_count():
    obj = SingleMin()
    X = torch.tensor([[0.0], [2.0]], dtype=DTYPE)
    Y = torch.tensor([[4.0], [1.0]], dtype=DTYPE)
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y)
    # Correct shape (last dim == num_objectives) is accepted via the public setter
    bo.Y_obj = torch.tensor([[3.0], [2.0]], dtype=DTYPE)
    assert bo.Y_obj.shape[-1] == 1
    # Wrong number of objective columns is rejected
    with pytest.raises(ValueError):
        bo.Y_obj = torch.tensor([[3.0, 3.0], [2.0, 2.0]], dtype=DTYPE)


def test_n_model_fit_restarts_rejects_zero():
    # It is passed straight to BoTorch as max_attempts, whose loop is
    # range(1, 1 + max_attempts), so anything below 1 never fits the model at all
    # and surfaces as a bare ModelFittingError instead of a misconfiguration.
    # Note this guards the setter only: __init__ assigns _n_model_fit_restarts
    # directly, so a bad value passed to the constructor still slips through.
    obj = SingleMin()
    X = torch.tensor([[0.0], [2.0]], dtype=DTYPE)
    Y = torch.tensor([[4.0], [1.0]], dtype=DTYPE)

    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y)
    with pytest.raises(ValueError):
        bo.n_model_fit_restarts = 0
    with pytest.raises(ValueError):
        bo.n_model_fit_restarts = -3
    # A single attempt is a legitimate setting
    bo.n_model_fit_restarts = 1
    assert bo.n_model_fit_restarts == 1


def test_metrics_logged_once_per_iteration_when_nothing_is_feasible():
    # Regression: _compute_best_f used to append None to _best_values, which
    # _compute_metrics already owns. Infeasible iterations logged twice, so the
    # history desynced and is_converged raised TypeError on the None entries.
    obj = NeverFeasible()
    X = torch.tensor([[0.0], [2.0], [4.0]], dtype=DTYPE)
    Y = obj.evaluate_true_objective(X)
    Y_con = torch.zeros(3, 1, dtype=DTYPE)
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y, Y_con=Y_con)
    bo._compute_feasible_mask(verbose=False)
    assert not bo.feasible_mask.any()

    for _ in range(12):
        bo._compute_acquisition_function_reference(verbose=False)
        bo._compute_metrics(verbose=False)

    assert len(bo.best_values) == 12
    assert all(v is not None for v in bo.best_values)
    # is_converged differences consecutive entries, so a None would raise here
    assert bo.is_converged(patience=10, verbose=False) is False


def test_ic_generator_without_feasible_points():
    # Regression: _ic_generator cloned the result of _compute_feasible_XY before
    # checking it, so the no-feasible-points branch died on None.clone().
    obj = NeverFeasible()
    X = torch.tensor([[0.0], [2.0], [4.0]], dtype=DTYPE)
    Y = obj.evaluate_true_objective(X)
    Y_con = torch.zeros(3, 1, dtype=DTYPE)
    bo = BayesianOptimizer(device=DEVICE, dtype=DTYPE, objective=obj, X=X, Y_obj=Y, Y_con=Y_con)
    bo._compute_feasible_mask(verbose=False)
    assert not bo.feasible_mask.any()

    ic = bo._ic_generator(acq_function=None, q=1, num_restarts=5, raw_samples=64)
    # (num_restarts, q, d), and the dtype must match the samples it gets concatenated with
    assert ic.shape == (5, 1, 1)
    assert ic.dtype == DTYPE
