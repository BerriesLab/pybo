import torch
import pytest
from pybo.objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg

DEVICE = torch.device("cpu")
DTYPE = torch.float64


class SingleMin(MCSingleObjectiveBase):
    # Carries its own ground-truth noise, as an objective standing in for a rig does.
    _NOISE_STD = 0.5

    def __init__(self):
        super().__init__(
            device=DEVICE,
            dtype=DTYPE,
            par_cfg=[ParCfg(bounds=(-1.0, 5.0))],
            obj_cfg=[ObjCfg(bounds=(0.0, 9.0), to_minimize=True, ref_point=10.0)],
        )

    def evaluate_true_objective(self, X, noisy: bool = False):
        Y = (X - 2).pow(2)
        if noisy:
            Y = Y + self._NOISE_STD * torch.randn_like(Y)
        return Y


class MixedMulti(MCMultiObjectiveBase):
    # obj 0 minimized, obj 1 maximized
    def __init__(self):
        super().__init__(
            device=DEVICE,
            dtype=DTYPE,
            par_cfg=[ParCfg(bounds=(0.0, 1.0)), ParCfg(bounds=(0.0, 1.0))],
            obj_cfg=[
                ObjCfg(label="a", bounds=(0.0, 1.0), to_minimize=True, ref_point=2.0),
                ObjCfg(label="b", bounds=(0.0, 1.0), to_minimize=False, ref_point=-1.0),
            ],
        )

    def evaluate_true_objective(self, X, noisy: bool = False):
        if noisy:
            raise ValueError(f"{type(self).__name__} declares no ground-truth noise.")
        return X.clone()


class YConstrained(MCSingleObjectiveBase):
    # Feasible when the constraint column of Y is <= 1
    def __init__(self):
        super().__init__(
            device=DEVICE,
            dtype=DTYPE,
            par_cfg=[ParCfg(bounds=(0.0, 1.0))],
            obj_cfg=[ObjCfg(bounds=(0.0, 1.0), to_minimize=True, ref_point=2.0)],
            ineq_Y_con_cfg=[IneqYConCfg(f=lambda Y: Y[..., 1:2] - 1.0)],
        )

    def evaluate_true_objective(self, X, noisy: bool = False):
        if noisy:
            raise ValueError(f"{type(self).__name__} declares no ground-truth noise.")
        return X.clone()


def test_single_forward_negates_minimize():
    obj = SingleMin()
    out = obj.forward(torch.tensor([[3.0]], dtype=DTYPE))
    assert torch.equal(out, torch.tensor([-3.0], dtype=DTYPE))


def test_multi_forward_negates_only_minimized_column():
    obj = MixedMulti()
    out = obj.forward(torch.tensor([[0.4, 0.7]], dtype=DTYPE))
    assert torch.allclose(out, torch.tensor([[-0.4, 0.7]], dtype=DTYPE))


def test_derived_attributes():
    obj = SingleMin()
    assert obj.dim == 1
    assert obj.num_obj == 1
    assert obj.num_objectives == obj.num_obj
    assert obj.to_minimize.tolist() == [True]
    assert torch.equal(obj.bounds, torch.tensor([[-1.0], [5.0]], dtype=DTYPE))


def test_num_objectives_matches_num_obj_multi():
    obj = MixedMulti()
    assert obj.num_objectives == 2


def test_noisy_flag_perturbs_without_shifting_the_objective():
    obj = SingleMin()
    X = torch.full((4096, 1), 3.0, dtype=DTYPE)
    clean = obj.evaluate_true_objective(X)
    # Clean stays exact and repeatable; noisy scatters around it without moving it.
    assert torch.equal(clean, obj.evaluate_true_objective(X))
    torch.manual_seed(0)
    noisy = obj.evaluate_true_objective(X, noisy=True)
    assert noisy.shape == clean.shape
    assert not torch.equal(noisy, clean)
    assert torch.allclose(noisy.std(), torch.tensor(0.5, dtype=DTYPE), atol=0.05)
    assert torch.allclose(noisy.mean(), clean.mean(), atol=0.05)


def test_objective_without_noise_refuses_to_invent_it():
    with pytest.raises(ValueError, match="no ground-truth noise"):
        MixedMulti().evaluate_true_objective(
            torch.zeros((1, 2), dtype=DTYPE), noisy=True)


def test_ref_point_negation_to_max_space():
    obj = MixedMulti()
    assert obj.ref_point.tolist() == [2.0, -1.0]


def test_x_feasibility_respects_bounds():
    obj = SingleMin()
    mask = obj.is_X_feasible(torch.tensor([[0.0], [6.0], [-2.0]], dtype=DTYPE))
    assert mask.tolist() == [True, False, False]


def test_y_feasibility_output_constraint():
    obj = YConstrained()
    Y = torch.tensor([[0.5, 0.5], [0.5, 2.0]], dtype=DTYPE)
    assert obj.is_Y_feasible(Y).tolist() == [True, False]
