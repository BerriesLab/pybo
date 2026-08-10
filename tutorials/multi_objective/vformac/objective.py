from pathlib import Path

import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from torch import Tensor

from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, TrkCfg, LinIneqXConCfg


class VFormAC(MCMultiObjectiveBase):
    """
    The ground truth is a Gaussian process fitted to the campaign's own 94 experiments,
    not an analytic function: generate_ground_truth.py writes gp_ground_truth.pt and
    this loads it. The posterior mean is the truth; the noise measured from the repeated
    settings is added on top by gt_obj_noise_std rather than baked into the mean.

    Over roughly two thirds of the declared box - V0 was only ever run over [60, 90] of
    its [60, 120] - the posterior mean reverts to the prior and the surface is flat at the
    data mean. That is what a GP does where nothing was measured, and it is accepted here
    rather than hidden by extrapolating a curve nothing supports.

    2x objectives:
        - Machining Time: to minimize.
        - Tool Wear: to minimize.

    1x tracker:
        - Orbiting Time: measured and recorded.

    4x parameters:
        - V0: The initial voltage.
        - dV: Twice the voltage step, i.e. each step rises the voltage by dV/2.
        - td1: Delay Time 1 in ns
        - td2: Delay Time 2 in ns

    Input constraints:
        - V0 + dV <= 150
        - td1 + td2 <= 1.8 * t_r
        - td1 + td2 >= 1.2 * t_r

    Reference point:
        - Machining Time: 200 min
        - Tool Wear: 150 um
    """

    _c = 20
    _t_d100 = 54 * 1000  # ns
    _t_r = _t_d100 * (1 - _c / 50)
    _MODEL_FILE = "gp_ground_truth.pt"

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="V0", bounds=(60, 120)),
                ParCfg(label="dV", bounds=(60, 85)),
                ParCfg(label="td1", bounds=(0.5 * self._t_r, 0.8 * self._t_r)),
                ParCfg(label="td2", bounds=(0.4 * self._t_r, 1.3 * self._t_r)),
            ],
            obj_cfg=[
                ObjCfg(label="Machining Time", to_minimize=True, ref_point=200.0),
                ObjCfg(label="Tool Wear", to_minimize=True, ref_point=150.0),
            ],
            trk_cfg=[
                TrkCfg(label="Orbiting Time"),
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1, -1], rhs=-150),
                LinIneqXConCfg(idxs=[2, 3], coeff=[-1, -1], rhs=-1.8 * self._t_r),
                LinIneqXConCfg(idxs=[2, 3], coeff=[1, 1], rhs=1.2 * self._t_r)
            ],
            # Measured from the 8 repeated settings, 14 degrees of freedom. Simulation
            # only: it is what evaluate_*_with_noise adds to the mean, never a variance
            # recorded against an observation.
            gt_obj_noise_std=[3.187, 9.549],
            gt_trk_noise_std=[1.064],
        )

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return self._posterior_mean(X, range(self.num_objectives))

    def evaluate_tracker(self, X: Tensor) -> Tensor:
        return self._posterior_mean(
            X, range(self.num_objectives, self.num_objectives + self.num_trk))
