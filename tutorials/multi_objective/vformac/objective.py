from pathlib import Path

import torch
from torch import Tensor
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, TrkCfg, LinIneqXConCfg


class VFormAC(MCMultiObjectiveBase):
    """
    The ground truth is a polynomial fitted to the campaign's own 94 experiments,
    not an analytic function: build_polynomial_gt writes polynomial_gt.json and this
    loads it. The fit is the truth; the noise measured from the repeated settings is
    added on top by gt_obj_noise_std rather than baked into the fit.

    Regenerate the file after changing the data or the degree:
        python -m ground_truth.build_polynomial_gt --root-dir data/vformac \
            --degree 2 --out tutorials/multi_objective/vformac/polynomial_gt.json

    Unlike the Gaussian process this replaces, a polynomial keeps extrapolating
    where nothing was measured - V0 was only ever run over [60, 90] of its
    [60, 120]. Over that third of the box the surface is the polynomial's guess,
    not evidence.

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
    _GT_FILE = "polynomial_gt.json"

    def __init__(self, device: torch.device, dtype: torch.dtype,
                 gt_file: str | Path = None):
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
            # only: it is what evaluate_*_with_noise adds to the fit, never a variance
            # recorded against an observation.
            gt_obj_noise_std=[3.187, 9.549],
            gt_trk_noise_std=[1.064],
        )

        # Anchored on __file__ rather than the working directory, so the campaign
        # runs from anywhere. Loaded after super().__init__, which is what defines
        # the device and dtype the coefficients land on.
        gt_file = gt_file or Path(__file__).resolve().parent / self._GT_FILE
        self._obj_fit = self._load_polynomial_gt(gt_file, "objectives")
        self._trk_fit = self._load_polynomial_gt(gt_file, "trackers")

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return self._evaluate_polynomial_gt(self._obj_fit, X)

    def evaluate_tracker(self, X: Tensor) -> Tensor:
        return self._evaluate_polynomial_gt(self._trk_fit, X)
