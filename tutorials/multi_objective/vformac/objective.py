from pathlib import Path

import torch
from torch import Tensor
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, TrkCfg, LinIneqXConCfg


class VFormAC(MCMultiObjectiveBase):
    """
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
                ParCfg(label="V0", unit="V", bounds=(60, 120)),
                ParCfg(label="dV", unit="V", bounds=(60, 85)),
                ParCfg(label="td1", unit="ns", bounds=(0.5 * self._t_r, 0.8 * self._t_r)),
                ParCfg(label="td2", unit="ns", bounds=(0.4 * self._t_r, 1.3 * self._t_r)),
            ],
            obj_cfg=[
                ObjCfg(label="Material Removal Rate", unit="mm^3/min", to_minimize=False, ref_point=0),
                ObjCfg(label="Tool Wear", unit="um", to_minimize=True, ref_point=150.0),
            ],
            trk_cfg=[
                TrkCfg(label="Orbiting Time", unit="min"),
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1, -1], rhs=-150),
                LinIneqXConCfg(idxs=[2, 3], coeff=[-1, -1], rhs=-1.8 * self._t_r),
                LinIneqXConCfg(idxs=[2, 3], coeff=[1, 1], rhs=1.2 * self._t_r)
            ],
            # Pooled within-setting std over the 8 settings measured twice, so 8 degrees
            # of freedom. In obj_cfg order, which is what makes 0.0385 the removal rate's
            # and 5.274 the wear's - the previous pair was indexed against a machining
            # time this problem no longer has, and put a std of 3.187 on a removal rate
            # whose whole range is 1.98 to 2.84. Simulation only: it is what
            # evaluate_*_with_noise adds to the fit, never a variance recorded against an
            # observation.
            gt_obj_noise_std=[0.0385, 5.274],
            gt_trk_noise_std=[1.131],
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
