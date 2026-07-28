import torch
from pybo.constraints.output_constraints import *
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import *


class SparkAccelerator(MCMultiObjectiveBase):
    """
    2x objectives:
        - Machining Time: to minimize.
        - Electrode Wear: to minimize.

    1x ineq_Y_con_cfg:
        - Orbiting Time: Requires values > 40 mins.

    3x parameters:
        - V0: The initial voltage.
        - dV: Twice the voltage step, i.e. each step rises the voltage by dV/2.
        - td1: Delay Time 1 in us
        - td2: Delay Time 2 in us

    Input constraints:
        - V0 + dV <= 150
        - td1 + td2 <= 1.8 * t_r
        - td1 + td2 >= 1.2 * t_r

    Output constraints:
        - orbiting_time <= orbiting_target + delta
        - orbiting_time >= orbiting_target - delta

    Reference point:
        - Machining Time: 300 min
        - Tool Wear: 150 um
    """

    _c = 20
    _t_d100 = 54  # us
    _t_r = _t_d100 * (1 - _c / 50)
    _orbiting_target = 40
    _delta = _orbiting_target * 0.05  # Plus minus 5 %

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
                ObjCfg(label="Machining Time", to_minimize=True, ref_point=300.0),
                ObjCfg(label="Tool Wear", to_minimize=True, ref_point=150.0),
            ],
            trk_cfg=[
                TrkCfg(label="Orbiting Time Deviation", bounds=(-20, 20)),
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1, -1], rhs=-150),
                LinIneqXConCfg(idxs=[2, 3], coeff=[-1, -1], rhs=-1.8 * self._t_r),
                LinIneqXConCfg(idxs=[2, 3], coeff=[1, 1], rhs=1.2 * self._t_r)
            ],
            ineq_Y_con_cfg=[
                IneqYConCfg(f=LowerBound(threshold=self._orbiting_target - self._delta, index=-1)),
                IneqYConCfg(f=UpperBound(threshold=self._orbiting_target + self._delta, index=-1))
            ]
        )
