import torch
from constraints.output_constraints import Identity, UpperBound, LowerBound
from objectives.base_class import MCMultiObjectiveBase
from objectives.variable_registry import ParCfg, ObjCfg, LinIneqXConCfg, TrkCfg, IneqYConCfg


class SparkAcceleratorConstrained(MCMultiObjectiveBase):
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
        - td1 + td2 <= td100

    Reference point:
        - Machining Time: 300 min
        - Tool Wear: 150 um
    """

    _c = 20
    _t_d100 = 54  # us
    _t_r = _t_d100 * (1 - _c / 50)
    _orbiting_target = 40
    _delta = 40 * 0.02

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="V0", bounds=(60, 120)),
                ParCfg(label="dV", bounds=(60, 85)),
                ParCfg(label="td1", bounds=(0.5 * self._t_r, 0.8 * self._t_r)),
                ParCfg(label="td2", bounds=(1.2 * self._t_r, 1.8 * self._t_r)),
            ],
            obj_cfg=[
                ObjCfg(label="Machining Time", to_minimize=True, ref_point=300.0),
                ObjCfg(label="Tool Wear", to_minimize=True, ref_point=150.0),
            ],
            trk_cfg=[
                TrkCfg(label="Orbiting Time Deviation"),
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1, -1], rhs=-150),
                LinIneqXConCfg(idxs=[2, 3], coeff=[-1, -1], rhs=-self._t_d100)
            ],
            ineq_Y_con_cfg=[
                IneqYConCfg(f=LowerBound(threshold=self._orbiting_target - self._delta, index=-1)),
                IneqYConCfg(f=UpperBound(threshold=self._orbiting_target + self._delta, index=-1))
            ]
        )
