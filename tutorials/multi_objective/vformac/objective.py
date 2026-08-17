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

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="V0", unit="V", bounds=(60, 120), resolution=1),
                ParCfg(label="dV", unit="V", bounds=(60, 85), resolution=1),
                ParCfg(label="td1", unit="ns", bounds=(0.5 * self._t_r, 0.8 * self._t_r), resolution=1),
                ParCfg(label="td2", unit="ns", bounds=(0.4 * self._t_r, 1.3 * self._t_r), resolution=1),
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
        )

    @staticmethod
    def _mrr(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 70.15) / 6.7177
        x1 = (X[..., 1] - 69.15) / 5.96049
        x2 = (X[..., 2] - 21663) / 2761.58
        x3 = (X[..., 3] - 25617.8) / 6557.44
        mrr = (2.20601
               + 0.0821756 * x0
               + 0.0833362 * x1
               - 0.11871 * x2
               - 0.15216 * x3
               - 0.0419381 * x0 ** 2
               - 0.0356976 * x0 * x1
               - 0.102584 * x0 * x2
               - 0.00995522 * x0 * x3
               + 0.016806 * x1 ** 2
               - 0.0208731 * x1 * x2
               - 0.00742038 * x1 * x3
               + 0.0399272 * x2 ** 2
               + 0.014172 * x2 * x3
               + 0.114538 * x3 ** 2)
        if noisy:
            mrr = mrr + 0.03381 * torch.randn_like(mrr)
        return mrr

    @staticmethod
    def _tw(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 70.15) / 6.7177
        x1 = (X[..., 1] - 69.15) / 5.96049
        x2 = (X[..., 2] - 21663) / 2761.58
        x3 = (X[..., 3] - 25617.8) / 6557.44
        tw = (71.2875
              + 3.02882 * x0
              + 2.1078 * x1
              + 0.443063 * x2
              - 1.7122 * x3
              - 3.23138 * x0 ** 2
              - 0.389705 * x0 * x1
              + 0.0938605 * x0 * x2
              + 0.5267 * x0 * x3
              + 2.52352 * x1 ** 2
              + 1.75328 * x1 * x2
              + 3.66823 * x1 * x3
              + 2.36657 * x2 ** 2
              + 4.71827 * x2 * x3
              + 3.71807 * x3 ** 2)
        if noisy:
            tw = tw + 6.74 * torch.randn_like(tw)
        return tw

    @staticmethod
    def _orbiting_time(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 70.15) / 6.7177
        x1 = (X[..., 1] - 69.15) / 5.96049
        x2 = (X[..., 2] - 21663) / 2761.58
        x3 = (X[..., 3] - 25617.8) / 6557.44
        orbiting_time = (18.9782
                         - 0.577431 * x0
                         - 0.861683 * x1
                         + 0.778852 * x2
                         + 0.713721 * x3
                         - 0.55626 * x0 ** 2
                         - 0.318592 * x0 * x1
                         + 0.355996 * x0 * x2
                         - 0.227398 * x0 * x3
                         + 0.395646 * x1 ** 2
                         + 0.4783 * x1 * x2
                         + 0.667992 * x1 * x3
                         - 0.452266 * x2 ** 2
                         - 0.161104 * x2 * x3
                         - 0.711864 * x3 ** 2)
        if noisy:
            orbiting_time = orbiting_time + 1.082 * torch.randn_like(orbiting_time)
        return orbiting_time

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        mrr = self._mrr(X, noisy)
        tw = self._tw(X, noisy)
        return torch.stack([mrr, tw], dim=-1)

    def evaluate_tracker(self, X: Tensor, noisy: bool = False) -> Tensor:
        orbiting_time = self._orbiting_time(X=X, noisy=noisy)
        return orbiting_time.unsqueeze(-1)
