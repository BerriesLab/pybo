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
        )

    @staticmethod
    def _mrr(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 73.0398) / 10.3042
        x1 = (X[..., 1] - 71.4636) / 8.24348
        x2 = (X[..., 2] - 21097.8) / 3046.4
        x3 = (X[..., 3] - 27003.8) / 8058.64
        mrr = (2.4836
               + 0.180844 * x0
               + 0.292312 * x1
               - 0.046974 * x2
               - 0.0418647 * x3
               + 0.081316 * x0 ** 2
               + 0.00627412 * x0 * x1
               + 0.0197576 * x0 * x2
               + 0.032413 * x0 * x3
               - 0.00772497 * x1 ** 2
               - 0.00530071 * x1 * x2
               - 0.0136146 * x1 * x3
               - 0.0524083 * x2 ** 2
               - 0.0787928 * x2 * x3
               + 3.32285e-05 * x3 ** 2)
        if noisy:
            mrr = mrr + 0.03381 * torch.randn_like(mrr)
        return mrr

    @staticmethod
    def _tw(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 73.0398) / 10.3042
        x1 = (X[..., 1] - 71.4636) / 8.24348
        x2 = (X[..., 2] - 21097.8) / 3046.4
        x3 = (X[..., 3] - 27003.8) / 8058.64
        tw = (73.4973
              + 0.880948 * x0
              + 3.992 * x1
              - 1.05452 * x2
              - 1.30625 * x3
              - 1.84203 * x0 ** 2
              + 0.0442813 * x0 * x1
              - 0.803941 * x0 * x2
              - 0.089807 * x0 * x3
              + 0.740801 * x1 ** 2
              - 0.343838 * x1 * x2
              - 1.03894 * x1 * x3
              + 1.47413 * x2 ** 2
              + 1.49911 * x2 * x3
              + 1.60942 * x3 ** 2)
        if noisy:
            tw = tw + 6.74 * torch.randn_like(tw)
        return tw

    @staticmethod
    def _orbiting_time(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 73.0398) / 10.3042
        x1 = (X[..., 1] - 71.4636) / 8.24348
        x2 = (X[..., 2] - 21097.8) / 3046.4
        x3 = (X[..., 3] - 27003.8) / 8058.64
        orbiting_time = (17.2021
                         - 1.46948 * x0
                         - 1.84554 * x1
                         + 0.294847 * x2
                         + 0.334982 * x3
                         - 0.921588 * x0 ** 2
                         - 0.455983 * x0 * x1
                         + 0.0783536 * x0 * x2
                         + 0.273357 * x0 * x3
                         + 0.107192 * x1 ** 2
                         + 0.00211395 * x1 * x2
                         + 0.145598 * x1 * x3
                         + 0.16593 * x2 ** 2
                         + 0.0182557 * x2 * x3
                         - 0.170509 * x3 ** 2)
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
