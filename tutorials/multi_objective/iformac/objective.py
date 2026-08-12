import torch
from torch import Tensor

from pybo.constraints.output_constraints import Identity
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg, TrkCfg


class IFormAC(MCMultiObjectiveBase):
    """
    2x objectives:
        - Material Removal Rate: to maximize.
        - Tool Wear: to minimize.

    1x ineq_Y_con_cfg:
        - Orbiting Time Deviation: how far the orbiting time falls outside target +/- delta,
          0 while it is inside. The band is applied in evaluate_true_constraint, so
          the column already carries the "<= 0 is feasible" quantity and the cfg
          passes it through unchanged.

    1x tracker:
        - Orbiting Time: the measured minutes, recorded because the constraint
          column holds a distance rather than the measurement itself.

    3x parameters:
        - Maximum Current [7.5, 15]
        - Pedestal Current [3.0, 7.5]
        - Maximum Ramp Time [0.1*78000, 1*78000] ns, where ON time = 78000 ns

    Reference point:
        - Material Removal Rate: 0 mm3/min
        - Tool Wear: 160 um
    """

    _t_on = 78000  # ns
    _orbiting_target = (21.28 + 20.88 + 23.28) / 3  # min
    _delta = _orbiting_target / 100 * 10  # min

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="Maximum Current", unit="A", bounds=(7.5, 15.0)),
                ParCfg(label="Pedestal Current", unit="A", bounds=(3.0, 7.5)),
                ParCfg(label="Maximum Ramp Time", unit="ns", bounds=(0.1 * self._t_on, self._t_on))
            ],
            obj_cfg=[
                ObjCfg(label="Material Removal Rate", unit="mm^3/min", to_minimize=False, ref_point=0),
                ObjCfg(label="Tool Wear", unit="um", to_minimize=True, ref_point=160.0)
            ],
            ineq_Y_con_cfg=[
                # The column is already the violation, so the cfg only has to read
                # it: the band itself lives in evaluate_true_constraint.
                IneqYConCfg(label="Orbiting Time Deviation", unit="min", f=Identity(index=-1))
            ],
            trk_cfg=[
                TrkCfg(label="Orbiting Time", unit="min"),
            ],
        )

    @staticmethod
    def _mrr(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 10.863) / 2.12656
        x1 = (X[..., 1] - 5.36859) / 1.43657
        x2 = (X[..., 2] - 49683.3) / 25415.8
        mrr = (1.10922
               + 0.155163 * x0
               + 0.350718 * x1
               - 0.167241 * x2
               + 0.0184164 * x0 ** 2
               + 0.0553096 * x0 * x1
               - 0.174047 * x0 * x2
               + 0.0525138 * x1 ** 2
               - 0.0113664 * x1 * x2
               + 0.174277 * x2 ** 2)
        if noisy:
            mrr = mrr + 0.002773 * torch.randn_like(mrr)
        return torch.clamp(mrr, min=0)

    @staticmethod
    def _tw(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 10.863) / 2.12656
        x1 = (X[..., 1] - 5.36859) / 1.43657
        x2 = (X[..., 2] - 49683.3) / 25415.8
        tw = (37.156
              + 8.7068 * x0
              + 13.1559 * x1
              - 9.12355 * x2
              + 5.14101 * x0 ** 2
              + 0.301578 * x0 * x1
              - 13.2499 * x0 * x2
              + 2.49776 * x1 ** 2
              - 5.06397 * x1 * x2
              + 9.49278 * x2 ** 2)
        if noisy:
            tw = tw + 8.957 * torch.randn_like(tw)
        return torch.clamp(tw, min=0)

    @staticmethod
    def _orbiting_time(X: torch.Tensor, noisy=False) -> torch.Tensor:
        x0 = (X[..., 0] - 10.863) / 2.12656
        x1 = (X[..., 1] - 5.36859) / 1.43657
        x2 = (X[..., 2] - 49683.3) / 25415.8
        orbiting_time = (20.9966
                         - 4.28653 * x0
                         - 3.7832 * x1
                         + 1.3988 * x2
                         + 0.877483 * x0 ** 2
                         + 1.52774 * x0 * x1
                         + 0.266602 * x0 * x2
                         + 0.414995 * x1 ** 2
                         - 0.415393 * x1 * x2
                         - 0.217852 * x2 ** 2)
        if noisy:
            orbiting_time = orbiting_time + 0.6754 * torch.randn_like(orbiting_time)
        return torch.clamp(orbiting_time, min=0)

    def evaluate_true_objective(self, X: torch.Tensor, noisy=False) -> torch.Tensor:
        material_remove_rate = self._mrr(X=X, noisy=noisy)
        tool_wear = self._tw(X=X, noisy=noisy)
        return torch.stack([material_remove_rate, tool_wear], dim=-1)

    def evaluate_tracker(self, X: torch.Tensor, noisy=False) -> torch.Tensor:
        # The measurement itself, recorded because the constraint column holds a
        # distance from the band rather than the minutes the rig would report.
        return self._orbiting_time(X=X, noisy=noisy).unsqueeze(dim=-1)

    def evaluate_true_constraint(self, X: Tensor, noisy=False) -> Tensor:
        """0 while the orbiting time is inside target +/- delta, and the distance
        from the nearer edge outside it - the "<= 0 is feasible" quantity the
        optimizer consumes directly. At most one of the two terms is non-zero. """
        orbiting_time = self._orbiting_time(X=X, noisy=noisy)
        below = torch.relu(self._orbiting_target - self._delta - orbiting_time)
        above = torch.relu(orbiting_time - self._orbiting_target - self._delta)
        return (below + above).unsqueeze(dim=-1)
