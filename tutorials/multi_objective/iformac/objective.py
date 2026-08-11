from pathlib import Path

import torch
from torch import Tensor
from pybo.constraints.output_constraints import Identity, LowerBound, UpperBound
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg, TrkCfg


class IFormACConstrained(MCMultiObjectiveBase):
    """
    2x objectives:
        - Material Removal Rate: to maximize.
        - Tool Wear: to minimize.

    1x ineq_Y_con_cfg:
        - Orbiting Time: a hinge around the target, feasible inside the band.

    1x tracker:
        - Orbiting Time Deviation: measured against the target.

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
    _GT_FILE = "polynomial_gt.json"

    def __init__(self, device: torch.device, dtype: torch.dtype,
                 gt_file: str | Path = None):
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
                IneqYConCfg(label="Orbiting Time", unit="min", f=Identity(index=-1))
            ],
            trk_cfg=[
                TrkCfg(label="Orbiting Time Deviation", unit="min", bounds=(-20, 20))
            ],
        )

        # Anchored on __file__ rather than the working directory, so the campaign
        # runs from anywhere. Loaded after super().__init__, which is what defines
        # the device and dtype the coefficients land on.
        gt_file = gt_file or Path(__file__).resolve().parent / self._GT_FILE
        self._obj_fit = self._load_polynomial_gt(gt_file, "objectives")
        self._con_fit = self._load_polynomial_gt(gt_file, "constraints")

    def _orbiting_time(self, X: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self._evaluate_polynomial_gt(self._con_fit, X).squeeze(-1), min=0)

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return self._evaluate_polynomial_gt(self._obj_fit, X)

    def evaluate_tracker(self, X: torch.Tensor) -> torch.Tensor:
        return self._orbiting_time(X=X).unsqueeze(dim=-1) - self._orbiting_target

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        """
        target - delta <= orbiting_time <= target + delta
        """
        Y = self._orbiting_time(X=X).unsqueeze(dim=-1)
        lb = LowerBound(threshold=self._orbiting_target - self._delta, index=-1)(Y).unsqueeze(dim=-1)
        ub = UpperBound(threshold=self._orbiting_target + self._delta, index=-1)(Y).unsqueeze(dim=-1)
        return lb + ub
