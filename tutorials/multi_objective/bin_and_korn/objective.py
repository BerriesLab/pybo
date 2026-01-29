import torch
from objectives.base_class import MCMultiObjectiveBase
from objectives.variable_registry import *


class BinhAndKorn(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions. """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=1, bounds=(0.0, 5.0)),
                ParCfg(label="P2", index=0, bounds=(0.0, 3.0))
            ],
            obj_cfg=[
                ObjCfg(label="Binh", index=1, bounds=(0.0, 140.0), to_minimize=True, ref_point=150, f=self._binh),
                ObjCfg(label="Korn", index=0, bounds=(0.0, 50.0), to_minimize=True, ref_point=60, f=self._korn)
            ],
            lin_eq_X_con_cfg=None,
            lin_ineq_X_con_cfg=None,
            nonlin_ineq_X_con_cfg=[
                NonLinIneqXConCfg(label="C1", index=0, f=self._input_c1, intra=True),
                NonLinIneqXConCfg(label="C2", index=1, f=self._input_c2, intra=True),
            ],
            ineq_Y_con_cfg=None,
        )

    def _binh(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., self.get_par_idx("P1")]
        x2 = X[..., self.get_par_idx("P2")]
        return 4 * x1 ** 2 + 4 * x2 ** 2

    def _korn(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., self.get_par_idx("P1")]
        x2 = X[..., self.get_par_idx("P2")]
        return (x1 - 5) ** 2 + (x2 - 5) ** 2

    def _input_c1(self, X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
        x1 = X[..., self.get_par_idx("P1")]
        x2 = X[..., self.get_par_idx("P2")]
        return 25 - ((x1 - 5) ** 2 + x2 ** 2)

    def _input_c2(self, X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
        x1 = X[..., self.get_par_idx("P1")]
        x2 = X[..., self.get_par_idx("P2")]
        return (x1 - 8) ** 2 + (x2 + 3) ** 2 - 7.7
