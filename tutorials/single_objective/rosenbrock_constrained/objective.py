from objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase
import torch
from torch import Tensor

from objectives.variable_registry import *


class RosenbrockConstrained(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-1.5, 1.5)),
                ParCfg(label="P2", index=1, bounds=(-1.5, 1.5))
            ],
            obj_cfg=[
                ObjCfg(label="Rosenbrock", index=0, f=self._f_1, bounds=None, to_minimize=True)
            ],
            nonlin_ineq_X_con_cfg=[
                NonLinIneqXConCfg(label="c1", index=0, f=self.disk_constraint, intra=True)
            ]
        )

    def _f_11(self, X: torch.Tensor) -> torch.Tensor:
        X0 = X[..., self.get_par_idx("P1")]
        X1 = X[..., self.get_par_idx("P2")]
        return 100 * (X1 - X0 ** 2) ** 2

    def _f_12(self, X: torch.Tensor) -> torch.Tensor:
        X0 = X[..., self.get_par_idx("P1")]
        return (1 - X0) ** 2

    def _f_1(self, X: torch.Tensor) -> torch.Tensor:
        return (self._f_11(X) + self._f_12(X)).unsqueeze(-1)

    def disk_constraint(self, X: Tensor) -> Tensor:
        X0 = X[..., self.get_par_idx("P1")]
        X1 = X[..., self.get_par_idx("P2")]
        return 2.0 - (X0 ** 2 + X1 ** 2)
