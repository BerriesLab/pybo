from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor

from objectives.variable_registry import ParCfg, ObjCfg


class Rosenbrock(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-2.0, 2.0)),
                ParCfg(label="P2", index=1, bounds=(-1.0, 3.0))
            ],
            obj_cfg=[
                ObjCfg(label="Rosenbrock", index=0, f=self._f_1, bounds=None, to_minimize=True)
            ],
        )

    @staticmethod
    def _f_11(X: torch.Tensor) -> torch.Tensor:
        X0 = X[..., 0]
        X1 = X[..., 1]
        return 100 * (X1 - X0 ** 2) ** 2

    @staticmethod
    def _f_12(X: torch.Tensor) -> torch.Tensor:
        X0 = X[..., 0]
        return (1 - X0) ** 2

    def _f_1(self, X: torch.Tensor) -> torch.Tensor:
        return (self._f_11(X) + self._f_12(X)).unsqueeze(-1)
