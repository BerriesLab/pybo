from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor

from objectives.variable_registry import ParCfg, ObjCfg


class Polynomial(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-2.0, 2.0))
            ],
            obj_cfg=[
                ObjCfg(label="F", index=0, f=self._f, bounds=(-2.0, 8.0), to_minimize=True, ref_point=None),
            ],
        )

    @staticmethod
    def _f(X: Tensor) -> Tensor:
        return X ** 4 - 2 * X ** 2 + 0.5 * X
