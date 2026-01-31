from objectives.base_class import MCSingleObjectiveBase
import torch
from torch import Tensor

from objectives.variable_registry import ParCfg, ObjCfg


class Quadratic(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-1.0, 5.0))
            ],
            obj_cfg=[
                ObjCfg(label="f", index=0, f=self._f, bounds=(0.0, 9.0), to_minimize=True, ref_point=10)
            ]
        )

    @staticmethod
    def _f(X: Tensor) -> Tensor:
        return (X - 2).pow(2)
