import torch
from objectives.base_class import MCSingleObjectiveBase

from objectives.variable_registry import *


class Periodic(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-3.0, 3.0))
            ],
            obj_cfg=[
                ObjCfg(label="F1", index=0, bounds=(-2.0, 2.0), to_minimize=True, f=self._f1)
            ],
        )

    @staticmethod
    def _f1(X: torch.Tensor) -> torch.Tensor:
        return torch.sin(2 * torch.pi * 2 * X) + 0.5 * torch.sin(4 * torch.pi * 2 * X)
