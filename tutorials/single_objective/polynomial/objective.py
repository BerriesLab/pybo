import torch
from pybo.objectives.base_class import MCSingleObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg


class Polynomial(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-2.0, 2.0))
            ],
            obj_cfg=[
                ObjCfg(bounds=(-2.0, 8.0), to_minimize=True, ref_point=10),
            ],
        )

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return X ** 4 - 2 * X ** 2 + 0.5 * X
