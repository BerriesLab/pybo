import torch
from objectives.base_class import MCSingleObjectiveBase
from objectives.variable_registry import ParCfg, ObjCfg


class Quadratic(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-1.0, 5.0)),
            ],
            obj_cfg=[
                ObjCfg(bounds=(0.0, 9.0), to_minimize=True, ref_point=10)
            ]
        )

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return (X - 2).pow(2)
