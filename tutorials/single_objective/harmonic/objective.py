import torch
from torch import Tensor
from pybo.objectives.base_class import MCSingleObjectiveBase
from pybo.objectives.variable_registry import *


class Harmonic(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-3.0, 3.0))
            ],
            obj_cfg=[
                ObjCfg(bounds=(-2.0, 2.0), to_minimize=True)
            ],
        )

    def evaluate_true_objective(self, X: Tensor, noisy: bool = False) -> Tensor:
        # Deterministic ground truth: there is no measurement here to be noisy.
        if noisy:
            raise ValueError(f"{type(self).__name__} declares no ground-truth "
                             f"noise. Run with --noise false.")
        return torch.cos(2 * torch.pi * X)
