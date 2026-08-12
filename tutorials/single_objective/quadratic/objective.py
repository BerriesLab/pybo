import torch
from pybo.objectives.base_class import MCSingleObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg


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

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        # Deterministic ground truth: there is no measurement here to be noisy.
        if noisy:
            raise ValueError(f"{type(self).__name__} declares no ground-truth "
                             f"noise. Run with --noise false.")
        return (X - 2).pow(2)
