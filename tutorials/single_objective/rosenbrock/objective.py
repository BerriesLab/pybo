import torch
from pybo.objectives.base_class import MCSingleObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg


class Rosenbrock(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-2.0, 2.0)),
                ParCfg(bounds=(-1.0, 3.0))
            ],
            obj_cfg=[
                ObjCfg(label="Rosenbrock", bounds=(0, 2500), ref_point=2600, to_minimize=True)
            ],
        )

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        X0 = X[..., 0]
        X1 = X[..., 1]
        term1 = 100 * (X1 - X0 ** 2) ** 2
        term2 = (1 - X0) ** 2
        return (term1 + term2).unsqueeze(-1)
