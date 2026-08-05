import torch
from pybo.objectives.base_class import MCSingleObjectiveBase
from pybo.objectives.variable_registry import *
from pybo.constraints.output_constraints import *


class PolynomialConstrained(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-2.0, 2.0))
            ],
            obj_cfg=[
                ObjCfg(bounds=(-2.0, 8.0), to_minimize=False, ref_point=10)
            ],
            ineq_Y_con_cfg=[
                IneqYConCfg(f=Identity(index=-1))
            ],
            gt_obj_noise_std=[0.02],
            gt_con_noise_std=[0.01]
        )

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return X ** 4 - 2 * X ** 2 + 0.5 * X

    def evaluate_true_constraint(self, X: torch.Tensor) -> torch.Tensor:
        # f(X) > -0.2 is feasible
        return self.evaluate_true_objective(X) + 0.02
