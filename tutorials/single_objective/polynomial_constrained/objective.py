import torch
from objectives.base_class import MCSingleObjectiveBase
from constraints.output_constraints import *

from objectives.variable_registry import *


class PolynomialConstrained(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-2.0, 2.0))
            ],
            obj_cfg=[
                ObjCfg(label="F1", index=0, bounds=(-2.0, 8.0), to_minimize=False, ref_point=None, f=self._f)
            ],
            ineq_Y_con_cfg=[
                IneqYConCfg(label="C1", index=0, f=Identity(index=-1))
            ],
        )

    @staticmethod
    def _f(X: torch.Tensor) -> torch.Tensor:
        return X ** 4 - 2 * X ** 2 + 0.5 * X

    def evaluate_true_constraint(self, X: torch.Tensor) -> torch.Tensor:
        # f(X) > -0.2 is feasible
        return self.evaluate_true_objective(X) + 0.02
