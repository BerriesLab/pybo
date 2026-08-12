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
        )

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        Y = X ** 4 - 2 * X ** 2 + 0.5 * X
        if noisy:
            Y = Y + 0.02 * torch.randn_like(Y)
        return Y

    def evaluate_true_constraint(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        # f(X) > -0.2 is feasible. The constraint is the objective shifted, so it is
        # the objective's own measurement that it has to be built on: an independent
        # draw here would be a second, disagreeing reading of the same quantity.
        return self.evaluate_true_objective(X, noisy=noisy) + 0.02
