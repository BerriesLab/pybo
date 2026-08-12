import torch
from pybo.objectives.base_class import MCObjectiveBase, MCSingleObjectiveBase
from pybo.objectives.variable_registry import *


class RosenbrockConstrained(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-1.5, 1.5)),
                ParCfg(bounds=(-1.5, 1.5))
            ],
            obj_cfg=[
                ObjCfg(label="Rosenbrock", bounds=(0, 2500), ref_point=2600, to_minimize=True)
            ],
            nonlin_ineq_X_con_cfg=[
                NonLinIneqXConCfg(f=self.disk_constraint, intra=True)
            ]
        )

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        # Deterministic ground truth: there is no measurement here to be noisy.
        if noisy:
            raise ValueError(f"{type(self).__name__} declares no ground-truth "
                             f"noise. Run with --noise false.")
        X0 = X[..., 0]
        X1 = X[..., 1]
        term1 = 100 * (X1 - X0 ** 2) ** 2
        term2 = (1 - X0) ** 2
        return (term1 + term2).unsqueeze(-1)

    @staticmethod
    def disk_constraint(X: torch.Tensor) -> torch.Tensor:
        X0 = X[..., 0]
        X1 = X[..., 1]
        return 2.0 - (X0 ** 2 + X1 ** 2)
