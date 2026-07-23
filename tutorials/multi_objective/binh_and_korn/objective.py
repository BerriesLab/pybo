import torch
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import *


class BinhAndKorn(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions. """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(0.0, 5.0)),
                ParCfg(bounds=(0.0, 3.0))
            ],
            obj_cfg=[
                ObjCfg(label="binh", bounds=(0.0, 140.0), to_minimize=True, ref_point=150),
                ObjCfg(label="korn", bounds=(0.0, 50.0), to_minimize=True, ref_point=60)
            ],
            lin_eq_X_con_cfg=None,
            lin_ineq_X_con_cfg=None,
            nonlin_ineq_X_con_cfg=[
                NonLinIneqXConCfg(f=self._input_c1, intra=True),
                NonLinIneqXConCfg(f=self._input_c2, intra=True),
            ],
            ineq_Y_con_cfg=None,
        )

    @staticmethod
    def _binh(X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        return 4 * x1 ** 2 + 4 * x2 ** 2

    @staticmethod
    def _korn(X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., 0]
        x2 = X[..., 1]
        return (x1 - 5) ** 2 + (x2 - 5) ** 2

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._binh(X), self._korn(X)], dim=-1)

    @staticmethod
    def _input_c1(X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
        x1 = X[..., 0]
        x2 = X[..., 1]
        return 25 - ((x1 - 5) ** 2 + x2 ** 2)

    @staticmethod
    def _input_c2(X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
        x1 = X[..., 0]
        x2 = X[..., 1]
        return (x1 - 8) ** 2 + (x2 + 3) ** 2 - 7.7
