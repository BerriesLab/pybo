import torch
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, LinIneqXConCfg, NonLinIneqXConCfg


class OsyczkaKundu(MCMultiObjectiveBase):
    r"""
    Two-objective problem with a set of linear inequality constraints on the input.
    ref: https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(0.0, 10)),
                ParCfg(bounds=(0.0, 10.0)),
                ParCfg(bounds=(1.0, 5.0)),
                ParCfg(bounds=(0.0, 6.0)),
                ParCfg(bounds=(1.0, 5.0)),
                ParCfg(bounds=(0.0, 10.0)),
            ],
            obj_cfg=[
                ObjCfg(label="Osyczka", to_minimize=True, ref_point=0.0),
                ObjCfg(label="Kundu", to_minimize=True, ref_point=160)
            ],
            lin_ineq_X_con_cfg=[
                LinIneqXConCfg(idxs=[0, 1], coeff=[1.0, 1.0], rhs=2.0),
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1.0, -1.0], rhs=-6.0),
                LinIneqXConCfg(idxs=[0, 1], coeff=[1.0, -1.0], rhs=-2.0),
                LinIneqXConCfg(idxs=[0, 1], coeff=[-1.0, 3.0], rhs=-2.0)
            ],
            nonlin_ineq_X_con_cfg=[
                NonLinIneqXConCfg(f=self._nonlinear_c1, intra=True),
                NonLinIneqXConCfg(f=self._nonlinear_c2, intra=True)
            ]
        )

    @staticmethod
    def _f1(X: torch.Tensor) -> torch.Tensor:
        return (
                - 25 * (X[..., 0] - 2).pow(2)
                - (X[..., 1] - 2).pow(2)
                - (X[..., 2] - 1).pow(2)
                - (X[..., 3] - 4).pow(2)
                - (X[..., 4] - 1).pow(2)
        )

    @staticmethod
    def _f2(X: torch.Tensor) -> torch.Tensor:
        return (
                + X[..., 0].pow(2)
                + X[..., 1].pow(2)
                + X[..., 2].pow(2)
                + X[..., 3].pow(2)
                + X[..., 4].pow(2)
                + X[..., 5].pow(2)
        )

    @staticmethod
    def _nonlinear_c1(X: torch.Tensor) -> torch.Tensor:
        return 4 - (X[..., 2] - 3).pow(2) - X[..., 3]

    @staticmethod
    def _nonlinear_c2(X: torch.Tensor) -> torch.Tensor:
        return (X[..., 4] - 3).pow(2) + X[..., 5] - 4

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        return torch.stack([f1, f2], dim=-1)
