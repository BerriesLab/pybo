import torch
from pybo.objectives.base_class import MCMultiObjectiveBase
from torch import Tensor

from pybo.objectives.variable_registry import *


class YieldRateCost(MCMultiObjectiveBase):
    """
    Three objective problem over four parameters, with no constraints.

    A smooth stand-in for a deposition process, written to exercise a mixed
    optimisation sense: Yield and Rate are maximized, Cost is minimized. All
    three pull against each other - over a uniform sample of the box every
    pairwise correlation in maximization space is negative (Yield/Rate -0.28,
    Yield/Cost -0.70, Rate/Cost -0.35), so the Pareto front is a surface rather
    than a point.

    Note:
    - Yield peaks on the expensive corner of the box, which is what sets it
      against Cost.
    - Rate wants the two parameters Yield and Cost both want low.
    - max_hv is left unset: the true optimum of this problem has not been
      computed.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="Temperature", bounds=(0.0, 1.0)),
                ParCfg(label="Pressure", bounds=(0.0, 1.0)),
                ParCfg(label="Flow", bounds=(0.0, 1.0)),
                ParCfg(label="Time", bounds=(0.0, 1.0)),
            ],
            obj_cfg=[
                ObjCfg(label="Yield", bounds=(0, 100), to_minimize=False, ref_point=5.0),
                ObjCfg(label="Rate", bounds=(0, 75), to_minimize=False, ref_point=0.0),
                ObjCfg(label="Cost", bounds=(0, 90), to_minimize=True, ref_point=90.0),
            ],
        )

    @staticmethod
    def _yield(X: Tensor) -> Tensor:
        """ A Gaussian bump centred on (0.8, 0.8), thinned by the third parameter. """
        x0 = X[..., 0]
        x1 = X[..., 1]
        x2 = X[..., 2]
        bump = torch.exp(-2 * ((x0 - 0.8).pow(2) + (x1 - 0.8).pow(2)))
        return 100 * bump * (1 - 0.3 * x2)

    @staticmethod
    def _rate(X: Tensor) -> Tensor:
        """ Linear in the last two parameters, penalised away from x0 = 0.2. """
        x0 = X[..., 0]
        x2 = X[..., 2]
        x3 = X[..., 3]
        return 50 * (x2 + 0.5 * x3) * (1 - 0.4 * (x0 - 0.2).pow(2))

    @staticmethod
    def _cost(X: Tensor) -> Tensor:
        """ Monotonically increasing in every parameter. """
        x0 = X[..., 0]
        x1 = X[..., 1]
        x2 = X[..., 2]
        x3 = X[..., 3]
        return 30 * (x0 + x1) + 20 * x2.pow(2) + 10 * x3

    def evaluate_true_objective(self, X: Tensor, noisy: bool = False) -> Tensor:
        f = torch.stack([self._yield(X=X), self._rate(X=X), self._cost(X=X)], dim=-1)
        if noisy:
            # 3% of each objective's range.
            f = f + f.new_tensor([3.0, 2.25, 2.7]) * torch.randn_like(f)
        return f
