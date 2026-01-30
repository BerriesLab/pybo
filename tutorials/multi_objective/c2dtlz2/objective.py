import torch
import math
from torch import Tensor
from objectives.base_class import MCMultiObjectiveBase
from constraints.output_constraints import Identity
from objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg


class C2DTLZ2(MCMultiObjectiveBase):
    r"""
    C2-DLTZ2 test problem.

    2-dimensional constrained problem evaluated on [0, 1]^d:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{i=m}^{d-1} (x_i - 0.5)^2

    The ineq_Y_con_cfg are imposed in the objective space as spherical exclusion regions:

        c(f) = min {(Σ_i (f_i - 1)^2 - r^2), (Σ_i (f_i - 1/√M)^2 - r^2)} ≤ 0

    where:
        - f = (f_1, ..., f_M) are the objectives,
        - M is the number of objectives,
        - r is the exclusion radius (here r = 0.2).

    Notes:
        - Feasibility (in BoTorch convention): c(f) ≤ 0 ⇒ feasible, c(f) > 0 ⇒ infeasible
        - The pareto front is completely concave. The goal is to minimize both objectives.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(0.0, 1.0)),
                ParCfg(label="P2", index=1, bounds=(0.0, 1.0)),
                ParCfg(label="P3", index=2, bounds=(0.0, 1.0)),
                ParCfg(label="P4", index=3, bounds=(0.0, 1.0)),
            ],
            obj_cfg=[
                ObjCfg(label="F1", index=0, bounds=None, to_minimize=True, ref_point=1.1, f=self._f1),
                ObjCfg(label="F2", index=1, bounds=None, to_minimize=True, ref_point=1.1, f=self._f2)
            ],
            max_hv=0.3996406303723544,  # approximate from nsga-ii
            ineq_Y_con_cfg=[
                IneqYConCfg(label="C1", index=0, f=Identity(index=-1))
            ],
        )

        self.k = self.dim - self.num_obj + 1
        self._r = 0.2

    def _g(self, X: Tensor) -> Tensor:
        xm = X[..., -self.k:]
        return torch.sum((xm - 0.5).pow(2), dim=-1)

    def _f1(self, X: Tensor) -> Tensor:
        x0 = X[..., 0]
        g_val = self._g(X)
        return (1 + g_val) * torch.cos(x0 * 0.5 * math.pi)

    def _f2(self, X: Tensor) -> Tensor:
        x0 = X[..., 0]
        g_val = self._g(X)
        return (1 + g_val) * torch.sin(x0 * 0.5 * math.pi)

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        f1 = self._f1(X)
        f2 = self._f2(X)
        # Calculate squared distances minus radius squared
        term11 = (f1 - 1).pow(2) + f2.pow(2) - self._r ** 2
        term12 = f1.pow(2) + (f2 - 1).pow(2) - self._r ** 2
        term1 = torch.min(term11, term12)
        term2 = (self._f1(X) - 1 / math.sqrt(2)).pow(2) + (self._f2(X) - 1 / math.sqrt(2)).pow(2) - self._r ** 2
        # Logic check:
        # If point is OUTSIDE spheres, torch.min(term1, term2) is POSITIVE.
        # We return -min, which is NEGATIVE (Feasible).
        # If point is INSIDE spheres, torch.min(term1, term2) is NEGATIVE.
        # We return -min, which is POSITIVE (Infeasible).
        val = -torch.min(term1, term2)
        return -torch.min(term1, term2).unsqueeze(-1)
