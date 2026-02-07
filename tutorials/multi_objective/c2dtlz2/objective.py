import torch
import math
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

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(0.0, 1.0)),
                ParCfg(bounds=(0.0, 1.0)),
                ParCfg(bounds=(0.0, 1.0)),
                ParCfg(index=3, bounds=(0.0, 1.0)),
            ],
            obj_cfg=[
                ObjCfg(to_minimize=True, ref_point=1.6),
                ObjCfg(to_minimize=True, ref_point=1.6)
            ],
            max_hv=0.3996406303723544,  # approximate from nsga-ii
            ineq_Y_con_cfg=[
                IneqYConCfg(f=Identity(index=-1))
            ],
        )

        self.k = self.dim - self.num_obj + 1
        self._r = 0.2

    def _g(self, X: torch.Tensor) -> torch.Tensor:
        xm = X[..., -self.k:]
        return torch.sum((xm - 0.5).pow(2), dim=-1)

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X[..., 0]
        g_val = self._g(X)
        return (1 + g_val) * torch.cos(x0 * 0.5 * math.pi)

    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        x0 = X[..., 0]
        g_val = self._g(X)
        return (1 + g_val) * torch.sin(x0 * 0.5 * math.pi)

    def evaluate_true_constraint(self, X: torch.Tensor) -> torch.Tensor:
        f1 = self._f1(X)
        f2 = self._f2(X)
        a = 1 / math.sqrt(2)

        # Disk 1: center in (1, 0)
        c1 = self._r ** 2 - ((f1 - 1).pow(2) + f2.pow(2))

        # Disk 2: center in (0, 1)
        c2 = self._r ** 2 - (f1.pow(2) + (f2 - 1).pow(2))

        # Disk 3: center in (1/sqrt(2), 1/sqrt(2))
        c3 = self._r ** 2 - ((f1 - a).pow(2) + (f2 - a).pow(2))

        # The constraint is violated if (f_1(X), f_2(X)) is inside any
        # of the three disks
        combined_violation = torch.max(torch.stack([c1, c2, c3], dim=-1), dim=-1)[0]

        # BoTorch convention:
        # > 0 -> Infeasible (dentro i cerchi rosa)
        # <= 0 -> Feasible (tutto il resto)
        return combined_violation.unsqueeze(-1) * 100

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._f1(X), self._f2(X)], dim=-1)
