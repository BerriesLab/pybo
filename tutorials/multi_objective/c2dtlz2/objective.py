import torch
import math
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.constraints.output_constraints import Identity
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg


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
                ParCfg(bounds=(0.0, 1.0)),
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

    # The noise sits with each measured objective, and everything computed from
    # them - the disks below - inherits it by taking them noisy rather than by
    # drawing again.
    def _f1(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        x0 = X[..., 0]
        g_val = self._g(X)
        f1 = (1 + g_val) * torch.cos(x0 * 0.5 * math.pi)
        return f1 + 0.015 * torch.randn_like(f1) if noisy else f1

    def _f2(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        x0 = X[..., 0]
        g_val = self._g(X)
        f2 = (1 + g_val) * torch.sin(x0 * 0.5 * math.pi)
        return f2 + 0.015 * torch.randn_like(f2) if noisy else f2

    def evaluate_true_constraint(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        # Which disk the measured (f1, f2) lands in. A constraint is never measured
        # on its own: it is read off the objectives, so a noisy run has to ask them
        # for the same values it recorded, not for a second independent draw.
        f1 = self._f1(X, noisy=noisy)
        f2 = self._f2(X, noisy=noisy)
        a = 1 / math.sqrt(2)

        # Disk 1: center in (1, 0)
        # c1 = self._r ** 2 - ((f1 - 1).pow(2) + f2.pow(2))
        d1_sq = (f1 - 1).pow(2) + f2.pow(2)

        # Disk 2: center in (0, 1)
        # c2 = self._r ** 2 - (f1.pow(2) + (f2 - 1).pow(2))
        d2_sq = f1.pow(2) + (f2 - 1).pow(2)

        # Disk 3: center in (1/sqrt(2), 1/sqrt(2))
        # c3 = self._r ** 2 - ((f1 - a).pow(2) + (f2 - a).pow(2))
        d3_sq = (f1 - a).pow(2) + (f2 - a).pow(2)

        # The constraint is violated if (f_1(X), f_2(X)) is inside any
        # of the three disks
        # combined_violation = torch.min(torch.stack([c1, c2, c3], dim=-1), dim=-1)[0]
        min_dist_sq = torch.min(torch.stack([d1_sq, d2_sq, d3_sq], dim=-1), dim=-1)[0]
        combined_violation = self._r ** 2 - min_dist_sq

        # BoTorch convention:
        # > 0 -> Infeasible
        # <= 0 -> Feasible
        return combined_violation.unsqueeze(-1)

    def evaluate_true_objective(self, X: torch.Tensor, noisy: bool = False) -> torch.Tensor:
        return torch.stack([self._f1(X, noisy=noisy), self._f2(X, noisy=noisy)], dim=-1)
