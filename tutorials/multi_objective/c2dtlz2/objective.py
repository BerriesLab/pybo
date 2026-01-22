import torch
import math
from torch import Tensor
from objectives.base_class import MCMultiOutputBase
from constraints.output_constraints import Identity


class C2DTLZ2MCMultiOutputObjective(MCMultiOutputBase):
    """
    C2-DLTZ2 test problem.

    2-dimensional constrained problem evaluated on [0, 1]^d:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{idx=m}^{d-1} (x_i - 0.5)^2

    The constraints are imposed in the objective space as spherical exclusion regions:

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
            dim=4,
            num_objectives=2,
            num_constraints=1,
            num_trackers=0,
            obj_to_minimize=[True, True],
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            ref_point=[1.1, 1.1],
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=0.3996406303723544,  # approximate from nsga-ii
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=[Identity(index=-1)],
        )

        self.k = self.dim - self.num_objectives + 1
        self._r = 0.2

    def evaluate_true_objective(self, X: Tensor) -> Tensor:
        X_m = X[..., -self.k:]
        g_X = (X_m - 0.5).pow(2).sum(dim=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = math.pi / 2
        for i in range(self.num_objectives):
            idx = self.num_objectives - 1 - i
            f_i = g_X_plus1.clone()
            f_i *= torch.cos(X[..., :idx] * pi_over_2).prod(dim=-1)
            if i > 0:
                f_i *= torch.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)
        return torch.stack(fs, dim=-1)

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        f_X = self.evaluate_true_objective(X)
        term1 = (f_X - 1).pow(2)
        mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
        indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
        indexer = indices[mask].view(f_X.shape[1], f_X.shape[-1] - 1)
        term2_inner = (
            f_X.unsqueeze(1)
            .expand(f_X.shape[0], f_X.shape[-1], f_X.shape[-1])
            .gather(dim=-1, index=indexer.repeat(f_X.shape[0], 1, 1))
        )
        term2 = (term2_inner.pow(2) - self._r ** 2).sum(dim=-1)
        min1 = (term1 + term2).min(dim=-1).values
        min2 = ((f_X - 1 / math.sqrt(f_X.shape[-1])).pow(2) - self._r ** 2).sum(dim=-1)
        return torch.min(min1, min2).unsqueeze(-1)

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected
