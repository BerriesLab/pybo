import torch
import math
from torch import Tensor
from objectives.base_class import MCMultiOutputBase
from constraints.output_constraints import Identity


class C2DTLZ2MCMultiOutputObjective(MCMultiOutputBase):
    r"""
    DLTZ2 test problem.

    d-dimensional problem evaluated on `[0, 1]^d`:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{i=m}^{d-1} (x_i - 0.5)^2

    The pareto front is given by the unit hypersphere \sum{i} f_i^2 = 1.
    Note: the pareto front is completely concave. The goal is to minimize
    both objectives.

    The constraint computes the minimum distance to two types of structures in objective space:
    Notes: negative constraint values imply feasibility in botorch.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype,):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=4,
            num_objectives=2,
            num_constraints=1,
            obj_to_minimize=[True, True],
            bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            ref_point=[1.1, 1.1],
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=0.3996406303723544,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=[Identity(index=-1)]
        )

        self.k = self.dim - self.num_objectives + 1
        self._r = 0.2

    def evaluate_true(self, X: Tensor) -> Tensor:
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
        f = torch.stack(fs, dim=-1)
        return super().evaluate_true(f)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        if X.ndim > 2:
            raise NotImplementedError("Batch X is not supported.")
        f_X = self.evaluate_true(X)
        term1 = (f_X - 1).pow(2)
        mask = ~(torch.eye(f_X.shape[-1], device=f_X.device).bool())
        indices = torch.arange(f_X.shape[1], device=f_X.device).repeat(f_X.shape[1], 1)
        indexer = indices[mask].view(f_X.shape[1], f_X.shape[-1] - 1)
        term2_inner = (
            f_X.unsqueeze(1)
            .expand(f_X.shape[0], f_X.shape[-1], f_X.shape[-1])
            .gather(dim=-1, index=indexer.repeat(f_X.shape[0], 1, 1))
        )
        term2 = (term2_inner.pow(2) - self._r**2).sum(dim=-1)
        min1 = (term1 + term2).min(dim=-1).values
        min2 = ((f_X - 1 / math.sqrt(f_X.shape[-1])).pow(2) - self._r**2).sum(dim=-1)
        slack_true = -torch.min(min1, min2).unsqueeze(-1)
        return -super().evaluate_slack_true(slack_true)

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected
