import torch
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from pybo.samplers.base_class import SamplerBase


class SobolSampler(SamplerBase):
    """ Keeps one engine per sampler so that successive calls advance through the
    Sobol sequence. draw_samples() calls this again whenever X constraints reject a
    block, and with a seed set, rebuilding the engine each time would redraw the
    identical infeasible block until the attempt budget ran out. """

    _engine: SobolEngine | None = None

    def _generate_base_samples(self, n: int) -> torch.Tensor:
        if self._engine is None:
            self._engine = SobolEngine(dimension=self.objective.dim, scramble=True, seed=self.seed)
        return self._engine.draw(n=n).to(device=self.device, dtype=self.dtype)


class LatinHypercubeSampler(SamplerBase):
    """Latin Hypercube Sampling should be used only for unconstrained input domains.
    This because the sampling is not additive. Therefore, every time a set of parameters
    X is rejected, one should resample the whole space from scratch. """

    def _generate_base_samples(self, n: int) -> torch.Tensor:
        sampler = LatinHypercube(d=self.objective.dim, seed=self.seed)
        samples = sampler.random(n=n)
        return torch.tensor(samples, device=self.device, dtype=self.dtype)


class UniformGridSampler(SamplerBase):
    def _generate_base_samples(self, n: int) -> torch.Tensor:
        points_per_dim = int(n ** (1 / self.objective.dim))
        grid_axes = [torch.linspace(0, 1, points_per_dim, device=self.device, dtype=self.dtype)
                     for _ in range(self.objective.dim)]
        grid = torch.meshgrid(*grid_axes, indexing='ij')
        return torch.stack(grid, dim=-1).reshape(-1, self.objective.dim)
