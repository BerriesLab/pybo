import torch
from scipy.stats._qmc import LatinHypercube
from pybo.samplers.base_class import SamplerBase


class LatinHypercubeSampler(SamplerBase):
    """Latin Hypercube Sampling should be used only for unconstrained input domains.
    This because the sampling is not additive. Therefore, every time a set of parameters
    X is rejected, one should resample the whole space from scratch.

    Keeps one generator per sampler so that a rejected block is followed by a different
    one. Building a new generator per call would repeat the same rejected block whenever
    a seed is set. """

    _engine: LatinHypercube | None = None

    def _generate_base_samples(self, n: int) -> torch.Tensor:
        if self._engine is None:
            self._engine = LatinHypercube(d=self.objective.dim, rng=self.seed)
        samples = self._engine.random(n=n)
        return torch.tensor(samples, device=self.device, dtype=self.dtype)
