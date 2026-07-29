import torch
from scipy.stats._qmc import LatinHypercube

from pybo.samplers.base_class import SamplerBase


class LatinHypercubeSampler(SamplerBase):
    """Latin Hypercube Sampling should be used only for unconstrained input domains.
    This because the sampling is not additive. Therefore, every time a set of parameters
    X is rejected, one should resample the whole space from scratch. """

    def _generate_base_samples(self, n: int) -> torch.Tensor:
        sampler = LatinHypercube(d=self.objective.dim, seed=self.seed)
        samples = sampler.random(n=n)
        return torch.tensor(samples, device=self.device, dtype=self.dtype)
