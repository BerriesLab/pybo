import torch
from pybo.samplers.base_class import SamplerBase


class RandomSampler(SamplerBase):
    """ Independent uniform draws over the search space - no low-discrepancy structure,
    so it is the baseline Sobol itself is measured against.

    With no seed the global torch RNG is used, so a tutorial's torch.manual_seed(seed)
    governs the whole run. A seed of its own gets a generator kept on the sampler, for
    the same reason SobolSampler keeps its engine: draw_samples() calls this again
    whenever X constraints reject a block, and reseeding each time would redraw the
    identical infeasible block until the attempt budget ran out. """

    _generator: torch.Generator | None = None

    def _generate_base_samples(self, n: int) -> torch.Tensor:
        if self.seed is not None and self._generator is None:
            self._generator = torch.Generator(device=self.device).manual_seed(self.seed)
        return torch.rand(n, self.objective.dim, generator=self._generator,
                          device=self.device, dtype=self.dtype)