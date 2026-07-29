import torch
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
