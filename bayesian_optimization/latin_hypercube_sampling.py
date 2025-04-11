from scipy.stats import qmc


class LatinHypercubeSampling:
    def __init__(self):
        self._bounds: list[tuple[float, float]] or None = None
        self._n_samples: int or None = None

    def set_bounds(self, bounds: list[tuple[float, float]]):
        self._bounds = bounds

    def get_bounds(self) -> list[tuple[float, float]] or None:
        return self._bounds

    def set_n_samples(self, n_samples: int):
        self._n_samples = n_samples

    def get_n_samples(self) -> int or None:
        return self._n_samples

    def sample_domain(self):
        sampler = qmc.LatinHypercube(d=len(self._bounds))
        sample = sampler.random(self._n_samples)
        l_bounds = [b[0] for b in self._bounds]
        u_bounds = [b[1] for b in self._bounds]
        rescaled_samples = qmc.scale(sample, l_bounds, u_bounds)
        return rescaled_samples