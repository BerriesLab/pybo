import torch
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from utils.types import SamplerType
from botorch.utils.transforms import unnormalize


class Sampler:
    def __init__(
            self,
            sampler_type: SamplerType = SamplerType.Sobol,
            bounds: torch.Tensor or None = None,
            n_dimensions: int = 2,
            normalize: bool = True,
    ):

        self.sampler_type = sampler_type
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.normalize = normalize

    def draw_samples(self, n) -> torch.Tensor:

        if self.sampler_type == SamplerType.LatinHypercube:
            sampler = LatinHypercube(d=self.n_dimensions)
            samples = sampler.random(n=n)
            x = torch.tensor(samples)

        elif self.sampler_type == SamplerType.Sobol:
            sampler = SobolEngine(dimension=self.n_dimensions, scramble=True)
            x = sampler.draw(n=n)

        else:
            raise ValueError("Invalid initial sampling type.")

        if self.normalize:
            return x
        else:
            if self.bounds is None:
                raise ValueError("If normalize is True, then bounds cannot be None.")
            return unnormalize(x, bounds=self.bounds)