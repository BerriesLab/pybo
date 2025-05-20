import torch
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from utils.types import SamplerType
from botorch.utils.transforms import unnormalize


def draw_samples(
        sampler_type: SamplerType,
        bounds: torch.Tensor or None = None,
        n_samples: int = 1000,
        n_dimensions: int = 2,
        normalize = True,
) -> torch.Tensor:

    if sampler_type == SamplerType.LatinHypercube:
        sampler = LatinHypercube(d=n_dimensions)
        samples = sampler.random(n=n_samples)
        x = torch.tensor(samples)
        
    elif sampler_type == SamplerType.Sobol:
        sampler = SobolEngine(dimension=n_dimensions, scramble=True)
        x = sampler.draw(n=n_samples)
    
    else:
        raise ValueError("Invalid initial sampling type.")

    if normalize:
        return x
    else:
        if bounds is None:
            raise ValueError("If normalize is True, then bounds cannot be None.")
        return unnormalize(x, bounds=bounds)