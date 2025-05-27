import torch
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from utils.types import SamplerType
from botorch.utils.transforms import unnormalize
from collections.abc import Callable


class Sampler:
    def __init__(
            self,
            sampler_type: SamplerType = SamplerType.Sobol,
            bounds: torch.Tensor | None = None,
            n_dimensions: int = 2,
            normalize: bool = True,
            constraint: list[Callable] | None = None
    ):

        self.sampler_type = sampler_type
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.normalize = normalize
        self.constraint = constraint

    def draw_samples(self, n) -> torch.Tensor:
        valid_x = []
        num_attempts = 0
        max_attempts = 100  # To avoid infinite loops

        while len(valid_x) < n and num_attempts < max_attempts:
            num_attempts += 1

            # Draw raw samples
            if self.sampler_type == SamplerType.LatinHypercube:
                sampler = LatinHypercube(d=self.n_dimensions)
                samples = sampler.random(n=n)
                x = torch.tensor(samples)

            elif self.sampler_type == SamplerType.Sobol:
                sampler = SobolEngine(dimension=self.n_dimensions, scramble=True)
                x = sampler.draw(n=n)

            else:
                raise ValueError("Invalid initial sampling type.")

            # Unnormalize if needed
            if not self.normalize:
                if self.bounds is None:
                    raise ValueError("If normalize is True, then bounds cannot be None.")
                x = unnormalize(x, bounds=self.bounds)

            # Apply constraints if any
            if self.constraint:
                constraint_mask = torch.ones(x.shape[0], dtype=torch.bool)
                for constraint_fn in self.constraint:
                    constraint_mask &= (constraint_fn(x) <= 0)
                x = x[constraint_mask]

            valid_x.append(x)

        # Concatenate and trim to n
        valid_samples = torch.cat(valid_x, dim=0)
        if valid_samples.shape[0] < n:
            raise RuntimeError(f"Only {valid_samples.shape[0]} valid samples found after {num_attempts} attempts.")

        return valid_samples[:n]
