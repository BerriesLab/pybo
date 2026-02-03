import torch
from torch import Tensor
from botorch.utils.transforms import unnormalize
from abc import ABC, abstractmethod
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from objectives.base_class import MCObjectiveBase


class SamplerBase(ABC):
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            objective: MCObjectiveBase,
            seed: int | None = None
    ):
        self.device = device
        self.dtype = dtype
        self.objective = objective
        self.seed = seed

    @abstractmethod
    def _generate_base_samples(self, n: int) -> Tensor:
        """ Generates n random samples of shape [n, dim] in the range [0, 1].
        It must be implemented by each subclass."""
        pass

    def draw_samples(self, n: int) -> torch.Tensor:
        """ Draws n random samples of shape [n, dim] subject to X constraints."""
        valid_X = []
        num_attempts = 0
        max_attempts = 1000

        while len(valid_X) < n and num_attempts < max_attempts:
            num_attempts += 1

            norm_X = self._generate_base_samples(n=n)
            X = unnormalize(norm_X, self.objective.bounds)
            feasible_mask = self.objective.is_X_feasible(X=X)
            feasible_norm_X = norm_X[feasible_mask]
            if feasible_norm_X.shape[0] > 0:
                feasible_X = unnormalize(feasible_norm_X, self.objective.bounds)
                valid_X.append(feasible_X)

        if not valid_X:
            raise RuntimeError(f"No valid samples found after {max_attempts} attempts.")

        return torch.cat(valid_X, dim=0)[:n]


class SobolSampler(SamplerBase):
    def _generate_base_samples(self, n: int) -> torch.Tensor:
        engine = SobolEngine(dimension=self.objective.dim, scramble=True, seed=self.seed)
        return engine.draw(n=n).to(device=self.device, dtype=self.dtype)


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
