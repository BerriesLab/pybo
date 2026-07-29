import torch

from pybo.samplers.base_class import SamplerBase


class UniformGridSampler(SamplerBase):
    def _generate_base_samples(self, n: int) -> torch.Tensor:
        points_per_dim = int(n ** (1 / self.objective.dim))
        grid_axes = [torch.linspace(0, 1, points_per_dim, device=self.device, dtype=self.dtype)
                     for _ in range(self.objective.dim)]
        grid = torch.meshgrid(*grid_axes, indexing='ij')
        return torch.stack(grid, dim=-1).reshape(-1, self.objective.dim)
