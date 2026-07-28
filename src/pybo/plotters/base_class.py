from pathlib import Path
from typing import Tuple
import matplotlib
import torch
from pybo.optimizer.optimizer import BayesianOptimizer
from matplotlib import pyplot as plt

matplotlib.use("Agg")  # A pure renderer backend, not compatible with plt.show().
print(f"Matplotlib backend: {matplotlib.get_backend()}")


class PlotterBase:

    def __init__(self, bo: BayesianOptimizer):
        self.bo = bo
        self.figsize: Tuple[int, int] = (8, 7)
        self.n_grid_points: int = 500

    def save_figure(self, filename: str | Path | None = None):
        # Fixed filename: the per-step folder gives uniqueness, mirroring
        # BayesianOptimizer.to_csv(latest=True). The counter this replaced dated
        # from the flat run directory, where it was the only thing separating one
        # step's figures from the next.
        path = Path(filename)
        save_path = Path.cwd() / path.parent / path.name

        self.fig.savefig(fname=save_path, dpi=600)
        return self

    def close_figure(self):
        plt.close(self.fig)

    def show_figure(self):
        plt.show(self.fig)

    def _generate_uniform_grid(self) -> torch.Tensor:
        """Generate a uniform dense grid over the input bounds for plotting."""
        bounds = self.bo.objective.bounds
        device = self.bo.device
        dtype = self.bo.dtype
        d = self.bo.objective.dim

        # Create linspaces for each dimension
        linspaces = []
        for i in range(d):
            linspace = torch.linspace(bounds[0, i], bounds[1, i], self.n_grid_points, device=device, dtype=dtype)
            linspaces.append(linspace)

        if d == 1:
            return linspaces[0].unsqueeze(-1)

        # For d > 1, create a meshgrid
        grids = torch.meshgrid(*linspaces, indexing='ij')
        X_grid = torch.stack(grids, dim=-1).reshape(-1, d)

        return X_grid
