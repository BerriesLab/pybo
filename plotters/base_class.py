from pathlib import Path
from typing import Tuple
import matplotlib
import torch
from optimizer.optimizer import BayesianOptimizer
from matplotlib import pyplot as plt

matplotlib.use("Agg")  # A pure renderer backend, not compatible with plt.show().
print(f"Matplotlib backend: {matplotlib.get_backend()}")


class PlotterBase:

    def __init__(self, bo: BayesianOptimizer):
        self.bo = bo
        self.figsize: Tuple[int, int] = (8, 7)
        self.n_grid_points: int = 500

    def save_figure(self, filename: str | Path | None = None):
        path = Path(filename)
        stem = path.stem
        suffix = path.suffix
        save_path = Path.cwd() / path.parent / f"{stem}_000{suffix}"

        i = 0
        while save_path.exists():
            i += 1
            save_path = Path.cwd() / path.parent / f"{stem}_{i:03d}{suffix}"

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
        d = self.bo.objective.dim

        # Create linspaces for each dimension
        linspaces = []
        for i in range(d):
            linspace = torch.linspace(bounds[0, i], bounds[1, i], self.n_grid_points, device=device)
            linspaces.append(linspace)

        if d == 1:
            return linspaces[0].unsqueeze(-1)

        # For d > 1, create a meshgrid
        grids = torch.meshgrid(*linspaces, indexing='ij')
        X_grid = torch.stack(grids, dim=-1).reshape(-1, d)

        return X_grid
