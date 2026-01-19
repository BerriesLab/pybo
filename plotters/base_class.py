from pathlib import Path
from typing import List, Tuple
import matplotlib
import torch
from bayesian_optimizer.optimizer import BayesianOptimizer

matplotlib.use("Agg")  # A pure renderer backend, not compatible with plt.show().
print(f"Matplotlib backend: {matplotlib.get_backend()}")
from matplotlib import pyplot as plt


class PlotterBase:
    """ A base class for plotting. """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            title: str | None = None,
            labels: List[str] | None = None,
            lims: List[Tuple[float, float]] | None = None,
            figsize: Tuple[int, int] = (8, 7),
    ):
        self.bo = bayesian_optimizer
        self.title = title
        self.labels = labels
        self.lims = lims
        self.figsize = figsize
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.cbar: plt.Colorbar | None = None
        self._initialize_figure()
        self._set_labels()
        self._set_x_lims()
        self._set_y_lims()
        self.legend_elements = []

    def _initialize_figure(self):
        if self.title is not None:
            self.ax.set_title(self.title)
        if self.labels is not None:
            self._set_labels()

    def _set_x_lims(self):
        if self.lims is not None and self.lims[0] is not None:
            self.ax.set_xlim(self.lims[0][0], self.lims[0][1])
        else:
            self.ax.autoscale(enable=True, axis='x')

    def _set_y_lims(self):
        if self.lims is not None and self.lims[1] is not None:
            self.ax.set_ylim(self.lims[1][0], self.lims[1][1])
        else:
            self.ax.autoscale(enable=True, axis='y')

    def _set_labels(self):
        if self.labels is not None:
            self.ax.set_xlabel(self.labels[0])
            self.ax.set_ylabel(self.labels[1])
        if self.cbar is not None:
            self.cbar.set_label(self.labels[2])

    # TODO: extend grid generation to more than one input dimensions
    def _generate_grid(self, n_grid_points=1000) -> torch.Tensor:
        """Generate a dense grid over the input bounds for plotting."""
        bounds = self.bo.objective.bounds
        device = self.bo.device
        dtype = self.bo.dtype

        X_grid = torch.linspace(
            bounds[0, 0].item(),
            bounds[1, 0].item(),
            n_grid_points,
            device=device,
            dtype=dtype
        ).unsqueeze(-1)

        return X_grid

    # TODO:
    # """ Generate samples for ground truth evaluation - random sampler or grid """
    # # When constraints apply to the input X, build the ground truth by using
    # # a random generator subject to constraints
    # X_gt = sampler.draw_samples(n=1000)

    def save_figure(self, filename: str | Path | None = None):
        if filename is None:
            filename = "noname.png"

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
