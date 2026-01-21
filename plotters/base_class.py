from pathlib import Path
from typing import List, Tuple
import matplotlib
import torch
from bayesian_optimizer.optimizer import BayesianOptimizer
from samplers.samplers import SamplerBase

matplotlib.use("Agg")  # A pure renderer backend, not compatible with plt.show().
print(f"Matplotlib backend: {matplotlib.get_backend()}")
from matplotlib import pyplot as plt


class PlotterBase:

    def __init__(self, bo: BayesianOptimizer, ):
        self.bo = bo
        self.figsize: Tuple[int, int] = (8, 7)

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


class PlotterBaseBak:
    """ A base class for plotting. """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            title: str | None = None,
            xlabel: str | None = None,
            ylabel: str | None = None,
            zlabel: str | None = None,
            xlim: Tuple[float, float] | None = None,
            ylim: Tuple[float, float] | None = None,
            zlim: Tuple[float, float] | None = None,
            cmap_label: str | None = None,
            labels: List[str] | None = None,
            lims: List[Tuple[float, float]] | None = None,
            figsize: Tuple[int, int] = (8, 7),
    ):
        self.bo = bayesian_optimizer
        self.title = title

        self.xlabel = xlabel
        self.ylable = ylabel
        self.zlabel = zlabel
        self.cmap_label = cmap_label

        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self._initialize_figure()

        self.labels = labels
        self.lims = lims
        self.figsize = figsize
        self.cbar: plt.Colorbar | None = None

        self._set_labels()
        self._set_x_lims()
        self._set_y_lims()
        self.legend_elements = []
        self._sampler: SamplerBase = None

    @property
    def sampler(self) -> SamplerBase:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: SamplerBase) -> None:
        self._sampler = sampler

    # def _initialize_figure(self):
    #     if self.title is not None:
    #         self.ax.set_title(self.title)
    #     if self.xlabel is not None:
    #         self.ax.set_xlabel(self.xlabel)
    #     if self.ylable is not None:
    #         self.ax.set_ylabel(self.ylable)
    #
    #     if self.labels is not None:
    #         self._set_labels()
    #
    # def _set_x_lims(self):
    #     if self.lims is not None and self.lims[0] is not None:
    #         self.ax.set_xlim(self.lims[0][0], self.lims[0][1])
    #     else:
    #         self.ax.autoscale(enable=True, axis='x')
    #
    # def _set_y_lims(self):
    #     if self.lims is not None and self.lims[1] is not None:
    #         self.ax.set_ylim(self.lims[1][0], self.lims[1][1])
    #     else:
    #         self.ax.autoscale(enable=True, axis='y')
    #
    # def _set_labels(self):
    #     if self.labels is not None:
    #         self.ax.set_xlabel(self.labels[0])
    #         self.ax.set_ylabel(self.labels[1])
    #     if self.cbar is not None:
    #         self.cbar.set_label(self.labels[2])

    def _generate_unif1orm_grid(self, n_points_per_dim=1000) -> torch.Tensor:
        """Generate a uniform dense grid over the input bounds for plotting."""
        bounds = self.bo.objective.bounds
        device = self.bo.device
        dtype = self.bo.dtype
        d = self.bo.objective.dim

        # Create linspaces for each dimension
        linspaces = [
            torch.linspace(
                start=bounds[0, i],
                end=bounds[1, i],
                steps=n_points_per_dim,
                device=device,
                dtype=dtype
            )
            for i in range(d)
        ]

        if d == 1:
            return linspaces[0].unsqueeze(-1)

        # For d > 1, create a meshgrid
        grids = torch.meshgrid(*linspaces, indexing='ij')
        # Stack and reshape to [N**d, d]
        X_grid = torch.stack(grids, dim=-1).reshape(-1, d)

        return X_grid

    # def _generate_random_X(self, n_points_per_dim=1000) -> torch.Tensor:
    # # TODO:
    # # """ Generate samples for ground truth evaluation - random sampler or grid """
    # # # When constraints apply to the input X, build the ground truth by using
    # # # a random generator subject to constraints
    # # X_gt = sampler.draw_samples(n=1000)

    # def save_figure(self, filename: str | Path | None = None):
    #     if filename is None:
    #         filename = self.title
    #
    #     path = Path(filename)
    #     stem = path.stem
    #     suffix = path.suffix
    #     save_path = Path.cwd() / path.parent / f"{stem}_000{suffix}"
    #
    #     i = 0
    #     while save_path.exists():
    #         i += 1
    #         save_path = Path.cwd() / path.parent / f"{stem}_{i:03d}{suffix}"
    #
    #     self.fig.savefig(fname=save_path, dpi=600)
    #     return self
    #
    # def close_figure(self):
    #     plt.close(self.fig)
    #
    # def show_figure(self):
    #     plt.show(self.fig)
