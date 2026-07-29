from pathlib import Path
import matplotlib
import torch
from pybo.optimizer.optimizer import BayesianOptimizer
from pybo.plotters.style import ensure_resolved, fig_cfg
from matplotlib import pyplot as plt

matplotlib.use("Agg")  # A pure renderer backend, not compatible with plt.show().
print(f"Matplotlib backend: {matplotlib.get_backend()}")


def save_figure(fig, filename: str | Path):
    """Write `fig` next to the cwd under the resolved file type.

    Each caller picks the stem; the extension comes from the settings, so the file type
    is decided in one place. Free-standing because the studies/analysis scripts need it
    too and they have no BayesianOptimizer to hang a PlotterBase off.
    """
    ensure_resolved()
    path = Path(filename).with_suffix("." + fig_cfg["format"])
    save_path = Path.cwd() / path.parent / path.name

    # tight_layout cannot place a colorbar's axes and would draw it over the plot, so a
    # figure that brought its own layout engine (constrained, say) keeps it.
    if fig.get_layout_engine() is None:
        fig.tight_layout(pad=fig_cfg["layout_pad"])
    fig.savefig(fname=save_path, dpi=fig_cfg["dpi"])
    return save_path


class PlotterBase:
    # Which entry of fig_cfg["figsize"] this plot is sized by. Subclasses set it; the
    # aspect ratio behind it lives in pybo/plotters/styles/defaults.py, and the
    # physical width comes from the active publisher style.
    plot_name: str = "experiment_1d"

    def __init__(self, bo: BayesianOptimizer):
        # No-op under the CLI, which has already resolved with --plot-style; this covers a
        # notebook or a test that builds a plotter without going through argparse.
        self.fig = None
        ensure_resolved()
        self.bo = bo
        self.figsize = fig_cfg["figsize"][self.plot_name]
        self.dpi = fig_cfg["dpi"]
        self.n_grid_points: int = 500

    def save_figure(self, filename: str | Path):
        # Fixed filename: the per-step folder gives uniqueness, mirroring
        # BayesianOptimizer.to_csv(latest=True). The counter this replaced dated
        # from the flat run directory, where it was the only thing separating one
        # step's figures from the next.
        save_figure(self.fig, filename)
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
