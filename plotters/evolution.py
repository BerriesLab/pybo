import numpy as np
from matplotlib import pyplot as plt

from mobo.mobo import Mobo
from plotters.base_class import PlotterBase
from plotters.utils import xy_plot_kwargs


class HypervolumePlotter(PlotterBase):
    def __init__(self, mobo: Mobo):
        super().__init__(
            title="Hypervolume",
            labels=["Hypervolume", "Number of observations"]
        )
        self.mobo = mobo

    def plot(self, epsilon: float = 1e-6):

        hv = self.mobo.hypervolume

        if len(hv) <= 1:
            print("Not enough data to plot.")
            return

        hv = np.array(hv)
        # X-axis: number of observations beyond initial front
        x = np.arange(len(hv)) * self.mobo.batch_size

        self.ax.plot(x, hv, **xy_plot_kwargs)
        if self.mobo.objective.max_hv is not None:
            self.ax.axhline(y=self.mobo.objective.max_hv, linestyle='--', color='black', label='Max HV')
        return self


class HypervolumeImprovementPlotter(PlotterBase):
    def __init__(self, mobo: Mobo):
        super().__init__(
            title="Hypervolume Improvement",
            labels=["Number of observations", r"$\log_{10}(\mathrm{HVI})$"]
        )
        self.mobo = mobo

    def plot(self):
        hv = self.mobo.hypervolume

        if len(hv) <= 1:
            print("Not enough data to plot.")
            return None

        hv = np.array(hv)
        hvi = np.diff(hv, prepend=np.nan)  # first element has no diff

        # Clip small values to avoid log10(0)
        hvi_log = np.log10(np.clip(hvi, 1e-6, None))

        # X-axis: number of observations beyond initial front
        x = np.arange(len(hv)) * self.mobo.batch_size
        mask = ~np.isnan(hvi_log)
        self.ax.plot(x[mask], hvi_log[mask], **xy_plot_kwargs)
        return self


class ElapsedTimePlotter(PlotterBase):
    def __init__(self, mobo: Mobo):
        super().__init__(
            title="Elapsed Time",
            labels=["Number of observations (beyond initial points)", "Elapsed Time (s)"]
        )
        self.mobo = mobo

    def plot(self):
        elapsed_time = self.mobo.elapsed_time
        x = np.array(range(len(elapsed_time))) * self.mobo.batch_size
        y = elapsed_time
        self.ax.plot(x, y, **xy_plot_kwargs)
        return self


# TODO: implement as a loop over PlotterBase classes
def plot_parameters_evolution(
        mobo,
        parameter_names: list[str] | None = None,
        display_figure: bool = False,
        output_path: Path | None = None,
):
    X = mobo.X.clone().cpu().numpy()
    n = mobo.objective.dim
    bounds = mobo.objective.bounds.clone().cpu().numpy()
    if parameter_names is None:
        if mobo.objective.parameter_names is not None:
            parameter_names = mobo.objective.parameter_names
        else:
            parameter_names = [f"p{i}" for i in range(n)]

    for i in range(n):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(X.shape[0]), X[:, i], **xy_plot_kwargs, label=parameter_names[i])
        plt.axhline(y=bounds[0][i], linestyle='--', color='black')
        plt.axhline(y=bounds[1][i], linestyle='--', color='black')
        plt.xlabel("Iteration")
        plt.ylabel(parameter_names[i])
        plt.title(f"Evolution of {parameter_names[i]}")
        plt.legend()
        plt.tight_layout()

        save_path = output_path or (
                Path.cwd() / f"parameter_evolution_{parameter_names[i].replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=300)

        if display_figure:
            plt.show()
        plt.close(fig)


class ParametersPlotter:
    def __init__(self, mobo: Mobo):
        self.mobo = mobo
        for i in range(self.mobo.objective.dim):
            plotter = PlotterBase(
                title=self.mobo.objective.parameter_names[i],
                labels=[
                    "Number of observations (beyond initial points)",
                    self.mobo.objective.parameter_names[i]
                ]
            )
            plotter.ax.plot(
                range(self.mobo.X.shape[0]),
                self.mobo.X[:, i],
                **xy_plot_kwargs,
                label=self.mobo.objective.parameter_names[i]
            )
            plotter.save_figure()
            plotter.close_figure()
