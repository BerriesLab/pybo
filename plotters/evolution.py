from pathlib import Path

import numpy as np
from mobo.mobo import Mobo
from plotters.base_class import PlotterBase
from plotters.utils import xy_plot_kwargs


class HypervolumePlotter(PlotterBase):
    def __init__(self, mobo: Mobo):
        super().__init__(
            title="Hypervolume",
            labels=[
                "Number of observations (beyond initial points)",
                "Hypervolume",
            ]
        )
        self.mobo = mobo

    def plot(self):
        hv = self.mobo.hypervolume

        hv = np.array(hv)

        # X-axis: number of observations beyond initial front
        x = np.arange(len(hv)) * self.mobo.batch_size

        self.ax.plot(x, hv, **xy_plot_kwargs)
        if self.mobo.objective.max_hv is not None:
            self.ax.axhline(y=self.mobo.objective.max_hv, linestyle='--', color='black', label='Max HV')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_metric_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class HypervolumeImprovementPlotter(PlotterBase):
    def __init__(self, mobo: Mobo):
        super().__init__(
            title="Hypervolume Improvement",
            labels=[
                "Number of observations (beyond initial points)",
                r"$\log_{10}(\mathrm{HVI})$"
            ]
        )
        self.mobo = mobo

    def plot(self, epsilon: float = 1e-6):
        hv = self.mobo.hypervolume

        hv = np.array(hv)
        hvi = np.diff(hv, prepend=np.nan)  # first element has no diff

        # Clip small values to avoid log10(0)
        hvi_log = np.log10(np.clip(hvi, epsilon, None))

        # X-axis: number of observations beyond initial front
        x = np.arange(len(hv)) * self.mobo.batch_size
        mask = ~np.isnan(hvi_log)
        self.ax.plot(x[mask], hvi_log[mask], **xy_plot_kwargs)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_metric_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ElapsedTimePlotter(PlotterBase):
    def __init__(self, mobo: Mobo):
        super().__init__(
            title="Elapsed Time",
            labels=[
                "Number of observations (beyond initial points)",
                "Elapsed Time (s)"
            ]
        )
        self.mobo = mobo

    def plot(self):
        elapsed_time = self.mobo.elapsed_time
        x = np.array(range(len(elapsed_time))) * self.mobo.batch_size
        y = elapsed_time
        self.ax.plot(x, y, **xy_plot_kwargs)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_metric_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ParameterPlotter(PlotterBase):
    def __init__(self, mobo: Mobo, idx: int):
        super().__init__(
            title=f"{mobo.objective.parameter_names[idx]}",
            labels=[
                "Number of observations (beyond initial points)",
                mobo.objective.parameter_names[idx]
                if mobo.objective.parameter_names
                else f"parameter {idx}"
            ]
        )
        self.mobo = mobo
        self.idx = idx

    def plot(self):
        self.ax.plot(
            range(self.mobo.X.shape[0] * self.mobo.batch_size),
            self.mobo.X[:, self.idx],
            **xy_plot_kwargs,
            label=self.mobo.objective.parameter_names[self.idx]
            if self.mobo.objective.parameter_names
            else f"parameter {self.idx}"
        )
        self.ax.axhline(y=self.mobo.objective.bounds[0][self.idx], linestyle='--', color='black')
        self.ax.axhline(y=self.mobo.objective.bounds[1][self.idx], linestyle='--', color='black')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_parameter_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ObjectivePlotter(PlotterBase):
    def __init__(self, mobo: Mobo, idx: int):
        super().__init__(
            title=mobo.objective.objective_names[idx],
            labels=[
                "Number of observations (beyond initial points)",
                mobo.objective.objective_names[idx]
                if mobo.objective.objective_names
                else f"objective {idx}"
            ]
        )
        self.mobo = mobo
        self.idx = idx

    def plot(self):
        self.ax.plot(
            range(self.mobo.X.shape[0] * self.mobo.batch_size),
            self.mobo.Y_obj[:, self.idx],
            **xy_plot_kwargs,
            label=self.mobo.objective.objective_names[self.idx]
            if self.mobo.objective.objective_names
            else f"objective {self.idx}"
        )
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_objective_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ConstraintPlotter(PlotterBase):
    def __init__(self, mobo: Mobo, idx: int):
        super().__init__(
            title=mobo.objective.constraint_names[idx],
            labels=[
                "Number of observations (beyond initial points)",
                mobo.objective.constraint_names[idx]
                if mobo.objective.constraint_names
                else f"constraint {idx}"
            ]
        )
        self.mobo = mobo
        self.idx = idx

    def plot(self):
        self.ax.plot(
            range(self.mobo.X.shape[0] * self.mobo.batch_size),
            self.mobo.Y_con[:, self.idx],
            **xy_plot_kwargs,
            label=self.mobo.objective.constraint_names[self.idx]
            if self.mobo.objective.constraint_names
            else f"constraint {self.idx}"
        )
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_constraint_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class TrackerPlotter(PlotterBase):
    def __init__(self, mobo: Mobo, idx: int):
        super().__init__(
            title=mobo.objective.tracker_names[idx],
            labels=[
                "Number of observations (beyond initial points)",
                mobo.objective.tracker_names[idx]
                if mobo.objective.tracker_names
                else f"tracker {idx}"
            ]
        )
        self.mobo = mobo
        self.idx = idx

    def plot(self):
        self.ax.plot(
            range(self.mobo.X.shape[0] * self.mobo.batch_size),
            self.mobo.Y_track[:, self.idx],
            **xy_plot_kwargs,
            label=self.mobo.objective.tracker_names[self.idx]
            if self.mobo.objective.tracker_names
            else f"tracker {self.idx}"
        )
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_tracker_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)
