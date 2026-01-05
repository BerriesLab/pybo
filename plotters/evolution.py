from pathlib import Path
import numpy as np
from mobo.bayesian_optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase
from plotters.utils import xy_plot_kwargs, line2d_plot_kwargs


class HypervolumePlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer):
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
        x = self.mobo.n_initial_samples + np.arange(len(hv)) * self.mobo.batch_size

        self.ax.plot(x, hv, **line2d_plot_kwargs)
        if self.mobo.objective.max_hv is not None:
            self.ax.axhline(y=self.mobo.objective.max_hv, linestyle='--', color='black', label='Max HV')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_metric_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class HypervolumeImprovementPlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer):
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
        x = self.mobo.n_initial_samples + np.arange(len(hv)) * self.mobo.batch_size
        mask = ~np.isnan(hvi_log)
        self.ax.plot(x[mask], hvi_log[mask], **line2d_plot_kwargs)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_metric_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ElapsedTimePlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer):
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
        x = self.mobo.n_initial_samples + np.arange(len(elapsed_time)) * self.mobo.batch_size
        y = elapsed_time
        self.ax.plot(x, y, **line2d_plot_kwargs)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_metric_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ParameterPlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer, idx: int):
        super().__init__(
            title=f"{mobo.objective.parameter_names[idx]}"
            if mobo.objective.parameter_names
            else f"parameter {idx}",
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
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(x_all, y_all,
                        **xy_plot_kwargs,
                        label=self.mobo.objective.parameter_names[self.idx]
                        if self.mobo.objective.parameter_names
                        else f"parameter {self.idx}")
        self.ax.axhline(y=self.mobo.objective.bounds[0][self.idx].detach().cpu().numpy(), linestyle='--', color='black')
        self.ax.axhline(y=self.mobo.objective.bounds[1][self.idx].detach().cpu().numpy(), linestyle='--', color='black')
        self.ax.axvline(x=self.mobo.n_initial_samples, linestyle='--', color='black')
        return self

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.mobo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.mobo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.mobo.batch_size]
            else:
                idx_start = self.mobo.n_initial_samples + i * self.mobo.batch_size
                idx_end = idx_start + self.mobo.batch_size
                y_batch_start = y_all[..., idx_start - self.mobo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.mobo.batch_size: idx_start]
                x_batch_end = x_all[idx_start: idx_end]

            # Plot all combinations of points from the two batches
            for col_start_idx in range(y_batch_start.shape[-1]):
                for col_end_idx in range(y_batch_end.shape[-1]):
                    y1 = y_batch_start[..., col_start_idx]
                    y2 = y_batch_end[..., col_end_idx]
                    x1 = x_batch_start[col_start_idx]
                    x2 = x_batch_end[col_end_idx]
                    self.ax.plot([x1, x2], [y1, y2],
                                 linestyle='-',
                                 color='gray',
                                 alpha=0.3)

    def _prepare_xy(self):
        # Prepare the initial scatter points
        x_init = self.mobo.n_initial_samples * np.ones(self.mobo.n_initial_samples)
        y_init = self.mobo.X[..., : self.mobo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.mobo.X.shape[0] - self.mobo.n_initial_samples) / self.mobo.batch_size)
        x_next_points = np.linspace(
            start=self.mobo.n_initial_samples + self.mobo.batch_size,
            stop=self.mobo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.mobo.batch_size)
        y_next = self.mobo.X[..., self.mobo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_parameter_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)


class ObjectivePlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer, idx: int):
        super().__init__(
            title=mobo.objective.objective_names[idx]
            if mobo.objective.objective_names is not None
            else f"objective {idx}",
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
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(x_all, y_all,
                        **xy_plot_kwargs,
                        label=self.mobo.objective.objective_names[self.idx]
                        if self.mobo.objective.objective_names
                        else f"objective {self.idx}"
                        )
        self.ax.axvline(x=self.mobo.n_initial_samples, linestyle='--', color='black')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_objective_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.mobo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.mobo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.mobo.batch_size]
            else:
                idx_start = self.mobo.n_initial_samples + i * self.mobo.batch_size
                idx_end = idx_start + self.mobo.batch_size
                y_batch_start = y_all[..., idx_start - self.mobo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.mobo.batch_size: idx_start]
                x_batch_end = x_all[idx_start: idx_end]

            # Plot all combinations of points from the two batches
            for col_start_idx in range(y_batch_start.shape[-1]):
                for col_end_idx in range(y_batch_end.shape[-1]):
                    y1 = y_batch_start[..., col_start_idx]
                    y2 = y_batch_end[..., col_end_idx]
                    x1 = x_batch_start[col_start_idx]
                    x2 = x_batch_end[col_end_idx]
                    self.ax.plot([x1, x2], [y1, y2],
                                 linestyle='-',
                                 color='gray',
                                 alpha=0.3)

    def _prepare_xy(self):
        # Prepare the initial scatter points
        x_init = self.mobo.n_initial_samples * np.ones(self.mobo.n_initial_samples)
        y_init = self.mobo.X[..., : self.mobo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.mobo.X.shape[0] - self.mobo.n_initial_samples) / self.mobo.batch_size)
        x_next_points = np.linspace(
            start=self.mobo.n_initial_samples + self.mobo.batch_size,
            stop=self.mobo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.mobo.batch_size)
        y_next = self.mobo.X[..., self.mobo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all


class ConstraintPlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer, idx: int):
        super().__init__(
            title=mobo.objective.constraint_names[idx]
            if mobo.objective.constraint_names
            else f"constraint {idx}",
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
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(
            x_all, y_all,
            **xy_plot_kwargs,
            label=self.mobo.objective.constraint_names[self.idx]
            if self.mobo.objective.constraint_names
            else f"constraint {self.idx}"
        )
        self.ax.axvline(x=self.mobo.n_initial_samples, linestyle='--', color='black')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_constraint_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.mobo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.mobo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.mobo.batch_size]
            else:
                idx_start = self.mobo.n_initial_samples + i * self.mobo.batch_size
                idx_end = idx_start + self.mobo.batch_size
                y_batch_start = y_all[..., idx_start - self.mobo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.mobo.batch_size: idx_start]
                x_batch_end = x_all[idx_start: idx_end]

            # Plot all combinations of points from the two batches
            for col_start_idx in range(y_batch_start.shape[-1]):
                for col_end_idx in range(y_batch_end.shape[-1]):
                    y1 = y_batch_start[..., col_start_idx]
                    y2 = y_batch_end[..., col_end_idx]
                    x1 = x_batch_start[col_start_idx]
                    x2 = x_batch_end[col_end_idx]
                    self.ax.plot([x1, x2], [y1, y2],
                                 linestyle='-',
                                 color='gray',
                                 alpha=0.3)

    def _prepare_xy(self):
        # Prepare the initial scatter points
        x_init = self.mobo.n_initial_samples * np.ones(self.mobo.n_initial_samples)
        y_init = self.mobo.X[..., : self.mobo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.mobo.X.shape[0] - self.mobo.n_initial_samples) / self.mobo.batch_size)
        x_next_points = np.linspace(
            start=self.mobo.n_initial_samples + self.mobo.batch_size,
            stop=self.mobo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.mobo.batch_size)
        y_next = self.mobo.X[..., self.mobo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all


class TrackerPlotter(PlotterBase):
    def __init__(self, mobo: BayesianOptimizer, idx: int):
        super().__init__(
            title=mobo.objective.tracker_names[idx]
            if mobo.objective.tracker_names
            else f"tracker {idx}",
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
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(x_all, y_all,
                        **xy_plot_kwargs,
                        label=self.mobo.objective.tracker_names[self.idx]
                        if self.mobo.objective.tracker_names
                        else f"tracker {self.idx}"
                        )
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = f"trajectory_tracker_{self.title.replace(" ", "_").lower()}.png"
        return super().save_figure(filename=filename)

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.mobo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.mobo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.mobo.batch_size]
            else:
                idx_start = self.mobo.n_initial_samples + i * self.mobo.batch_size
                idx_end = idx_start + self.mobo.batch_size
                y_batch_start = y_all[..., idx_start - self.mobo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.mobo.batch_size: idx_start]
                x_batch_end = x_all[idx_start: idx_end]

            # Plot all combinations of points from the two batches
            for col_start_idx in range(y_batch_start.shape[-1]):
                for col_end_idx in range(y_batch_end.shape[-1]):
                    y1 = y_batch_start[..., col_start_idx]
                    y2 = y_batch_end[..., col_end_idx]
                    x1 = x_batch_start[col_start_idx]
                    x2 = x_batch_end[col_end_idx]
                    self.ax.plot([x1, x2], [y1, y2],
                                 linestyle='-',
                                 color='gray',
                                 alpha=0.3)

    def _prepare_xy(self):
        # Prepare the initial scatter points
        x_init = self.mobo.n_initial_samples * np.ones(self.mobo.n_initial_samples)
        y_init = self.mobo.X[..., : self.mobo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.mobo.X.shape[0] - self.mobo.n_initial_samples) / self.mobo.batch_size)
        x_next_points = np.linspace(
            start=self.mobo.n_initial_samples + self.mobo.batch_size,
            stop=self.mobo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.mobo.batch_size)
        y_next = self.mobo.X[..., self.mobo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all
