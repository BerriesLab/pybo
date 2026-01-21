from fileinput import filename
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase
from plotters.base_class import PlotterBase
from plotters.styles import xy_plot_kwargs, line2d_plot_kwargs


class BestValuePlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel("Best value")

    def plot(self):
        best_values = np.array(self.bo.best_values)
        x = self.bo.n_initial_samples + np.arange(len(best_values)) * self.bo.batch_size
        self.ax.plot(x, best_values, **line2d_plot_kwargs)

        if self.bo.objective.best_value is not None:
            self.ax.axhline(y=self.bo.objective.best_value, linestyle='--', color='black', label='Max HV')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "best_value.png"
        return super().save_figure(filename=filename)


class HypervolumePlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCMultiObjectiveBase):
            raise TypeError("Objective must be of type MCMultiObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel("Hypervolume")

    def plot(self):
        hv = np.array(self.bo.hypervolume)
        # X-axis: number of observations beyond the initial front
        x = self.bo.n_initial_samples + np.arange(len(hv)) * self.bo.batch_size

        self.ax.plot(x, hv, **line2d_plot_kwargs)
        if self.bo.objective.max_hv is not None:
            self.ax.axhline(y=self.bo.objective.max_hv, linestyle='--', color='black', label='Max HV')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"hv.png"
        return super().save_figure(filename=filename)


class HypervolumeImprovementPlotter(PlotterBase):
    def __init__(self, b0: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(b0.objective, MCMultiObjectiveBase):
            raise TypeError("Objective must be of type MCMultiObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel(r"$\log_{10}(\mathrm{HVI})$")

    def plot(self, epsilon: float = 1e-6):
        hv = self.bo.hypervolume

        hv = np.array(hv)
        hvi = np.diff(hv, prepend=np.nan)  # the first element has no diff

        # Clip small values to avoid log10(0)
        hvi_log = np.log10(np.clip(hvi, epsilon, None))

        # X-axis: number of observations beyond the initial front
        x = self.bo.n_initial_samples + np.arange(len(hv)) * self.bo.batch_size
        mask = ~np.isnan(hvi_log)
        self.ax.plot(x[mask], hvi_log[mask], **line2d_plot_kwargs)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"hvi.png"
        return super().save_figure(filename=filename)


class ElapsedTimePlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel("Elapsed Time")

    def plot(self):
        elapsed_time = self.bo.elapsed_time
        x = self.bo.n_initial_samples + np.arange(len(elapsed_time)) * self.bo.batch_size
        y = elapsed_time
        self.ax.plot(x, y, **line2d_plot_kwargs)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "elapsed_time.png"
        return super().save_figure(filename=filename)


class ParameterPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, idx: int):
        super().__init__(bo=bo)

        self.idx = idx
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel(
            bo.objective.parameter_names[idx] if bo.objective.parameter_names else f"Parameter {idx:02d}")

    def plot(self):
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(x_all, y_all,
                        **xy_plot_kwargs,
                        label=self.bo.objective.parameter_names[self.idx]
                        if self.bo.objective.parameter_names
                        else f"parameter {self.idx:02d}")
        self.ax.axhline(y=self.bo.objective.bounds[0][self.idx].detach().cpu().numpy(), linestyle='--', color='black')
        self.ax.axhline(y=self.bo.objective.bounds[1][self.idx].detach().cpu().numpy(), linestyle='--', color='black')
        self.ax.axvline(x=self.bo.n_initial_samples, linestyle='--', color='black')
        return self

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.bo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.bo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.bo.batch_size]
            else:
                idx_start = self.bo.n_initial_samples + i * self.bo.batch_size
                idx_end = idx_start + self.bo.batch_size
                y_batch_start = y_all[..., idx_start - self.bo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.bo.batch_size: idx_start]
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
        x_init = self.bo.n_initial_samples * np.ones(self.bo.n_initial_samples)
        y_init = self.bo.X[..., : self.bo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.bo.X.shape[0] - self.bo.n_initial_samples) / self.bo.batch_size)
        x_next_points = np.linspace(
            start=self.bo.n_initial_samples + self.bo.batch_size,
            stop=self.bo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.bo.batch_size)
        y_next = self.bo.X[..., self.bo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"parameter_{self.idx:02d}.png"
        return super().save_figure(filename=filename)


class ObjectivePlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, idx: int = 0):
        super().__init__(bo=bo)

        self.idx = idx
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel(
            bo.objective.objective_names[idx] if bo.objective.objective_names else f"Objective {idx:02d}")

    def plot(self):
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(x_all, y_all,
                        **xy_plot_kwargs,
                        label=self.bo.objective.objective_names[self.idx]
                        if self.bo.objective.objective_names
                        else f"objective {self.idx}"
                        )
        self.ax.axvline(x=self.bo.n_initial_samples, linestyle='--', color='black')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"objective_{self.idx:02d}.png"
        return super().save_figure(filename=filename)

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.bo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.bo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.bo.batch_size]
            else:
                idx_start = self.bo.n_initial_samples + i * self.bo.batch_size
                idx_end = idx_start + self.bo.batch_size
                y_batch_start = y_all[..., idx_start - self.bo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.bo.batch_size: idx_start]
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
        x_init = self.bo.n_initial_samples * np.ones(self.bo.n_initial_samples)
        y_init = self.bo.X[..., : self.bo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.bo.X.shape[0] - self.bo.n_initial_samples) / self.bo.batch_size)
        x_next_points = np.linspace(
            start=self.bo.n_initial_samples + self.bo.batch_size,
            stop=self.bo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.bo.batch_size)
        y_next = self.bo.X[..., self.bo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all


class ConstraintPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, idx: int = 0):
        super().__init__(bo=bo)

        self.idx = idx
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel(
            bo.objective.objective_names[idx] if bo.objective.objective_names else f"Constraint {idx:02d}")

    def plot(self):
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(
            x_all, y_all,
            **xy_plot_kwargs,
            label=self.bo.objective.constraint_names[self.idx]
            if self.bo.objective.constraint_names
            else f"constraint {self.idx}"
        )
        self.ax.axvline(x=self.bo.n_initial_samples, linestyle='--', color='black')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"constraint_{self.idx:02d}.png"
        return super().save_figure(filename=filename)

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.bo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.bo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.bo.batch_size]
            else:
                idx_start = self.bo.n_initial_samples + i * self.bo.batch_size
                idx_end = idx_start + self.bo.batch_size
                y_batch_start = y_all[..., idx_start - self.bo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.bo.batch_size: idx_start]
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
        x_init = self.bo.n_initial_samples * np.ones(self.bo.n_initial_samples)
        y_init = self.bo.X[..., : self.bo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.bo.X.shape[0] - self.bo.n_initial_samples) / self.bo.batch_size)
        x_next_points = np.linspace(
            start=self.bo.n_initial_samples + self.bo.batch_size,
            stop=self.bo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.bo.batch_size)
        y_next = self.bo.X[..., self.bo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all


class TrackerPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, idx: int = 0):
        super().__init__(bo=bo)

        self.idx = idx
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel(
            bo.objective.objective_names[idx] if bo.objective.objective_names else f"Tracker {idx:02d}")

    def plot(self):
        n_iter, x_all, y_all = self._prepare_xy()
        self._connect_subsequent_batches(n_iter, x_all, y_all)
        self.ax.scatter(x_all, y_all,
                        **xy_plot_kwargs,
                        label=self.bo.objective.tracker_names[self.idx]
                        if self.bo.objective.tracker_names
                        else f"tracker {self.idx}"
                        )
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"tracker_{self.idx:02d}.png"
        return super().save_figure(filename=filename)

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        for i in range(n_iter):
            if i == 0:
                idx_start = 0
                idx_end = self.bo.n_initial_samples
                y_batch_start = y_all[..., idx_start:idx_end]
                y_batch_end = y_all[..., idx_end: idx_end + self.bo.batch_size]
                x_batch_start = x_all[idx_start:idx_end]
                x_batch_end = x_all[idx_end: idx_end + self.bo.batch_size]
            else:
                idx_start = self.bo.n_initial_samples + i * self.bo.batch_size
                idx_end = idx_start + self.bo.batch_size
                y_batch_start = y_all[..., idx_start - self.bo.batch_size: idx_start]
                y_batch_end = y_all[..., idx_start: idx_end]
                x_batch_start = x_all[idx_start - self.bo.batch_size: idx_start]
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
        x_init = self.bo.n_initial_samples * np.ones(self.bo.n_initial_samples)
        y_init = self.bo.X[..., : self.bo.n_initial_samples, self.idx].detach().cpu().numpy()
        # Prepare the next set of scatter points
        n_iter = int((self.bo.X.shape[0] - self.bo.n_initial_samples) / self.bo.batch_size)
        x_next_points = np.linspace(
            start=self.bo.n_initial_samples + self.bo.batch_size,
            stop=self.bo.X.shape[0],
            num=n_iter,
        )
        x_next = np.repeat(a=x_next_points, repeats=self.bo.batch_size)
        y_next = self.bo.X[..., self.bo.n_initial_samples:, self.idx].detach().cpu().numpy()
        # Plot all scatter points
        x_all = np.concatenate((x_init, x_next))
        y_all = np.concatenate((y_init, y_next))
        return n_iter, x_all, y_all
