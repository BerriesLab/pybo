from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase
from plotters.base_class import PlotterBase
from plotters.styles import *


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
        self.ax.plot(x, best_values, **metrics_line2d)
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

        self.ax.plot(x, hv, **metrics_line2d)
        if self.bo.objective.max_hv is not None:
            self.ax.axhline(y=self.bo.objective.max_hv, linestyle='--', color='black', label='Max HV')
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or f"hv.png"
        return super().save_figure(filename=filename)


class HypervolumeImprovementPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCMultiObjectiveBase):
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
        self.ax.plot(x[mask], hvi_log[mask], **metrics_line2d)
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
        self.ax.plot(x, y, **metrics_line2d)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "elapsed_time.png"
        return super().save_figure(filename=filename)
