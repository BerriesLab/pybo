import torch
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import axvline

from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.variable_registry import VariableRegistry
from plotters.base_class import PlotterBase
from plotters.styles import *


class EvolutionPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, data_obj: VariableRegistry, data_tensor: torch.Tensor, prefix: str):
        super().__init__(bo=bo)
        self.config = data_obj.cfg
        self.idx = self.config.index
        self.data_tensor = data_tensor
        self.prefix = prefix

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations (beyond initial points)")
        self.ax.set_ylabel(self.config.label or f"{prefix.capitalize()} {self.idx:02d}")

    def _prepare_xy(self):
        if self.data_tensor is None:
            return 0, np.array([]), np.array([])

        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size
        y_raw = self.data_tensor[..., self.idx].detach().cpu().numpy()

        # X-coords: Initial points at n_init, BO points at their respective counts
        x_init = n_init * np.ones(n_init)
        n_bo_points = len(y_raw) - n_init
        n_iter = n_bo_points // q

        x_next_steps = np.linspace(n_init + q, len(y_raw), n_iter)
        x_next = np.repeat(x_next_steps, q)

        return n_iter, np.concatenate([x_init, x_next]), y_raw

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size

        for i in range(n_iter):
            # Slicing logic for "web" connections
            start, mid = (0, n_init) if i == 0 else (n_init + (i - 1) * q, n_init + i * q)
            end = n_init + (i + 1) * q

            y_start, y_end = y_all[start:mid], y_all[mid:end]
            x_start, x_end = x_all[start:mid], x_all[mid:end]

            for y1, x1 in zip(y_start, x_start):
                for y2, x2 in zip(y_end, x_end):
                    self.ax.plot([x1, x2], [y1, y2], **evolution_interconnections)

    def plot(self):
        n_iter, x_all, y_all = self._prepare_xy()
        if len(y_all) > 0:
            self._connect_subsequent_batches(n_iter, x_all, y_all)
            self.ax.scatter(x_all, y_all, label=self.config.label, **evolution_scatter)
            self.ax.axvline(x=self.bo.n_initial_samples, **general_axline)
            if hasattr(self.config, 'bounds') and self.config.bounds:
                self.ax.axhline(y=self.config.bounds[0], **general_axline)
                self.ax.axhline(y=self.config.bounds[1], **general_axline)
            return self
        return self

    def save_figure(self, filename: str | Path | None = None):
        clean_label = self.config.label.lower().replace(" ", "_")
        filename = filename or f"{self.prefix}_{clean_label}.png"
        return super().save_figure(filename=filename)


class ParameterEvolution(EvolutionPlotter):
    def __init__(self, bo: BayesianOptimizer, par: VariableRegistry):
        super().__init__(bo, par, bo.X, "parameter")


class ObjectiveEvolution(EvolutionPlotter):
    def __init__(self, bo: BayesianOptimizer, obj: VariableRegistry):
        super().__init__(bo, obj, bo.Y_obj, "objective")


class ConstraintEvolution(EvolutionPlotter):
    def __init__(self, bo: BayesianOptimizer, con: VariableRegistry):
        super().__init__(bo, con, bo.Y_con, "constraint")


class TrackerEvolution(EvolutionPlotter):
    def __init__(self, bo: BayesianOptimizer, trk: VariableRegistry):
        super().__init__(bo, trk, bo.Y_track, "tracker")
