from pathlib import Path
import torch
import numpy as np
from typing import cast
from matplotlib import pyplot as plt
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.variable_registry import ParCfg
from plotters.base_class import PlotterBase
from plotters.styles import *


def plot_and_save_evolutions(bo: BayesianOptimizer, ):
    """ A helper functions to plot and sav all evolution plots. """
    for idx in range(bo.objective.num_par):
        EvolutionPlotter(bo=bo, y=("par", idx)).plot().save_figure().close_figure()
    for idx in range(bo.objective.num_obj):
        EvolutionPlotter(bo=bo, y=("obj", idx)).plot().save_figure().close_figure()
    for idx in range(bo.objective.num_con):
        EvolutionPlotter(bo=bo, y=("con", idx)).plot().save_figure().close_figure()
    for idx in range(bo.objective.num_trk):
        EvolutionPlotter(bo=bo, y=("trk", idx)).plot().save_figure().close_figure()


class EvolutionPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, y: tuple[str, str | int]):
        """
        Args:
            bo: The BayesianOptimizer instance.
            y: Tuple like ("par", "P1"), ("obj", 0), or ("trk", "Time").
        """
        super().__init__(bo=bo)

        raw_cfg = bo.objective.get_config(*y)
        self.category = y[0].lower()

        # 2. Select the correct data tensor based on category
        self.data_tensor = self._map_tensor(self.category)
        self.config = raw_cfg
        self.idx = raw_cfg.index

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel("Number of observations")
        self.ax.set_ylabel(self.config.label.capitalize())

    def _map_tensor(self, category: str) -> torch.Tensor:
        """ Maps the category string to the corresponding BO tensor. """
        mapping = {
            "par": self.bo.X,
            "obj": self.bo.Y_obj,
            "con": self.bo.Y_con,
            "trk": self.bo.Y_track
        }
        tensor = mapping.get(category)
        if tensor is None:
            raise ValueError(f"No tensor found for category: {category}")
        return tensor

    def _prepare_xy(self):
        if self.data_tensor is None or self.data_tensor.numel() == 0:
            return 0, np.array([]), np.array([])

        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size
        # Extract the specific column for this parameter/objective
        y_raw = self.data_tensor[..., self.idx].detach().cpu().numpy()

        # Handle X-axis coordinates logic
        x_init = n_init * np.ones(n_init)
        n_bo_points = len(y_raw) - n_init

        if n_bo_points <= 0:
            return 0, x_init, y_raw

        n_iter = n_bo_points // q
        x_next_steps = np.linspace(n_init + q, n_init + (n_iter * q), n_iter)
        x_next = np.repeat(x_next_steps, q)

        return n_iter, np.concatenate([x_init, x_next]), y_raw

    def _connect_subsequent_batches(self, n_iter, x_all, y_all):
        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size

        for i in range(n_iter):
            # Connect initial points to first batch, then batch to batch
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

            # Vertical line showing end of initialisation
            self.ax.axvline(x=self.bo.n_initial_samples, **general_axline)

            # Plot horizontal bounds if they exist (Common for parameters/constraints)
            # Use 'cast' so the IDE sees the bounds attribute
            cfg_with_bounds = cast(ParCfg, self.config)
            if hasattr(cfg_with_bounds, 'bounds') and cfg_with_bounds.bounds:
                self.ax.axhline(y=cfg_with_bounds.bounds[0], color='r', linestyle='--', alpha=0.5)
                self.ax.axhline(y=cfg_with_bounds.bounds[1], color='r', linestyle='--', alpha=0.5)

        return self

    def save_figure(self, filename: str | Path | None = None):
        clean_label = self.config.label.lower().replace(" ", "_")
        filename = filename or f"evolution_{self.category}_{clean_label}.png"
        return super().save_figure(filename=filename)
