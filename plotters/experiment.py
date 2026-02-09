import torch
from typing import TypeAlias

import numpy as np
from botorch.utils.multi_objective import is_non_dominated
from matplotlib import pyplot as plt
from pathlib import Path

from matplotlib.colors import LogNorm
from sympy import false

from optimizer.optimizer import BayesianOptimizer
from objectives.variable_registry import *
from plotters.base_class import PlotterBase
from plotters.styles import *
from objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase

from samplers.samplers import SobolSampler
from utils.helpers import project_linear_equalities

AxisSpec: TypeAlias = tuple[str, str | int, bool]  # (kind, id, use_log)


class Experiment1DPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, x: tuple[str, str | int] = ("par", 0),
                 y: tuple[str, str | int] = ("obj", 0), z: tuple[str, str | int] | None = None,
                 cmap='coolwarm', grid=True, seed=None):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.x_cfg = bo.objective.get_config(*x)
        self.y_cfg = bo.objective.get_config(*y)
        self.z_cfg = bo.objective.get_config(*z) if z else None

        self.cmap = cmap
        self.n_grid_points = 100000
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.mappable = None
        self.cbar = None
        self.grid = grid
        self.seed = seed
        self.vmin, self.vmax = None, None

        self.ax.set_xlabel(self.x_cfg.label.capitalize())
        self.ax.set_ylabel(self.y_cfg.label.capitalize())

        if hasattr(self.x_cfg, 'bounds') and self.x_cfg.bounds is not None:
            low, high = self.x_cfg.bounds
            padding = (high - low) * 0.05
            self.ax.set_xlim(low - padding, high + padding)
        if hasattr(self.y_cfg, 'bounds') and self.y_cfg.bounds is not None:
            low, high = self.y_cfg.bounds
            padding = (high - low) * 0.05
            self.ax.set_ylim(low - padding, high + padding)
        if hasattr(self.z_cfg, 'bounds') and self.z_cfg.bounds is not None:
            self.vmin, self.vmax = self.z_cfg.bounds

    @staticmethod
    def _get_data(cfg, X_gt, Y_obj, Y_con, Y_trk):
        """ Extracts the correct column from the correct tensor (Identical to 2D). """
        if cfg is None: return None
        if isinstance(cfg, ParCfg): return X_gt[..., cfg.index]
        if isinstance(cfg, ObjCfg): return Y_obj[..., cfg.index]
        if isinstance(cfg, TrkCfg): return Y_trk[..., cfg.index]
        if isinstance(cfg, IneqYConCfg): return Y_con[..., cfg.index]
        raise TypeError(f"Unrecognised configuration type: {type(cfg)}")

    def plot_ground_truth(self, zorder: int = 10):
        # 1. Generate Samples
        if self.grid:
            X_gt = self._generate_uniform_grid()
        else:
            sampler = SobolSampler(
                device=self.bo.device,
                dtype=self.bo.dtype,
                objective=self.bo.objective,
                seed=self.seed
            )
            X_gt = sampler.draw_samples(self.n_grid_points)

        Y_obj = self.bo.objective.evaluate_true_objective(X_gt)
        Y_con = self.bo.objective.evaluate_true_constraint(X_gt)
        Y_trk = self.bo.objective.evaluate_tracker(X_gt)

        # 2. Slice Data
        x_vals = self._get_data(self.x_cfg, X_gt, Y_obj, Y_con, Y_trk)
        y_vals = self._get_data(self.y_cfg, X_gt, Y_obj, Y_con, Y_trk)
        z_vals = self._get_data(self.z_cfg, X_gt, Y_obj, Y_con, Y_trk)

        # 3. Feasibility Masking
        Y_full = torch.cat([Y_obj, Y_con], dim=-1) if Y_con is not None else Y_obj
        is_feasible = self.bo.objective.is_Y_feasible(Y_full)
        is_infeasible = torch.logical_not(is_feasible)

        is_best = torch.zeros_like(is_feasible, dtype=torch.bool)
        feasible_indices = torch.where(is_feasible)[0]
        if feasible_indices.numel() > 0:
            Y_max_space = Y_obj[is_feasible].clone()
            Y_max_space[..., self.bo.objective.to_minimize] *= -1
            is_best_idx = torch.argmax(Y_max_space)
            is_best[feasible_indices[is_best_idx]] = True

        is_exclusive_feasible = torch.logical_and(is_feasible, torch.logical_not(is_best))

        # === Render Layers ===

        # Layer 1: Infeasible
        if is_infeasible.any():
            kwargs = experiment_scatter_gnd_truth_infeasible.copy()
            if z_vals is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_vals[is_infeasible].cpu(),
                y=y_vals[is_infeasible].cpu(),
                c=z_vals[is_infeasible].cpu() if z_vals is not None else None,
                cmap=self.cmap if z_vals is not None else None,
                vmin=self.vmin,
                vmax=self.vmax,
                zorder=zorder,
                **kwargs
            )

        # Layer 2: Feasible but not best
        if is_feasible.any():
            kwargs = experiment_scatter_gnd_truth_feasible.copy()
            if z_vals is not None: kwargs.pop("facecolor")
            scatter = self.ax.scatter(
                x=x_vals[is_exclusive_feasible].cpu(),
                y=y_vals[is_exclusive_feasible].cpu(),
                c=z_vals[is_exclusive_feasible].cpu() if z_vals is not None else None,
                cmap=self.cmap if z_vals is not None else None,
                vmin=self.vmin,
                vmax=self.vmax,
                zorder=zorder + 1,
                **kwargs
            )
            if self.mappable is None:
                self.mappable = scatter

        # Layer 3: Best
        if is_best.any():
            kwargs = experiment_scatter_gnd_truth_best_value.copy()
            if z_vals is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_vals[is_best].cpu(),
                y=y_vals[is_best].cpu(),
                c=z_vals[is_best].cpu() if z_vals is not None else None,
                cmap=self.cmap if z_vals is not None else None,
                vmin=self.vmin,
                vmax=self.vmax,
                zorder=zorder + 2,
                **kwargs
            )

        return self

    def plot_gp_posterior(self, zorder: int = 0):
        if self.bo.model is None or not isinstance(self.y_cfg, ObjCfg):
            return self

        if self.grid:
            X = self._generate_uniform_grid()
        else:
            sampler = SobolSampler(
                device=self.bo.device,
                dtype=self.bo.dtype,
                objective=self.bo.objective,
                seed=self.seed
            )
            X = sampler.draw_samples(self.n_grid_points)

        with torch.no_grad():
            posterior = self.bo.model.posterior(X)
            mean = posterior.mean[..., self.y_cfg.index].squeeze()
            std = posterior.variance.sqrt()[..., self.y_cfg.index].squeeze()

        X_np = X.squeeze().cpu().numpy()
        m_np, s_np = mean.cpu().numpy(), std.cpu().numpy()

        # === Render Layers ===

        # Layer 1: GP Mean
        self.ax.plot(
            X_np,
            m_np,
            zorder=zorder + 3,
            **gp_mean
        )

        # Layer 2: GP 1 * sigma
        self.ax.fill_between(
            x=X_np,
            y1=m_np - s_np,
            y2=m_np + s_np,
            zorder=zorder + 2,
            **gp_confidence_interval_1sigma
        )

        # Layer 2: GP 2 * sigma
        self.ax.fill_between(
            x=X_np,
            y1=m_np - 2 * s_np,
            y2=m_np + 2 * s_np,
            zorder=zorder + 1,
            **gp_confidence_interval_2sigma
        )

        # Layer 2: GP 3 * sigma
        self.ax.fill_between(
            x=X_np,
            y1=m_np - 3 * s_np,
            y2=m_np + 3 * s_np,
            zorder=zorder,
            **gp_confidence_interval_3sigma
        )

        return self

    def plot_observations(self, zorder: int = 20):
        if self.bo.X is None: return self

        X, Y_obj = self.bo.X, self.bo.Y_obj
        Y_con, Y_track = self.bo.Y_con, self.bo.Y_trk

        x_obs = self._get_data(self.x_cfg, X, Y_obj, Y_con, Y_track)
        y_obs = self._get_data(self.y_cfg, X, Y_obj, Y_con, Y_track)
        z_obs = self._get_data(self.z_cfg, X, Y_obj, Y_con, Y_track)

        Y_full = torch.cat([Y_obj, Y_con], dim=-1) if Y_con is not None else Y_obj
        is_feasible = self.bo.objective.is_Y_feasible(Y_full)
        is_infeasible = torch.logical_not(is_feasible)

        # Identify Best
        is_best = torch.zeros_like(is_feasible, dtype=torch.bool)
        feasible_indices = torch.where(is_feasible)[0]
        if feasible_indices.numel() > 0:
            Y_max_space = Y_obj[is_feasible].clone()
            Y_max_space[..., self.bo.objective.to_minimize] *= -1
            is_best_idx = torch.argmax(Y_max_space)
            is_best[feasible_indices[is_best_idx]] = True

        is_exclusive_feasible = torch.logical_and(is_feasible, torch.logical_not(is_best))

        if is_infeasible.any():
            kwargs = experiment_scatter_observations_infeasible.copy()
            if z_obs is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_obs[is_infeasible].cpu(),
                y=y_obs[is_infeasible].cpu(),
                c=z_obs[is_infeasible].cpu() if z_obs is not None else None,
                cmap=self.cmap if z_obs is not None else None,
                vmin=self.vmin,
                vmax=self.vmax,
                zorder=zorder,
                **kwargs
            )

        if is_exclusive_feasible.any():
            kwargs = experiment_scatter_observations_feasible.copy()
            if z_obs is not None: kwargs.pop("facecolor")
            scatter = self.ax.scatter(
                x=x_obs[is_exclusive_feasible].cpu(),
                y=y_obs[is_exclusive_feasible].cpu(),
                c=z_obs[is_exclusive_feasible].cpu() if z_obs is not None else None,
                cmap=self.cmap if z_obs is not None else None,
                vmin=self.vmin,
                vmax=self.vmax,
                zorder=zorder + 1,
                **kwargs
            )
            if self.mappable is None:
                self.mappable = scatter

        if is_best.any():
            kwargs = experiment_scatter_observations_best_value.copy()
            if z_obs is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_obs[is_best].cpu(),
                y=y_obs[is_best].cpu(),
                c=z_obs[is_best].cpu() if z_obs is not None else None,
                cmap=self.cmap if z_obs is not None else None,
                vmin=self.vmin,
                vmax=self.vmax,
                zorder=zorder + 2,
                **kwargs
            )

        return self

    def plot_next_X(self, zorder: int = 30):
        if self.bo.new_X is not None:
            kwargs = next_X_1d.copy()
            label = kwargs.pop("label")  # Get label if it exists
            new_x_np = self.bo.new_X.detach().cpu().numpy().flatten()

            for i, x in enumerate(new_x_np):
                # Only apply the label to the very first line
                current_label = label if i == 0 else "_nolegend_"

                self.ax.axvline(
                    x=x,
                    zorder=zorder,
                    label=current_label,
                    **kwargs
                )
        return self

    def add_colorbar(self):
        """ Adds a colorbar to the right of the plot based on the Z configuration. """
        # Only add if a Z dimension was requested and we have a mappable object
        if self.z_cfg is None or self.mappable is None:
            return

        # Check if colorbar already exists to avoid duplicates
        if hasattr(self, 'cbar') and self.cbar is not None:
            return

        # Create the colorbar
        self.cbar = self.fig.colorbar(
            self.mappable,
            ax=self.ax,
            fraction=0.046,
            pad=0.04
        )

        # Set the label from our CfgBase object
        self.cbar.set_label(self.z_cfg.label.capitalize())
        self.cbar.ax.tick_params()

    def plot_legend(self, zorder: int = 100):
        leg = self.ax.legend(loc='upper right', frameon=True)
        leg.set_zorder(zorder)

    def plot(self):
        self.plot_gp_posterior()
        self.plot_ground_truth()
        self.plot_observations()
        self.plot_next_X()
        self.add_colorbar()
        self.plot_legend()
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)


class Experiment2DPlotter(PlotterBase):
    def __init__(
            self,
            bo: BayesianOptimizer,
            x: AxisSpec = ("par", 0),
            y: AxisSpec = ("par", 1),
            z: AxisSpec | None = ("obj", 0),
            cmap='coolwarm',
            grid=True,
            seed=None
    ):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.x_cfg = bo.objective.get_config(*x)
        self.y_cfg = bo.objective.get_config(*y)
        self.z_cfg = bo.objective.get_config(*z) if z else None

        self.cmap = cmap
        self.n_grid_points = 1000
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.mappable = None
        self.cbar = None
        self.grid = grid
        self.seed = seed
        self.vmin, self.vmax = None, None

        self.ax.set_xlabel(self.x_cfg.label.capitalize())
        self.ax.set_ylabel(self.y_cfg.label.capitalize())

        # --- bounds + scale validation ---
        if hasattr(self.x_cfg, 'bounds') and self.x_cfg.bounds is not None:
            low, high = self.x_cfg.bounds
            self.ax.set_xlim(low, high)

        if hasattr(self.y_cfg, 'bounds') and self.y_cfg.bounds is not None:
            low, high = self.y_cfg.bounds
            self.ax.set_ylim(low, high)

        if self.z_cfg is not None and hasattr(self.z_cfg, 'bounds') and self.z_cfg.bounds is not None:
            self.vmin, self.vmax = self.z_cfg.bounds

    @staticmethod
    def _get_data(cfg, X_gt, Y_obj, Y_con, Y_trk):
        """ Extracts the correct column from the correct tensor. """
        if cfg is None: return None
        if isinstance(cfg, ParCfg): return X_gt[..., cfg.index]
        if isinstance(cfg, ObjCfg): return Y_obj[..., cfg.index]
        if isinstance(cfg, TrkCfg): return Y_trk[..., cfg.index]
        if isinstance(cfg, IneqYConCfg): return Y_con[..., cfg.index]
        raise TypeError(f"Unrecognised configuration type: {type(cfg)}")

    def plot_ground_truth(self, zorder: int = 0):
        if self.z_cfg is None:
            return self

        N = self.n_grid_points
        # Implicitly force grid generation for contour background
        X_gt = self._generate_uniform_grid()

        Y_obj = self.bo.objective.evaluate_true_objective(X_gt)
        Y_con = self.bo.objective.evaluate_true_constraint(X_gt)
        Y_trk = self.bo.objective.evaluate_tracker(X_gt)

        # Extract data for contour (Z axis)
        z_vals = self._get_data(self.z_cfg, X_gt, Y_obj, Y_con, Y_trk)

        X_grid_np = X_gt[..., self.x_cfg.index].reshape(N, N).cpu().numpy()
        Y_grid_np = X_gt[..., self.y_cfg.index].reshape(N, N).cpu().numpy()
        Z_grid_np = z_vals.reshape(N, N).cpu().numpy()

        # --- Handle X Constraints (Input Feasibility) ---
        # We mask the contour where X is invalid (e.g. geometric constraints)
        is_x_feasible = self.bo.objective.is_X_feasible(X_gt)
        x_feas_mask_np = is_x_feasible.reshape(N, N).cpu().numpy()

        Z_masked_np = np.ma.masked_where(np.logical_not(x_feas_mask_np), Z_grid_np)

        cp = self.ax.contourf(
            X_grid_np,
            Y_grid_np,
            Z_masked_np,
            levels=100,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            zorder=zorder,
            alpha=0.8
        )
        self.mappable = cp
        return self

    def plot_observations(self, zorder: int = 10):
        if self.bo.X is None: return self

        X, Y_obj = self.bo.X, self.bo.Y_obj
        Y_con, Y_track = self.bo.Y_con, self.bo.Y_trk

        x_obs = self._get_data(self.x_cfg, X, Y_obj, Y_con, Y_track)
        y_obs = self._get_data(self.y_cfg, X, Y_obj, Y_con, Y_track)

        Y_full = torch.cat([Y_obj, Y_con], dim=-1) if Y_con is not None else Y_obj

        # --- Handle Both X and Y Constraints ---
        input_mask = self.bo.objective.is_X_feasible(X)
        output_mask = self.bo.objective.is_Y_feasible(Y_full)

        # A point is only truly feasible if it satisfies BOTH input and output constraints
        is_feasible = torch.logical_and(input_mask, output_mask)
        is_infeasible = torch.logical_not(is_feasible)

        # Identify Best
        is_best = torch.zeros_like(is_feasible, dtype=torch.bool)
        feasible_indices = torch.where(is_feasible)[0]
        if feasible_indices.numel() > 0:
            Y_max_space = Y_obj[is_feasible].clone()
            Y_max_space[..., self.bo.objective.to_minimize] *= -1
            is_best_idx = torch.argmax(Y_max_space)
            is_best[feasible_indices[is_best_idx]] = True

        is_exclusive_feasible = torch.logical_and(is_feasible, torch.logical_not(is_best))

        # === Render Layers ===

        if is_infeasible.any():
            self.ax.scatter(
                x=x_obs[is_infeasible].cpu().numpy(),
                y=y_obs[is_infeasible].cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_observations_infeasible
            )

        if is_exclusive_feasible.any():
            self.ax.scatter(
                x=x_obs[is_exclusive_feasible].cpu().numpy(),
                y=y_obs[is_exclusive_feasible].cpu().numpy(),
                zorder=zorder + 1,
                **experiment_scatter_observations_feasible
            )

        if is_best.any():
            self.ax.scatter(
                x=x_obs[is_best].cpu().numpy(),
                y=y_obs[is_best].cpu().numpy(),
                zorder=zorder + 2,
                **experiment_scatter_observations_best_value,
            )

        return self

    def plot_next_X(self, zorder: int = 20):
        if self.bo.new_X is not None:
            X_new = self.bo.new_X.detach().cpu().numpy()
            self.ax.scatter(
                x=X_new[:, self.x_cfg.index],
                y=X_new[:, self.y_cfg.index],
                zorder=zorder,
                **next_X_2d
            )
        return self

    def plot_trajectory(self, zorder: int = 5):
        """
        Batch-aware trajectory:
          - init -> first BO batch (n_init x q)
          - BO batch i -> BO batch i+1 (q x q)
          - last observed batch -> pending new_X (q x q)
        """
        X_np = self.bo.X.detach().cpu().numpy()
        n_pts = X_np.shape[0]
        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size
        X_new = self.bo.new_X.detach().cpu().numpy() if self.bo.new_X is not None else None

        idx_x = self.x_cfg.index
        idx_y = self.y_cfg.index

        # If X includes only the initial dataset, connect all X to New_X.
        if n_pts == n_init:
            if X_new is not None:
                for i in range(n_init):
                    for j in range(len(X_new)):
                        self.ax.annotate(
                            text="",
                            xy=(X_new[j, idx_x], X_new[j, idx_y]),
                            xytext=(X_np[i, idx_x], X_np[i, idx_y]),
                            zorder=zorder,
                            arrowprops=arrow_future,
                        )
            return self

        n_bo_obs = n_pts - n_init
        n_batches = int(n_bo_obs / q)
        batches = []
        for b in range(n_batches):
            s = n_init + b * q
            e = s + q
            batches.append(X_np[s:e])

        # 1) init -> first observed batch
        first_batch = batches[0]
        for i in range(n_init):
            for j in range(first_batch.shape[0]):
                self.ax.annotate(
                    text="",
                    xy=(first_batch[j, idx_x], first_batch[j, idx_y]),
                    xytext=(X_np[i, idx_x], X_np[i, idx_y]),
                    zorder=zorder,
                    arrowprops=arrow_past,
                )

        # 2) connect observed batch k -> batch k+1
        for k in range(len(batches) - 1):
            A = batches[k]
            B = batches[k + 1]
            for i in range(A.shape[0]):
                for j in range(B.shape[0]):
                    self.ax.annotate(
                        text="",
                        xy=(float(B[j, idx_x]), float(B[j, idx_y])),
                        xytext=(float(A[i, idx_x]), float(A[i, idx_y])),
                        zorder=zorder,
                        arrowprops=arrow_past,
                    )

        # 3) last observed batch -> pending new_X
        if X_new is not None and len(X_new) > 0:
            last_batch = batches[-1]
            for i in range(last_batch.shape[0]):
                for j in range(len(X_new)):
                    self.ax.annotate(
                        text="",
                        xy=(float(X_new[j, idx_x]), float(X_new[j, idx_y])),
                        xytext=(float(last_batch[i, idx_x]), float(last_batch[i, idx_y])),
                        zorder=zorder + 2,
                        arrowprops=arrow_future,
                    )

        return self

    def add_colorbar(self):
        """ Adds a colorbar to the right of the plot. """
        if self.mappable is None:
            return

        if hasattr(self, 'cbar') and self.cbar is not None:
            return

        self.cbar = self.fig.colorbar(
            self.mappable,
            ax=self.ax,
            fraction=0.046,
            pad=0.04
        )
        if self.z_cfg:
            self.cbar.set_label(self.z_cfg.label.capitalize())
        self.cbar.ax.tick_params()

    def plot(self):
        self.plot_ground_truth()
        self.plot_observations()
        self.plot_next_X()
        self.plot_trajectory()
        self.add_colorbar()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment_2d.png"
        return super().save_figure(filename=filename)


class ParetoFront2DPlotter(PlotterBase):
    def __init__(self, bo: BayesianOptimizer, x: tuple[str, str | int], y: tuple[str, str | int],
                 z: tuple[str, str | int] | None = None, cmap='coolwarm', grid=False, seed=None):
        """
        Initializes the 2D Pareto Plotter.
        Args:
            bo: The BayesianOptimizer instance.
            x: Tuple of (category, y) e.g., ("par", "P1") or ("obj", 0).
            y: Tuple of (category, y).
            z: Optional tuple for the color map (Heatmap).
        """
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCMultiObjectiveBase):
            raise TypeError("Objective must be of type MCMultiObjectiveBase")

        self.x_cfg = bo.objective.get_config(*x)
        self.y_cfg = bo.objective.get_config(*y)
        self.z_cfg = bo.objective.get_config(*z) if z else None

        self.cmap = cmap
        self.n_grid_points = 100
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.mappable = None  # To store the scatter for the colorbar
        self.cbar = None  # To store the color bar
        self.grid = grid
        self.seed = seed
        self.vmin, self.vmax = None, None

        self.ax.set_xlabel(self.x_cfg.label.capitalize())
        self.ax.set_ylabel(self.y_cfg.label.capitalize())

        if hasattr(self.x_cfg, 'bounds') and self.x_cfg.bounds is not None:
            low, high = self.x_cfg.bounds
            padding = (high - low) * 0.05
            self.ax.set_xlim(low - padding, high + padding)
        if hasattr(self.y_cfg, 'bounds') and self.y_cfg.bounds is not None:
            low, high = self.y_cfg.bounds
            padding = (high - low) * 0.05
            self.ax.set_ylim(low - padding, high + padding)
        if hasattr(self.z_cfg, 'bounds') and self.z_cfg.bounds is not None:
            self.vmin, self.vmax = self.z_cfg.bounds

    @staticmethod
    def _get_data(cfg, X_gt, Y_obj, Y_con, Y_trk):
        """ Extracts the correct column from the correct tensor based on the Config type. """
        if cfg is None: return None
        if isinstance(cfg, ParCfg): return X_gt[..., cfg.index]
        if isinstance(cfg, ObjCfg): return Y_obj[..., cfg.index]
        if isinstance(cfg, TrkCfg): return Y_trk[..., cfg.index]
        if isinstance(cfg, IneqYConCfg): return Y_con[..., cfg.index]

        raise TypeError(f"Unrecognised configuration type: {type(cfg)}")

    def plot_ground_truth(self):
        # Generate ground truth X
        if self.grid:
            X_gt = self._generate_uniform_grid()
        else:
            sampler = SobolSampler(
                device=self.bo.device,
                dtype=self.bo.dtype,
                objective=self.bo.objective,
                seed=self.seed if isinstance(self.seed, int) else None
            )
            X_gt = sampler.draw_samples(self.n_grid_points ** 2)

        if self.bo.objective.lin_eq_X_con is not None:
            X_gt = project_linear_equalities(X=X_gt, lin_eq_cons=self.bo.objective.lin_eq_X_con)

        Y_obj = self.bo.objective.evaluate_true_objective(X_gt)
        Y_con = self.bo.objective.evaluate_true_constraint(X_gt)
        Y_trk = self.bo.objective.evaluate_tracker(X_gt)

        # Slice data for axes
        x_vals = self._get_data(self.x_cfg, X_gt, Y_obj, Y_con, Y_trk)
        y_vals = self._get_data(self.y_cfg, X_gt, Y_obj, Y_con, Y_trk)
        z_vals = self._get_data(self.z_cfg, X_gt, Y_obj, Y_con, Y_trk)

        # Feasibility masking
        Y_full = torch.cat([Y_obj, Y_con], dim=-1) if Y_con is not None else Y_obj
        input_mask = self.bo.objective.is_X_feasible(X_gt)
        output_mask = self.bo.objective.is_Y_feasible(Y_full)
        is_feasible = torch.logical_and(input_mask, output_mask)
        is_infeasible = torch.logical_and(input_mask, torch.logical_not(output_mask))

        # Identify Pareto Points
        is_pareto = torch.zeros_like(is_feasible, dtype=torch.bool)
        feasible_indices = torch.where(is_feasible)[0]

        if feasible_indices.numel() > 0:
            Y_max_space = Y_obj[is_feasible].clone()
            Y_max_space[..., self.bo.objective.to_minimize] *= -1
            pareto_sub_mask = is_non_dominated(Y_max_space)
            is_pareto[feasible_indices[pareto_sub_mask]] = True

        mask_exclusive_feasible = torch.logical_and(is_feasible, torch.logical_not(is_pareto))

        # === Render Layers ===

        # Layer 1: Infeasible
        if is_infeasible.any():
            kwargs = experiment_scatter_gnd_truth_infeasible.copy()
            if z_vals is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_vals[is_infeasible].cpu().numpy(),
                y=y_vals[is_infeasible].cpu().numpy(),
                c=z_vals[is_infeasible].cpu().numpy() if z_vals is not None else None,
                vmin=self.vmin if self.vmin is not None else None,
                vmax=self.vmax if self.vmax is not None else None,
                cmap=self.cmap if z_vals is not None else None,
                **kwargs
            )

        # Layer 2: Feasible but Dominated
        if mask_exclusive_feasible.any():
            kwargs = experiment_scatter_gnd_truth_feasible.copy()
            if z_vals is not None: kwargs.pop("facecolor")
            self.mappable = self.ax.scatter(
                x=x_vals[mask_exclusive_feasible].cpu().numpy(),
                y=y_vals[mask_exclusive_feasible].cpu().numpy(),
                c=z_vals[mask_exclusive_feasible].cpu().numpy() if z_vals is not None else None,
                vmin=self.vmin if z_vals is not None else None,
                vmax=self.vmax if z_vals is not None else None,
                cmap=self.cmap if z_vals is not None else None,
                **kwargs
            )

        # Layer 3: Pareto Front
        if is_pareto.any():
            kwargs = experiment_scatter_gnd_truth_pareto_front.copy()
            if z_vals is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_vals[is_pareto].cpu().numpy(),
                y=y_vals[is_pareto].cpu().numpy(),
                c=z_vals[is_pareto].cpu().numpy() if z_vals is not None else None,
                vmin=self.vmin if z_vals is not None else None,
                vmax=self.vmax if z_vals is not None else None,
                cmap=self.cmap if z_vals is not None else None,
                **kwargs
            )
        return self

    def plot_observations(self, zorder=4):
        X, Y_obj = self.bo.X, self.bo.Y_obj
        Y_con, Y_track = self.bo.Y_con, self.bo.Y_trk

        x_obs = self._get_data(self.x_cfg, X, Y_obj, Y_con, Y_track)
        y_obs = self._get_data(self.y_cfg, X, Y_obj, Y_con, Y_track)
        z_obs = self._get_data(self.z_cfg, X, Y_obj, Y_con, Y_track)

        # Feasibility mask
        Y_full = torch.cat([Y_obj, Y_con], dim=-1) if Y_con is not None else Y_obj
        input_mask = self.bo.objective.is_X_feasible(X)
        output_mask = self.bo.objective.is_Y_feasible(Y_full)
        is_feasible = torch.logical_and(input_mask, output_mask)

        # Identify Observed Pareto
        is_pareto = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
        f_idx = torch.where(is_feasible)[0]
        if f_idx.numel() > 0:
            Y_ms = Y_obj[is_feasible].clone()
            Y_ms[..., self.bo.objective.to_minimize] *= -1
            is_pareto[f_idx[is_non_dominated(Y_ms)]] = True

        # Render
        mask_infeasible = torch.logical_not(is_feasible)
        mask_dominated = torch.logical_and(is_feasible, torch.logical_not(is_pareto))

        if mask_infeasible.any():
            kwargs = experiment_scatter_observations_infeasible.copy()
            if z_obs is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_obs[mask_infeasible].cpu().numpy(),
                y=y_obs[mask_infeasible].cpu().numpy(),
                c=z_obs[mask_infeasible].cpu().numpy() if z_obs is not None else None,
                vmin=self.vmin if z_obs is not None else None,
                vmax=self.vmax if z_obs is not None else None,
                cmap=self.cmap if z_obs is not None else None,
                **kwargs
            )

        if mask_dominated.any():
            kwargs = experiment_scatter_observations_feasible.copy()
            if z_obs is not None: kwargs.pop("facecolor")
            scatter_obs = self.ax.scatter(
                x=x_obs[mask_dominated].cpu().numpy(),
                y=y_obs[mask_dominated].cpu().numpy(),
                c=z_obs[mask_dominated].cpu().numpy() if z_obs is not None else None,
                vmin=self.vmin if z_obs is not None else None,
                vmax=self.vmax if z_obs is not None else None,
                cmap=self.cmap if z_obs is not None else None,
                **kwargs
            )

            # If ground truth didn't set the mappable, we do it here
            if self.mappable is None:
                self.mappable = scatter_obs

        if is_pareto.any():
            kwargs = experiment_scatter_observations_pareto_front.copy()
            if z_obs is not None: kwargs.pop("facecolor")
            self.ax.scatter(
                x=x_obs[is_pareto].cpu().numpy(),
                y=y_obs[is_pareto].cpu().numpy(),
                c=z_obs[is_pareto].cpu().numpy() if z_obs is not None else None,
                vmin=self.vmin if z_obs is not None else None,
                vmax=self.vmax if z_obs is not None else None,
                cmap=self.cmap if z_obs is not None else None,
                **kwargs
            )

        return self

    def add_colorbar(self):
        """ Adds a colorbar to the right of the plot based on the Z configuration. """
        # Only add if a Z dimension was requested and we have a mappable object
        if self.z_cfg is None or self.mappable is None:
            return

        # Check if colorbar already exists to avoid duplicates
        if hasattr(self, 'cbar') and self.cbar is not None:
            return

        # Create the colorbar
        self.cbar = self.fig.colorbar(
            self.mappable,
            ax=self.ax,
            fraction=0.046,
            pad=0.04
        )

        # Set the label from our CfgBase object
        self.cbar.set_label(self.z_cfg.label.capitalize())
        self.cbar.ax.tick_params()

    def plot(self):
        self.plot_ground_truth()
        self.plot_observations()
        self.add_colorbar()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)
