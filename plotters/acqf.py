import torch
from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.base_class import MCSingleObjectiveBase
from objectives.variable_registry import *
from plotters.base_class import PlotterBase
import numpy as np
import matplotlib.pyplot as plt
from plotters.styles import *


class Acqf1DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer, x: tuple[str, str | int] = ("par", 0),
                 z: tuple[str, str] | None = None, cmap="coolwarm", grid: bool = True, seed=None):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.x_cfg = bo.objective.get_config(*x)
        # self.y_cfg = bo.objective.get_config(*y)
        self.z_cfg = bo.objective.get_config(*z) if z else None

        self.cmap = cmap
        self.n_grid_points = 100000
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.mappable = None
        self.cbar = None
        self.grid = grid
        self.seed = seed
        self.vmin, self.vmax = None, None

        self.ax.set_xlabel(self.x_cfg.label)
        self.ax.set_ylabel(self.bo.acqf.__name__)

        if hasattr(self.x_cfg, 'bounds') and self.x_cfg.bounds is not None:
            low, high = self.x_cfg.bounds
            padding = (high - low) * 0.05
            self.ax.set_xlim(low - padding, high + padding)
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

    def plot_acquisition(self, zorder: int = 0):
        if self.bo.acqf_instance is None:
            raise ValueError("Acquisition function must be set before plotting.")

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

        Y_obj = self.bo.objective.evaluate_true_objective(X)
        Y_con = self.bo.objective.evaluate_true_constraint(X)
        Y_trk = self.bo.objective.evaluate_trackers(X)

        z_vals = self._get_data(self.z_cfg, X, Y_obj, Y_con, Y_trk)

        with torch.no_grad():
            acq_values = self.bo.acqf_instance(X.unsqueeze(1))

        X_np = X.squeeze().detach().cpu().numpy()
        acq_np = acq_values.squeeze().detach().cpu().numpy()

        if getattr(self.bo.acqf, "_log"):
            y = -np.log(np.abs(acq_np))
        else:
            y = acq_np

        kwargs = acqf_1d.copy()
        if z_vals is not None:
            kwargs.pop("color")
        scatter = self.ax.scatter(
            x=X_np,
            y=y,
            c=z_vals if z_vals is not None else None,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            zorder=zorder,
            **kwargs,
        )
        if self.mappable is None:
            self.mappable = scatter

        return self

    def plot_next_X(self, zorder: int = 1):
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

        # Set the label from our Cfg object
        self.cbar.set_label(self.z_cfg.label)
        self.cbar.ax.tick_params()

    def plot_legend(self, zorder: int = 100):
        leg = self.ax.legend(loc='upper right', frameon=True)
        leg.set_zorder(zorder)

    def plot(self):
        self.plot_acquisition()
        self.add_colorbar()
        self.plot_next_X()
        self.plot_legend()
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acqf.png"
        return super().save_figure(filename=filename)


class Acqf2DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer, x: tuple[str, str | int] = ("par", 0),
                 y: tuple[str, str | int] = ("par", 1), z: tuple[str, str | int] | None = None,
                 cmap='viridis', grid=True):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.x_cfg = bo.objective.get_config(*x)
        self.y_cfg = bo.objective.get_config(*y)
        self.z_cfg = bo.objective.get_config(*z) if z else None

        self.cmap = cmap
        self.n_grid_points = 100
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.mappable = None
        self.grid = grid

        self.ax.set_xlabel(self.x_cfg.label)
        self.ax.set_ylabel(self.y_cfg.label)

        if hasattr(self.x_cfg, 'bounds') and self.x_cfg.bounds is not None:
            low, high = self.x_cfg.bounds
            self.ax.set_xlim(low, high)
        if hasattr(self.y_cfg, 'bounds') and self.y_cfg.bounds is not None:
            low, high = self.y_cfg.bounds
            self.ax.set_ylim(low, high)

    @staticmethod
    def _get_data(cfg, X_gt, Y_obj, Y_con, Y_trk):
        """ Extracts the correct column from the correct tensor. """
        if cfg is None: return None
        if isinstance(cfg, ParCfg): return X_gt[..., cfg.index]
        if isinstance(cfg, ObjCfg): return Y_obj[..., cfg.index]
        if isinstance(cfg, TrkCfg): return Y_trk[..., cfg.index]
        if isinstance(cfg, IneqYConCfg): return Y_con[..., cfg.index]
        raise TypeError(f"Unrecognised configuration type: {type(cfg)}")

    def plot_acquisition(self, zorder: int = 0):
        # Generate grid
        X_grid = self._generate_uniform_grid()
        N = self.n_grid_points

        # Evaluate Acquisition Function
        with torch.no_grad():
            # (N*N, 1) -> (N, N)
            acq_values = self.bo.acqf_instance(X_grid.unsqueeze(1))

        # Extract X, Y coordinates for plotting
        X_grid_np = X_grid[..., self.x_cfg.index].reshape(N, N).cpu().numpy()
        Y_grid_np = X_grid[..., self.y_cfg.index].reshape(N, N).cpu().numpy()
        Z_np = acq_values.reshape(N, N).cpu().numpy()

        # Handle Log Scaling if necessary
        if getattr(self.bo.acqf, "_log", False):
            Z_np = -np.log(np.abs(Z_np))

        # Check feasibility for masking
        feas_mask_np = self.bo.objective.is_X_feasible(X_grid).reshape(N, N).cpu().numpy()
        Z_masked_np = np.ma.masked_where(np.logical_not(feas_mask_np), Z_np)

        cp = self.ax.contourf(
            X_grid_np,
            Y_grid_np,
            Z_masked_np,
            levels=20,
            cmap=self.cmap,
            zorder=zorder,
            alpha=0.8
        )
        self.mappable = cp
        return self

    def plot_observations(self, zorder: int = 10):
        if self.bo.X is None: return self

        X, Y_obj = self.bo.X, self.bo.Y_obj
        Y_con, Y_track = self.bo.Y_con, self.bo.Y_track

        x_obs = self._get_data(self.x_cfg, X, Y_obj, Y_con, Y_track)
        y_obs = self._get_data(self.y_cfg, X, Y_obj, Y_con, Y_track)

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

    def plot_new_X(self, zorder: int = 20):
        if self.bo.new_X is not None:
            X = self.bo.new_X.detach().cpu().numpy()
            self.ax.scatter(
                x=X[:, self.x_cfg.index],
                y=X[:, self.y_cfg.index],
                zorder=zorder,
                **next_X_2d
            )
        return self

    def plot_trajectory(self, zorder: int = 5):
        """
        Batch-aware trajectory logic connecting observations to new_X.
        """
        X_np = self.bo.X.detach().cpu().numpy()
        n_pts = X_np.shape[0]
        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size
        X_new = self.bo.new_X.detach().cpu().numpy() if self.bo.new_X is not None else None

        idx_x = self.x_cfg.index
        idx_y = self.y_cfg.index

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

        # 2) connect observed batches
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
        """ Adds a colorbar for the acquisition values. """
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
        self.cbar.set_label(self.bo.acqf.__name__)
        self.cbar.ax.tick_params()

    def plot(self):
        self.plot_acquisition()
        self.plot_observations()
        self.plot_trajectory()
        self.plot_new_X()
        self.add_colorbar()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acquisition_2d.png"
        return super().save_figure(filename=filename)
