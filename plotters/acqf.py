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
        X = self.bo.new_X.detach().cpu().numpy()
        if X is not None:
            if X.ndim == 0:
                X = [X.item()]
            for i, x in enumerate(X):
                self.ax.axvline(
                    x=x,
                    zorder=zorder,
                    **next_X_1d
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

    def __init__(self, bo: BayesianOptimizer, x: tuple[str, str | int] | None = None,
                 y: tuple[str, str | int] | None = None):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        # TODO: by default the method should plot p0 vs p1.
        # self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        # self.ax.set_xlabel(
        #     self.bo.objective.objective_names[0] if bo.objective.objective_names is not None else r"$x_1$")
        # self.ax.set_ylabel(
        #     self.bo.objective.objective_names[1] if bo.objective.objective_names is not None else r"$x_2$")
        # self.ax.set_xlim(self.bo.objective.bounds[:, 0].detach().cpu().numpy())
        # self.ax.set_ylim(self.bo.objective.bounds[:, 1].detach().cpu().numpy())

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.x = x
        self.y = y

        # Lock X and Y scales from VariableRegistry bounds
        self.ax.set_xlim(self.x.bounds)
        self.ax.set_ylim(self.y.bounds)

        self.ax.set_xlabel(x.label)
        self.ax.set_ylabel(y.label)

    def plot_acquisition(self, zorder: int = 0):
        N = self.n_grid_points
        X_grid = self._generate_uniform_grid()
        X_np = X_grid[:, self.x.index].reshape(N, N).cpu().numpy()
        Y_np = X_grid[:, self.y.index].reshape(N, N).cpu().numpy()
        with torch.no_grad():
            Z_np = self.bo.acqf_instance(X_grid.unsqueeze(1)).reshape(N, N).cpu().numpy()
        feas_mask_np = self.bo.objective.is_X_feasible(X_grid).reshape(N, N).cpu().numpy()
        Z_masked_np = np.ma.masked_where(np.logical_not(feas_mask_np), Z_np)

        cp = self.ax.contourf(
            X_np,
            Y_np,
            Z_masked_np,
            zorder=zorder,
            **experiment_contour_gnd_truth
        )
        self.fig.colorbar(
            cp,
            ax=self.ax
        )

        return self

    def plot_observations(self, zorder: int = 1):
        X_best, Y_best = self.bo.best_feasible_X, self.bo.best_feasible_Y
        X_f, Y_f = self.bo.compute_feasible_XY()
        X_i, Y_i = self.bo.compute_infeasible_XY()

        X_f_plot, Y_f_plot = X_f, Y_f
        if X_f is not None and X_best is not None:
            # Create a mask for all points that are NOT the current best
            # This assumes X_f and X_best are torch tensors; use np.equal for numpy
            mask = torch.logical_not(X_f == X_best).all(dim=-1)
            X_f_plot, Y_f_plot = X_f[mask], Y_f[mask]

        # Plot feasible observations (excluding the optimum)
        if X_f_plot is not None and X_f_plot.nelement() > 0:
            self.ax.scatter(
                x=X_f_plot[..., self.x.index].detach().cpu().numpy(),
                y=X_f_plot[..., self.y.index].detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_observations_feasible
            )

        # Plot infeasible observations
        if X_i is not None:
            self.ax.scatter(
                x=X_i[..., self.x.index].detach().cpu().numpy(),
                y=X_i[..., self.y.index].detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_observations_infeasible
            )

        # Plot optimum
        if X_best is not None:
            self.ax.scatter(
                x=X_best[..., self.x.index].detach().cpu().numpy(),
                y=X_best[..., self.y.index].detach().cpu().numpy(),
                zorder=zorder,
                **optimum,
            )

        return self

    def plot_new_X(self, zorder: int = 2):
        if self.bo.new_X is not None:
            X = self.bo.new_X.detach().cpu().numpy()
            self.ax.scatter(
                x=X[:, self.x.index],
                y=X[:, self.y.index],
                color='red', marker='*', s=200,
                edgecolor='white', label="Next X", zorder=zorder,
            )
        return self

    def plot_trajectory(self, zorder: int = 3):
        """
        Batch-aware trajectory:
          - init -> first BO batch (n_init x q)
          - BO batch i -> BO batch i+1 (q x q), for all adjacent observed batches
          - last observed batch -> pending new_X (q x q) as "future"
        """
        X_np = self.bo.X.detach().cpu().numpy()
        n_pts = X_np.shape[0]
        n_init = self.bo.n_initial_samples
        q = self.bo.batch_size
        X_new = self.bo.new_X.detach().cpu().numpy()  # returns (q,2) or None

        # If X includes only the initial dataset, connect all X to New_X.
        if n_pts == n_init:
            for i in range(n_init):
                for j in range(len(X_new)):
                    self.ax.annotate(
                        text="",
                        xy=(X_new[j, self.x.index], X_new[j, self.y.index]),
                        xytext=(X_np[i, self.x.index], X_np[i, self.y.index]),
                        zorder=zorder,
                        arrowprops=arrow_future,
                    )
            return self

        # If X includes also Bayesian optimized points, connect the initial dataset
        # to the first q-batch (1), then iteratively connect the k-th batch to
        # the k-th + 1 batch, and finally connect the last observed batch to the New_X.
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
                    xy=(first_batch[j, self.x.index], first_batch[j, self.y.index]),
                    xytext=(X_np[i, self.x.index], X_np[i, self.y.index]),
                    zorder=zorder,
                    arrowprops=arrow_past,
                )

        # 2) connect observed batch k -> batch k+1 (fully connected)
        for k in range(len(batches) - 1):
            A = batches[k]
            B = batches[k + 1]
            for i in range(A.shape[0]):
                for j in range(B.shape[0]):
                    self.ax.annotate(
                        text="",
                        xy=(float(B[j, self.x.index]), float(B[j, self.y.index])),
                        xytext=(float(A[i, self.x.index]), float(A[i, self.y.index])),
                        zorder=zorder,
                        arrowprops=arrow_past,
                    )

        # 3) last observed batch -> pending new_X (fully connected "future")
        if X_new is not None and len(X_new) > 0:
            last_batch = batches[-1]
            for i in range(last_batch.shape[0]):
                for j in range(len(X_new)):
                    self.ax.annotate(
                        text="",
                        xy=(float(X_new[j, self.x.index]), float(X_new[j, self.y.index])),
                        xytext=(float(last_batch[i, self.x.index]), float(last_batch[i, self.y.index])),
                        zorder=zorder + 2,
                        arrowprops=arrow_future,
                    )

        return self

    def plot(self):
        self.plot_acquisition()
        self.plot_observations()
        self.plot_trajectory()
        self.plot_new_X()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acquisition.png"
        return super().save_figure(filename=filename)
