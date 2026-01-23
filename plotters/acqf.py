import torch
from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.base_class import MCSingleObjectiveBase
from plotters.base_class import PlotterBase
import numpy as np
import matplotlib.pyplot as plt
from plotters.styles import *


class Acqf1DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel(bo.objective.objective_names[0] if bo.objective.objective_names is not None else r"$x$")
        self.ax.set_ylabel(bo.objective.objective_names[1] if bo.objective.objective_names is not None else r"$f(x)$")

    def plot_acquisition(self):
        if self.bo.acqf_instance is None:
            raise ValueError("Acquisition function must be set before plotting.")

        X_grid = self._generate_uniform_grid()

        with torch.no_grad():
            acq_values = self.bo.acqf_instance(X_grid.unsqueeze(1))

        X_np = X_grid.squeeze().detach().cpu().numpy()
        acq_np = acq_values.squeeze().detach().cpu().numpy()

        if getattr(self.bo.acqf, "_log"):
            log_abs_acqf = np.log(np.abs(acq_np))
            self.ax.plot(X_np, -log_abs_acqf, **acqf_1d)
            self.ax.set_ylabel(r'$-\log \left( | \mathrm{Acquisition\ Value} | \right) $')
        else:
            self.ax.plot(X_np, acq_np, **acqf_1d)
            self.ax.set_ylabel(r'$\mathrm{Acquisition\ Value}$')

        return self

    def plot_next_X(self):
        X = self.bo.new_X.detach().cpu().numpy()
        if X is not None:
            if X.ndim == 0:
                X = [X.item()]
            for i, x in enumerate(X):
                self.ax.axvline(x=x, **new_X_1d)
        return self

    def plot(self):
        self.plot_acquisition()
        self.plot_next_X()
        self.ax.legend(loc='upper right', fontsize='small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acqf.png"
        return super().save_figure(filename=filename)


class Acqf2DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel(
            self.bo.objective.objective_names[0] if bo.objective.objective_names is not None else r"$x_1$")
        self.ax.set_ylabel(
            self.bo.objective.objective_names[1] if bo.objective.objective_names is not None else r"$x_2$")
        self.ax.set_xlim(self.bo.objective.bounds[:, 0].detach().cpu().numpy())
        self.ax.set_ylim(self.bo.objective.bounds[:, 1].detach().cpu().numpy())

    def plot_acquisition(self, zorder: int = 0):
        N = self.n_grid_points
        X_grid = self._generate_uniform_grid()
        X_np = X_grid[:, 0].reshape(N, N).cpu().numpy()
        Y_np = X_grid[:, 1].reshape(N, N).cpu().numpy()
        with torch.no_grad():
            Z_np = self.bo.acqf_instance(X_grid.unsqueeze(1)).reshape(N, N).cpu().numpy()
        feas_mask_np = self.bo.objective.is_input_feasible(X_grid).reshape(N, N).cpu().numpy()
        Z_masked_np = np.ma.masked_where(np.logical_not(feas_mask_np), Z_np)

        cp = self.ax.contourf(
            X_np,
            Y_np,
            Z_masked_np,
            zorder=zorder,
            **contour_gnd_truth
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
                X_f_plot[..., 0].detach().cpu().numpy(),
                X_f_plot[..., 1].detach().cpu().numpy(),
                zorder=zorder,
                **scatter_observations_feasible
            )

        # Plot infeasible observations
        if X_i is not None:
            self.ax.scatter(
                X_i[..., 0].detach().cpu().numpy(),
                X_i[..., 1].detach().cpu().numpy(),
                zorder=zorder,
                **scatter_observations_infeasible
            )

        # Plot optimum
        if X_best is not None:
            self.ax.scatter(
                X_best[..., 0].detach().cpu().numpy(),
                X_best[..., 1].detach().cpu().numpy(),
                zorder=zorder,
                **optimum,
            )

        return self

    def plot_new_X(self, zorder: int = 2):
        if self.bo.new_X is not None:
            X = self.bo.new_X.detach().cpu().numpy()
            self.ax.scatter(
                X[:, 0], X[:, 1],
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
                        xy=(X_new[j, 0], X_new[j, 1]),
                        xytext=(X_np[i, 0], X_np[i, 1]),
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
                    xy=(first_batch[j, 0], first_batch[j, 1]),
                    xytext=(X_np[i, 0], X_np[i, 1]),
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
                        xy=(float(B[j, 0]), float(B[j, 1])),
                        xytext=(float(A[i, 0]), float(A[i, 1])),
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
                        xy=(float(X_new[j, 0]), float(X_new[j, 1])),
                        xytext=(float(last_batch[i, 0]), float(last_batch[i, 1])),
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
