import torch
from matplotlib import pyplot as plt
from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase
from plotters.styles import *
from objectives.base_class import MCSingleObjectiveBase


class Experiment1DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel(bo.objective.objective_names[0] if bo.objective.objective_names is not None else r"$x$")
        self.ax.set_ylabel(bo.objective.objective_names[1] if bo.objective.objective_names is not None else r"$f(x)$")

    def plot_gp_posterior(self, zorder: int = 0):
        if self.bo.model is None:
            return self

        X_grid = self._generate_uniform_grid()

        with torch.no_grad():
            # Note: model.posterior() returns a value for each GP model, including those associated with
            # the constraints. Therefore, here we keep only the posterior associated with the objective outcomes.
            posterior = self.bo.model.posterior(X_grid)
            mean = posterior.mean[..., self.bo.objective.outcomes].squeeze(-1)
            std = posterior.variance.sqrt()[..., self.bo.objective.outcomes].squeeze(-1)

        X_np = X_grid.squeeze().cpu().numpy()
        mean_np = mean.cpu().numpy()
        std_np = std.cpu().numpy()

        # Plot GP Mean
        self.ax.plot(
            X_np,
            mean_np,
            zorder=zorder,
            **gp_mean
        )

        # Plot Confidence Intervals (shaded area)
        self.ax.fill_between(
            X_np,
            mean_np - std_np,
            mean_np + std_np,
            zorder=zorder,
            **gp_confidence_interval_1sigma
        )
        self.ax.fill_between(
            X_np,
            mean_np - 2 * std_np,
            mean_np + 2 * std_np,
            zorder=zorder,
            **gp_confidence_interval_2sigma
        )
        self.ax.fill_between(
            X_np,
            mean_np - 3 * std_np,
            mean_np + 3 * std_np,
            zorder=zorder,
            **gp_confidence_interval_3sigma
        )

        return self

    def plot_ground_truth(self, zorder: int = 1):
        X_grid = self._generate_uniform_grid()
        Y_obj_gt = self.bo.objective.evaluate_true_objective(X_grid)
        Y_con_gt = self.bo.objective.evaluate_true_constraint(X_grid)
        if Y_con_gt is not None:
            Y_full = torch.cat([Y_obj_gt, Y_con_gt], dim=-1)
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in self.bo.objective.constraints]).all(dim=0).squeeze()
        else:
            feasible_mask = torch.ones_like(X_grid, dtype=torch.bool, device=X_grid.device)

        if feasible_mask.any():
            feasible_X = X_grid[feasible_mask]
            feasible_Y = Y_obj_gt[feasible_mask]
            self.ax.scatter(
                x=feasible_X.detach().cpu().numpy(),
                y=feasible_Y.detach().cpu().numpy(),
                zorder=zorder,
                **feasible_ground_truth
            )
        infeasible_mask = torch.logical_not(feasible_mask)
        if infeasible_mask.any():
            infeasible_X = X_grid[infeasible_mask]
            infeasible_Y = Y_obj_gt[infeasible_mask]
            self.ax.scatter(
                x=infeasible_X.detach().cpu().numpy(),
                y=infeasible_Y.detach().cpu().numpy(),
                zorder=zorder,
                **infeasible_ground_truth
            )

        return self

    def plot_observations(self, zorder: int = 2):
        # Note: scatter plots are used as feasible and infeasible regions may lead to discontinuities.
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
                X_f_plot.detach().cpu().numpy(),
                Y_f_plot.detach().cpu().numpy(),
                zorder=zorder,
                **feasible_observations
            )

        # Plot infeasible observations
        if X_i is not None:
            self.ax.scatter(
                X_i.detach().cpu().numpy(),
                Y_i.detach().cpu().numpy(),
                zorder=zorder,
                **infeasible_observations
            )

        # Plot optimum
        if X_best is not None:
            self.ax.scatter(
                X_best.detach().cpu().numpy(),
                Y_best.detach().cpu().numpy(),
                zorder=zorder,
                **optimum,
            )

        return self

    def plot_next_X(self, zorder: int = 4):
        new_x = self.bo.new_X

        if new_x is not None:
            # Flatten to handle batch acquisition (q > 1)
            new_x_np = new_x.detach().cpu().numpy().flatten()

            for i, x in enumerate(new_x_np):
                self.ax.axvline(
                    x=x,
                    zorder=zorder,
                    **new_X_1d
                )
        return self

    def plot(self):
        self.plot_gp_posterior()
        self.plot_ground_truth()
        self.plot_observations()
        self.plot_next_X()
        self.ax.legend(loc='upper right', fontsize='small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)


class Experiment2DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel(bo.objective.objective_names[0] if bo.objective.objective_names is not None else r"$x_1$")
        self.ax.set_ylabel(bo.objective.objective_names[1] if bo.objective.objective_names is not None else r"$x_2$")

    def plot_ground_truth(self):
        """Plot the ground truth surface."""
        X_grid = self._generate_uniform_grid()
        Y_obj = self.bo.objective.evaluate_true_objective(X_grid)
        N = self.n_grid_points
        X_np = X_grid[:, 0].reshape(N, N).cpu().numpy()
        Y_np = X_grid[:, 1].reshape(N, N).cpu().numpy()
        Z_np = Y_obj.reshape(N, N).cpu().numpy()
        cp = self.ax.contourf(X_np, Y_np, Z_np, levels=50, cmap='viridis', alpha=0.8)
        self.fig.colorbar(cp, ax=self.ax)
        return self

    # TODO
    def plot_constraints(self):
        raise NotImplementedError()

    def plot_observations(self, zorder: int = 4):
        """Plot static experimental points."""
        X_f, _ = self.bo.compute_feasible_XY()
        if X_f is not None:
            self.ax.scatter(
                x=X_f[:, 0].cpu(),
                y=X_f[:, 1].cpu(),
                c='white',
                edgecolors='black',
                s=40,
                label='Feasible',
                zorder=zorder
            )
        X_i, _ = self.bo.compute_infeasible_XY()

        if X_i is not None:
            self.ax.scatter(
                x=X_i[:, 0].cpu(),
                y=X_i[:, 1].cpu(),
                c='red',
                marker='x',
                s=40,
                label='Infeasible',
                zorder=zorder
            )
        return self

    def plot_new_X(self, zorder: int = 4):
        X_new = self.bo.new_X.detach().cpu().numpy()
        if X_new is not None:
            self.ax.scatter(
                x=X_new[:, 0],
                y=X_new[:, 1],
                c='red',
                marker='*',
                s=150,
                edgecolors='black',
                label='Next Candidate',
                zorder=zorder + 3,
                alpha=0.8
            )

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
                        arrowprops=style_arrow_future,
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
                    arrowprops=style_arrow_past,
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
                        arrowprops=style_arrow_past,
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
                        arrowprops=style_arrow_future,
                    )

        return self

    def plot(self):
        """Main plotting pipeline."""
        self.plot_ground_truth()
        self.plot_new_X()
        self.plot_trajectory()
        self.plot_observations()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)
