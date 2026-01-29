from enum import Enum
from typing import Union
import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.variable_registry import VariableRegistry
from plotters.base_class import PlotterBase
from plotters.styles import *
from objectives.base_class import MCSingleObjectiveBase, MCMultiObjectiveBase

NameLike = Union[str, Enum, int]


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
            # the ineq_Y_con_cfg. Therefore, here we keep only the posterior associated with the objective outcomes.
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
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in self.bo.objective.ineq_Y_con]).all(dim=0).squeeze()
        else:
            feasible_mask = torch.ones_like(X_grid, dtype=torch.bool, device=X_grid.device)

        if feasible_mask.any():
            feasible_X = X_grid[feasible_mask]
            feasible_Y = Y_obj_gt[feasible_mask]
            self.ax.scatter(
                x=feasible_X.detach().cpu().numpy(),
                y=feasible_Y.detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_gnd_truth_feasible
            )
        infeasible_mask = torch.logical_not(feasible_mask)
        if infeasible_mask.any():
            infeasible_X = X_grid[infeasible_mask]
            infeasible_Y = Y_obj_gt[infeasible_mask]
            self.ax.scatter(
                x=infeasible_X.detach().cpu().numpy(),
                y=infeasible_Y.detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_gnd_truth_infeasible
            )

        return self

    def plot_observations(self, zorder: int = 2):
        # Note: scatter plots are used as infeasible regions may lead to discontinuities.
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
                **experiment_scatter_observations_feasible
            )

        # Plot infeasible observations
        if X_i is not None:
            self.ax.scatter(
                X_i.detach().cpu().numpy(),
                Y_i.detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_observations_infeasible
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
                    **next_X_1d
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

    def __init__(self, bo: BayesianOptimizer, x: VariableRegistry, y: VariableRegistry):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

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

    def plot_ground_truth(self, zorder: int = 0):
        N = self.n_grid_points
        X_grid = self._generate_uniform_grid()
        Y_obj_gt = self.bo.objective.evaluate_true_objective(X_grid)
        X_np = X_grid[:, self.x.index].reshape(N, N).cpu().numpy()
        Y_np = X_grid[:, self.y.index].reshape(N, N).cpu().numpy()
        Z_np = Y_obj_gt.reshape(N, N).cpu().numpy()
        feas_mask_np = self.bo.objective.is_input_feasible(X_grid).reshape(N, N).cpu().numpy()
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
                X_f_plot[..., self.x.index].detach().cpu().numpy(),
                X_f_plot[..., self.y.index].detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_observations_feasible
            )

        # Plot infeasible observations
        if X_i is not None:
            self.ax.scatter(
                X_i[..., self.x.index].detach().cpu().numpy(),
                X_i[..., self.y.index].detach().cpu().numpy(),
                zorder=zorder,
                **experiment_scatter_observations_infeasible
            )

        # Plot optimum
        if X_best is not None:
            self.ax.scatter(
                X_best[..., self.x.index].detach().cpu().numpy(),
                X_best[..., self.y.index].detach().cpu().numpy(),
                zorder=zorder,
                **optimum,
            )

        return self

    def plot_next_X(self, zorder: int = 2):
        X_new = self.bo.new_X.detach().cpu().numpy()
        if X_new is not None:
            self.ax.scatter(
                x=X_new[:, self.x.index],
                y=X_new[:, self.y.index],
                zorder=zorder + 3,
                **next_X_2d
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
        self.plot_ground_truth()
        self.plot_next_X()
        self.plot_trajectory()
        self.plot_observations()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)


class ParetoFront2DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer, x: VariableRegistry, y: VariableRegistry,
                 z: VariableRegistry | None = None, cmap='viridis'):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCMultiObjectiveBase):
            raise TypeError("Objective must be of type MCMultiObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.n_grid_points = 100
        self.cmap = cmap

        self.x = x
        self.y = y
        self.z = z

        # Lock X and Y scales from VariableRegistry bounds
        if self.x.bounds is not None:
            self.ax.set_xlim(self.x.bounds)
        if self.y.bounds is not None:
            self.ax.set_ylim(self.y.bounds)

        self.ax.set_xlabel(x.label)
        self.ax.set_ylabel(y.label)

    def _get_data(self, axis_config, X_gt, Y_obj, Y_con, Y_track):
        if axis_config is None:
            return None
        if isinstance(axis_config, self.bo.objective.Obj):
            return Y_obj[..., axis_config.index]
        if isinstance(axis_config, self.bo.objective.Trk):
            return Y_track[..., axis_config.index]
        if isinstance(axis_config, self.bo.objective.Con):
            return Y_con[..., axis_config.index]
        if isinstance(axis_config, self.bo.objective.Par):
            return X_gt[..., axis_config.index]
        raise TypeError(f"Unrecognised type: {type(axis_config)}")

    def plot_ground_truth(self):
        X_gt = self._generate_uniform_grid()
        input_mask = self.bo.objective.is_input_feasible(X_gt)
        Y_obj = self.bo.objective.evaluate_true_objective(X_gt)
        Y_con = self.bo.objective.evaluate_true_constraint(X_gt)
        Y_track = self.bo.objective.evaluate_trackers(X_gt)

        x_gt = self._get_data(self.x, X_gt, Y_obj, Y_con, Y_track)
        y_gt = self._get_data(self.y, X_gt, Y_obj, Y_con, Y_track)
        z_gt = self._get_data(self.z, X_gt, Y_obj, Y_con, Y_track)

        output_mask = self.bo.objective.is_output_feasible(Y_obj)
        is_feasible = torch.logical_and(input_mask, output_mask)
        is_infeasible = torch.logical_and(input_mask, torch.logical_not(output_mask))

        # Identify Pareto
        Y_max_space = Y_obj.clone()
        Y_max_space[..., self.bo.objective.to_minimize] *= -1
        is_pareto = torch.zeros_like(is_feasible, dtype=torch.bool)
        feasible_indices = torch.where(is_feasible)[0]
        if feasible_indices.numel() > 0:
            from botorch.utils.multi_objective.pareto import is_non_dominated
            pareto_sub_mask = is_non_dominated(Y_max_space[is_feasible])
            is_pareto[feasible_indices[pareto_sub_mask]] = True

        mask_exclusive_feasible = torch.logical_and(is_feasible, torch.logical_not(is_pareto))

        # === Render ===

        # Layer 1: Infeasible ground truth
        if is_infeasible.any():
            self.ax.scatter(
                x_gt[is_infeasible].cpu().numpy(),
                y_gt[is_infeasible].cpu().numpy(),
                **experiment_scatter_gnd_truth_infeasible
            )

        # Layer 2: Feasible dominated ground truth
        kwargs = experiment_scatter_gnd_truth_feasible.copy()
        kwargs.pop("facecolor")
        if mask_exclusive_feasible.any():
            self.ax.scatter(
                x_gt[mask_exclusive_feasible].cpu().numpy(),
                y_gt[mask_exclusive_feasible].cpu().numpy(),
                c=z_gt[mask_exclusive_feasible].cpu().numpy() if z_gt is not None else
                experiment_scatter_gnd_truth_feasible[
                    "facecolor"],
                vmin=self.z.bounds[0] if self.z and self.z.bounds else None,
                vmax=self.z.bounds[1] if self.z and self.z.bounds else None,
                cmap=self.cmap if z_gt is not None else None,
                **kwargs
            )

        # Layer 3: Pareto ground truth
        kwargs = experiment_scatter_gnd_truth_pareto_front.copy()
        kwargs.pop("facecolor")
        if is_pareto.any():
            self.ax.scatter(
                x_gt[is_pareto].cpu().numpy(),
                y_gt[is_pareto].cpu().numpy(),
                c=z_gt[is_pareto].cpu().numpy() if z_gt is not None else experiment_scatter_gnd_truth_pareto_front[
                    "facecolor"],
                vmin=self.z.bounds[0] if self.z and self.z.bounds else None,
                vmax=self.z.bounds[1] if self.z and self.z.bounds else None,
                cmap=self.cmap if z_gt is not None else None,
                **kwargs
            )
        return self

    def plot_observations(self, zorder=4):
        X, Y_obj = self.bo.X, self.bo.Y_obj
        Y_con, Y_track = self.bo.Y_con, self.bo.Y_track

        x_obs = self._get_data(self.x, X, Y_obj, Y_con, Y_track)
        y_obs = self._get_data(self.y, X, Y_obj, Y_con, Y_track)
        z_obs = self._get_data(self.z, X, Y_obj, Y_con, Y_track)

        is_feasible = torch.logical_and(self.bo.objective.is_input_feasible(X),
                                        self.bo.objective.is_output_feasible(Y_obj))

        is_pareto = torch.zeros(X.shape[0], dtype=torch.bool, device=X.device)
        f_idx = torch.where(is_feasible)[0]
        if f_idx.numel() > 0:
            from botorch.utils.multi_objective.pareto import is_non_dominated
            Y_ms = Y_obj[is_feasible].clone()
            Y_ms[..., self.bo.objective.to_minimize] *= -1
            is_pareto[f_idx[is_non_dominated(Y_ms)]] = True

        mask_infeasible = torch.logical_not(is_feasible)
        mask_dominated = torch.logical_and(is_feasible, torch.logical_not(is_pareto))

        # Render Observations
        if mask_infeasible.any():
            self.ax.scatter(x_obs[mask_infeasible].cpu().numpy(), y_obs[mask_infeasible].cpu().numpy(),
                            zorder=zorder, **experiment_scatter_observations_infeasible)

        if mask_dominated.any():
            self.ax.scatter(
                x_obs[mask_dominated].cpu().numpy(),
                y_obs[mask_dominated].cpu().numpy(),
                c=z_obs[mask_dominated].cpu().numpy() if z_obs is not None else None,
                vmin=self.z.bounds[0] if self.z and self.z.bounds else None,
                vmax=self.z.bounds[1] if self.z and self.z.bounds else None,
                cmap=self.cmap if z_obs is not None else None,
                edgecolors="black",
                zorder=zorder, s=20
            )

        if is_pareto.any():
            self.ax.scatter(
                x_obs[is_pareto].cpu().numpy(),
                y_obs[is_pareto].cpu().numpy(),
                c=z_obs[is_pareto].cpu().numpy() if z_obs is not None else None,
                vmin=self.z.bounds[0] if self.z and self.z.bounds else None,
                vmax=self.z.bounds[1] if self.z and self.z.bounds else None,
                cmap=self.cmap if z_obs is not None else None,
                edgecolors='black', linewidths=1.5, s=50, zorder=zorder + 1, label='Observed Pareto'
            )
        return self

    def plot(self):
        self.plot_ground_truth()
        self.plot_observations()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)
