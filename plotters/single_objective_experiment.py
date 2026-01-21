import torch
from typing import List
from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase

style_arrow_future = dict(
    arrowstyle='->',
    color='red',
    lw=1.5,
    alpha=0.8,
    shrinkA=3,
    shrinkB=3,
    connectionstyle="arc3,rad=0.1",
    ls='--'
)

style_arrow_past = dict(
    arrowstyle='->',
    color='white',
    lw=1.5,
    alpha=0.8,
    shrinkA=3,
    shrinkB=3,
    connectionstyle="arc3,rad=0.1",
)


class Experiment1DPlotter(PlotterBase):
    """
    Visualiser for single-objective optimization problems (1D).
    It handles ground truth, GP posterior, and experimental observations.
    """

    def __init__(self, bayesian_optimizer: BayesianOptimizer, lims: List[tuple[float, float]] or None = None):
        super().__init__(
            bayesian_optimizer=bayesian_optimizer,
            lims=lims,
            labels=(r"$x$", r"$f(x)$"),
            # title="Experiment"
        )
        self.n_grid_points = 100

    def plot_ground_truth(self, zorder: int = 1):
        """Plots the true objective function as a dashed red line."""
        # Generate 1D grid using the base class method
        X_grid = self._generate_uniform_grid(n_points_per_dim=self.n_grid_points)

        # Evaluate the true objective function
        Y_obj_gt = self.bo.objective.evaluate_true_objective(X_grid)

        if X_grid is not None and Y_obj_gt is not None:
            self.ax.plot(
                X_grid.detach().cpu().numpy(),
                Y_obj_gt.detach().cpu().numpy(),
                color='red',
                linestyle='--',
                linewidth=1.5,
                label="Ground Truth",
                zorder=zorder,
            )
        return self

    def plot_gp_posterior(self, sigma: float = 2.0, zorder: int = 2):
        """Plots the GP mean and the confidence interval (uncertainty)."""
        if self.bo.model is None:
            return self

        X_grid = self._generate_uniform_grid(n_points_per_dim=self.n_grid_points)

        with torch.no_grad():
            # Get posterior distribution from the GP model
            posterior = self.bo.model.posterior(X_grid)
            mean = posterior.mean.squeeze(-1)
            std = posterior.variance.sqrt().squeeze(-1)

        X_np = X_grid.squeeze().cpu().numpy()
        mean_np = mean.cpu().numpy()
        std_np = std.cpu().numpy()

        # Plot GP Mean
        self.ax.plot(X_np, mean_np, color='blue', label='GP Mean', zorder=zorder)

        # Plot Confidence Interval (shaded area)
        self.ax.fill_between(
            X_np,
            mean_np - sigma * std_np,
            mean_np + sigma * std_np,
            color='blue',
            alpha=0.2,
            label=f'Confidence ±{sigma}σ',
            zorder=zorder - 1
        )
        return self

    def plot_observations(self, zorder: int = 3):
        """Plots sampled points, distinguishing between feasible and infeasible ones."""
        # 1. Feasible Observations (Black circles)
        X_f, Y_f = self.bo.compute_feasible_XY()
        if X_f is not None:
            self.ax.scatter(
                X_f.detach().cpu().numpy(),
                Y_f.detach().cpu().numpy(),
                color='black',
                marker='o',
                facecolors='none',
                label="Feasible Obs.",
                zorder=zorder,
            )

        # 2. Infeasible Observations (Red crosses - violated constraints)
        X_i, Y_i = self.bo.compute_infeasible_XY()
        if X_i is not None:
            self.ax.scatter(
                X_i.detach().cpu().numpy(),
                Y_i.detach().cpu().numpy(),
                color='red',
                marker='x',
                label="Infeasible (Violated)",
                zorder=zorder,
            )
        return self

    def plot_next_X(self, zorder: int = 5):
        """ Draws vertical lines at the coordinates suggested for the next iteration. """
        new_x = self.bo.new_X

        if new_x is not None:
            # Flatten to handle batch acquisition (q > 1)
            new_x_np = new_x.detach().cpu().numpy().flatten()

            for i, x in enumerate(new_x_np):
                label = "Next Sample (New X)" if i == 0 else None
                self.ax.axvline(
                    x=x,
                    color='green',  # Using green to distinguish from red ground truth
                    linestyle=':',
                    linewidth=2,
                    label=label,
                    zorder=zorder
                )
        return self

    def plot_optimum(self, zorder: int = 4):
        """Highlights the best feasible solution found so far."""
        X_best, Y_best = self.bo.best_feasible_X, self.bo.best_feasible_Y
        if X_best is not None:
            self.ax.scatter(
                X_best.detach().cpu().numpy(),
                Y_best.detach().cpu().numpy(),
                color='orange',
                marker='D',
                s=50,
                edgecolors='black',
                label="Best Found",
                zorder=zorder,
            )
        return self

    def plot(self):
        """Executes the full 1D plotting pipeline with legend locked at top-right."""
        self.plot_gp_posterior()
        self.plot_ground_truth()
        self.plot_observations()
        self.plot_optimum()
        self.plot_next_X()
        self.ax.legend(loc='upper right', fontsize='small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)


class Experiment2DPlotter(PlotterBase):

    def __init__(self, bayesian_optimizer: BayesianOptimizer, lims: List[tuple[float, float]] or None = None):
        super().__init__(
            bo=bayesian_optimizer,
            lims=lims,
            labels=(r"$x_1$", r"$x_2$"),
            # title="Experiment"
        )
        self.n_grid_points = 100

    def _generate_uniform_grid(self, n_points_per_dim=100) -> torch.Tensor:
        """Generate a 2D grid for the background contour plot."""
        bounds = self.bo.objective.bounds
        x = torch.linspace(bounds[0, 0], bounds[1, 0], n_points_per_dim, device=self.bo.device)
        y = torch.linspace(bounds[0, 1], bounds[1, 1], n_points_per_dim, device=self.bo.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def plot_objective(self):
        """Plot the ground truth surface."""
        X_grid = self._generate_uniform_grid(self.n_grid_points)
        Y_obj = self.bo.objective.evaluate_true_objective(X_grid)
        N = self.n_grid_points
        X_np = X_grid[:, 0].reshape(N, N).cpu().numpy()
        Y_np = X_grid[:, 1].reshape(N, N).cpu().numpy()
        Z_np = Y_obj.reshape(N, N).cpu().numpy()

        cp = self.ax.contourf(X_np, Y_np, Z_np, levels=50, cmap='viridis', alpha=0.8)
        if self.cbar is None:
            self.cbar = self.fig.colorbar(cp, ax=self.ax)
        return self

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
        # to the first q-batch (1), then connect iteratively the k-th batch to
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
        self.plot_objective()
        self.plot_new_X()
        self.plot_trajectory()
        self.plot_observations()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "experiment.png"
        return super().save_figure(filename=filename)
