from typing import List
import torch
from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase


class SingleObjectivePlotter(PlotterBase):
    """
    Visualiser for single-objective optimization problems (1D).
    It handles ground truth, GP posterior, and experimental observations.
    """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            lims: List[tuple[float, float]] | None = None,
            labels: list[str] | None = (r"$x$", r"$f(x)$"),
    ):
        super().__init__(bayesian_optimizer=bayesian_optimizer, lims=lims, labels=labels)
        self.n_grid_points = 500  # High resolution for smooth plotting

    def plot_ground_truth(self, zorder: int = 1):
        """Plots the true objective function as a dashed red line."""
        # Generate 1D grid using the base class method
        X_grid = self._generate_grid(n_grid_points=self.n_grid_points)

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

        X_grid = self._generate_grid(n_grid_points=self.n_grid_points)

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

        # Strictly lock legend to the upper right corner
        self.ax.legend(loc='upper right', fontsize='small', frameon=True)
        return self


class TwoVariablesOneObjective(PlotterBase):
    """
    2D Plotter that handles initial random sampling batches.
    It shows all initial points pointing towards the first Bayesian-suggested point,
    then follows a sequential trajectory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_grid_points = 100

    def _generate_grid(self, n_grid_points=100) -> torch.Tensor:
        """Generate a 2D grid for the background contour plot."""
        bounds = self.bo.objective.bounds
        x = torch.linspace(bounds[0, 0], bounds[1, 0], n_grid_points, device=self.bo.device)
        y = torch.linspace(bounds[0, 1], bounds[1, 1], n_grid_points, device=self.bo.device)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def plot_objective(self):
        """Plot the ground truth surface."""
        X_grid = self._generate_grid(self.n_grid_points)
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
            self.ax.scatter(X_f[:, 0].cpu(), X_f[:, 1].cpu(), c='white',
                            edgecolors='black', s=40, label='Feasible', zorder=zorder)
        X_i, _ = self.bo.compute_infeasible_XY()
        if X_i is not None:
            self.ax.scatter(X_i[:, 0].cpu(), X_i[:, 1].cpu(), c='red',
                            marker='x', s=40, label='Infeasible', zorder=zorder)
        return self

    def plot_trajectory(self, zorder: int = 3):
        """
        Logic:
        1. All initial random points point to the 1st Bayesian point.
        2. Following points connect sequentially.
        """
        X = self.bo.X
        if X is None or X.shape[0] < 2:
            return self

        X_np = X.detach().cpu().numpy()
        # n_init is the number of random samples (preliminary data)
        n_init = getattr(self.bo, '_n_initial_samples', None)

        # --- STEP 1: Preliminary data arrows ---
        # If we have at least one Bayesian point (index n_init)
        if len(X_np) > n_init:
            first_bo_point = X_np[n_init]
            for i in range(n_init):
                start = X_np[i]
                self.ax.annotate(
                    '', xy=first_bo_point, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='white', lw=1, alpha=0.5,
                                    shrinkA=3, shrinkB=3),
                    zorder=zorder
                )

        # --- STEP 2: Sequential BO trajectory ---
        # Start arrows from the first BO point onwards
        for i in range(n_init, len(X_np) - 1):
            self.ax.annotate(
                '', xy=X_np[i + 1], xytext=X_np[i],
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5, alpha=0.8,
                                shrinkA=3, shrinkB=3, connectionstyle="arc3,rad=0.1"),
                zorder=zorder
            )

        # --- STEP 3: Candidate Point (The Future) ---
        if self.bo.new_X is not None:
            new_x_np = self.bo.new_X.detach().cpu().numpy()
            last_obs = X_np[-1]
            for nx in new_x_np:
                self.ax.annotate(
                    '', xy=nx, xytext=last_obs,
                    arrowprops=dict(arrowstyle='->', color='lime', lw=2, ls=':'),
                    zorder=zorder + 2
                )
                self.ax.scatter(nx[0], nx[1], c='lime', marker='*', s=150,
                                edgecolors='black', label='Next Candidate', zorder=zorder + 3)

        # Ensure arrows stay inside the axes
        bounds = self.bo.objective.bounds.cpu().numpy()
        self.ax.set_xlim(bounds[0, 0], bounds[1, 0])
        self.ax.set_ylim(bounds[0, 1], bounds[1, 1])
        return self

    def plot(self):
        """Main plotting pipeline."""
        self.plot_objective()
        self.plot_trajectory()
        self.plot_observations()
        self.ax.legend(loc='upper right', fontsize='x-small', frameon=True)
        return self
