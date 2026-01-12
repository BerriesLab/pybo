from pathlib import Path
from typing import List

import torch
from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase


class SingleObjectivePlotter(PlotterBase):
    """
    A class for visualizing a single objective optimization problem.
    It requires only minimal user input: most settings
    are automatically inferred from the passed BayesianOptimizer.

    Key features:
    - Plot feasible solutions.
    - Plot unfeasible solutions.
    - Mark optimal solution.
    - Plot expected improvement.
    """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            use_tracker: bool = False,
            X_gt: torch.Tensor | None = None,
            lims: List[tuple[float, float]] | None = None,
            labels: list[str] | None = (r"$x$", r"$f(x)$"),
    ):
        """
        :param bayesian_optimizer: The BayesianOptimizer object containing data to plot.
        """
        super().__init__(bayesian_optimizer=bayesian_optimizer, lims=lims, labels=labels)
        self.X_gt = X_gt
        self.n_grid_points = 500

    def plot_ground_truth(self, zorder: int = 2):
        X_gt = self._generate_grid()
        Y_obj_gt = self.bo.objective.evaluate_true_objective(X_gt)
        if X_gt is not None and Y_obj_gt is not None:
            self.ax.scatter(
                x=X_gt.detach().cpu().numpy(),
                y=Y_obj_gt.detach().cpu().numpy(),
                c='r',
                s=1,
                label="Ground truth",
                zorder=zorder,
            )
        return self

    def plot_objective(self, zorder: int = 3):
        X, Y = self.bo.compute_feasible_XY()
        if X is not None and Y is not None:
            self.ax.scatter(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                facecolors='none',
                edgecolors='black',
                linewidths=1.0,
                label="Feasible Observations",
                zorder=zorder,
            )

        X, Y = self.bo.compute_infeasible_XY()
        if X is not None and Y is not None:
            self.ax.scatter(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                facecolors='none',
                edgecolors='red',
                linewidths=1.0,
                label="Infeasible Observations",
                zorder=zorder,
            )
        return self

    def plot_mean(self, color: str = 'blue', linewidth: float = 2.0,
                  label: str = 'Mean', zorder: int = 1):
        """Plot the GP posterior mean."""
        if self.bo.model is None:
            raise ValueError("Model must be fitted before plotting GP mean.")

        X_grid = self._generate_grid()

        with torch.no_grad():
            posterior = self.bo.model.posterior(X_grid)
            mean = posterior.mean.squeeze()

        X_np = X_grid.squeeze().detach().cpu().numpy()
        mean_np = mean.detach().cpu().numpy()

        self.ax.plot(X_np, mean_np, color=color, linewidth=linewidth, label=label)

        return self

    def plot_confidence(self, sigma: float = 3.0, color: str = 'blue',
                        alpha: float = 0.2, label: str = None, zorder: int = 0):
        """Plot the GP confidence region (±sigma standard deviations).

        :param sigma: Number of standard deviations for confidence region.
        :param color: Color for the shaded region.
        :param alpha: Transparency of the shaded region.
        :param label: Label for legend (default: '±{sigma}σ').
        :param zorder: stack order.
        """
        if self.bo.model is None:
            raise ValueError("Model must be fitted before plotting GP confidence.")

        X_grid = self._generate_grid()

        with torch.no_grad():
            posterior = self.bo.model.posterior(X_grid)
            mean = posterior.mean.squeeze()
            std = posterior.variance.squeeze().sqrt()

        X_np = X_grid.squeeze().detach().cpu().numpy()
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()

        lower = mean_np - sigma * std_np
        upper = mean_np + sigma * std_np

        if label is None:
            label = f'±{sigma}σ'

        self.ax.fill_between(X_np, lower, upper, color=color, alpha=alpha, label=label, zorder=zorder)

        return self

    def plot_optimum(self, zorder: int = 4):
        """Plot the optimal solution."""
        X, Y = self.bo.best_feasible_X, self.bo.best_feasible_Y
        if X is not None and Y is not None:
            self.ax.scatter(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                facecolors='orange',
                edgecolors='black',
                linewidths=1.0,
                marker="D",
                label="Best Observation",
                zorder=zorder,
            )
        return self

    def plot_next_X(self, zorder: int = 5):
        X = self.bo.new_X.detach().cpu().numpy()
        if X is not None:
            if X.ndim == 0:
                X = [X.item()]
            for i, x in enumerate(X):
                self.ax.axvline(
                    x=x,
                    linestyle='--',
                    color='red',
                    alpha=0.7,
                    label="Next X" if i == 0 else None,
                    zorder=zorder,
                )
            return self

    def plot_legend(self, loc: str = "upper right"):
        self.ax.legend(loc=loc)
        return self

    def plot(self):
        self.plot_confidence()
        self.plot_mean()
        self.plot_ground_truth()
        self.plot_objective()
        self.plot_optimum()
        self.plot_next_X()
        self.plot_legend()
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = "objective.png"
        return super().save_figure(filename=filename)
