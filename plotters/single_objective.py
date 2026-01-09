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
    ):
        """
        :param bayesian_optimizer: The BayesianOptimizer object containing data to plot.
        """
        super().__init__(lims=lims)
        self.bayesian_optimizer = bayesian_optimizer
        self.X_gt = X_gt
        self.n_grid_points = 500

    # TODO: extend grid generation to more than one input dimensions
    def _generate_grid(self) -> torch.Tensor:
        """Generate a dense grid over the input bounds for plotting."""
        bounds = self.bayesian_optimizer.objective.bounds
        device = self.bayesian_optimizer.device
        dtype = self.bayesian_optimizer.dtype

        X_grid = torch.linspace(
            bounds[0, 0].item(),
            bounds[1, 0].item(),
            self.n_grid_points,
            device=device,
            dtype=dtype
        ).unsqueeze(-1)

        return X_grid

    # TODO:
    # """ Generate samples for ground truth evaluation - random sampler or grid """
    # # When constraints apply to the input X, build the ground truth by using
    # # a random generator subject to constraints
    # X_gt = sampler.draw_samples(n=1000)

    def plot_ground_truth(self):
        X_gt = self._generate_grid()
        Y_obj_gt = self.bayesian_optimizer.objective.evaluate_true_objective(X_gt)
        if X_gt is not None and Y_obj_gt is not None:
            self.ax.scatter(
                x=X_gt.detach().cpu().numpy(),
                y=Y_obj_gt.detach().cpu().numpy(),
                c='r',
                s=1,
                label="Ground truth",
            )
        return self

    def plot_objective(self):
        X, Y = self.bayesian_optimizer.compute_feasible_XY()
        if X is not None and Y is not None:
            self.ax.scatter(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                facecolors='none',
                edgecolors='black',
                linewidths=1.0,
                label="Feasible Observations"
            )

        X, Y = self.bayesian_optimizer.compute_infeasible_XY()
        if X is not None and Y is not None:
            self.ax.scatter(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                facecolors='none',
                edgecolors='red',
                linewidths=1.0,
                label="Infeasible Observations"
            )

        # === Update legend ===
        self.ax.legend(handles=self.legend_elements, loc="best")

        return self

    def plot_mean(self, color: str = 'blue', linewidth: float = 2.0,
                  label: str = 'GP Mean'):
        """Plot the GP posterior mean."""
        if self.bayesian_optimizer.model is None:
            raise ValueError("Model must be fitted before plotting GP mean.")

        X_grid = self._generate_grid()

        with torch.no_grad():
            posterior = self.bayesian_optimizer.model.posterior(X_grid)
            mean = posterior.mean.squeeze()

        X_np = X_grid.squeeze().detach().cpu().numpy()
        mean_np = mean.detach().cpu().numpy()

        self.ax.plot(X_np, mean_np, color=color, linewidth=linewidth, label=label)

        return self

    def plot_confidence(self, sigma: float = 3.0, color: str = 'blue',
                        alpha: float = 0.2, label: str = None):
        """Plot the GP confidence region (±sigma standard deviations).

        :param sigma: Number of standard deviations for confidence region.
        :param color: Color for the shaded region.
        :param alpha: Transparency of the shaded region.
        :param label: Label for legend (default: '±{sigma}σ').
        """
        if self.bayesian_optimizer.model is None:
            raise ValueError("Model must be fitted before plotting GP confidence.")

        X_grid = self._generate_grid()

        with torch.no_grad():
            posterior = self.bayesian_optimizer.model.posterior(X_grid)
            mean = posterior.mean.squeeze()
            std = posterior.variance.squeeze().sqrt()

        X_np = X_grid.squeeze().detach().cpu().numpy()
        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()

        lower = mean_np - sigma * std_np
        upper = mean_np + sigma * std_np

        if label is None:
            label = f'±{sigma}σ'

        self.ax.fill_between(X_np, lower, upper, color=color, alpha=alpha, label=label)

        return self

    def plot_optimum(self):
        """Plot the optimal solution."""
        X, Y = self.bayesian_optimizer.best_feasible_X, self.bayesian_optimizer.best_feasible_Y
        if X is not None and Y is not None:
            self.ax.scatter(
                X.detach().cpu().numpy(),
                Y.detach().cpu().numpy(),
                facecolors='orange',
                edgecolors='black',
                linewidths=1.0,
                marker="D",
                label="Best Observation"
            )
        return self

    def plot_next_X(self):
        X = self.bayesian_optimizer.new_X.detach().cpu().numpy()
        if X is not None:
            if X.ndim == 0:
                X = [X.item()]
            for i, x in enumerate(X):
                self.ax.axvline(
                    x=x,
                    linestyle='--',
                    color='red',
                    alpha=0.7,
                    label="Next X" if i == 0 else None
                )
            return self

    def plot_legend(self, loc: str = "upper right"):
        self.ax.legend(loc=loc)
        return self

    def plot(self):
        self.plot_confidence()
        self.plot_ground_truth()
        self.plot_objective()
        self.plot_optimum()
        self.plot_next_X()
        self.plot_legend()
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = "experiment.png"
        return super().save_figure(filename=filename)
