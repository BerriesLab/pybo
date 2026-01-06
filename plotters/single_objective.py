import torch
from bayesian_optimizer.bayesian_optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase


class SingleObjectivePlotter(PlotterBase):
    """
    A class for visualizing a single objective optimization problem.
    It requires only minimal user input: most settings
    are automatomatically inferred from the passed BayesianOptimizer.

    Key features:
    - Plot feasible solutions.
    - Plot unfeasible solutions.
    - Mark optimal solution.
    - Plot expected improvement.
    """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            title: str | None = "Pareto front",
            # idx_x: int = 0,
            # idx_y: int = 1,
            pareto_idxs: list[int] | None = None,
            use_tracker: bool = False,
            X_gt: torch.Tensor | None = None,
    ):
        """
        :param bayesian_optimizer: The BayesianOptimizer object containing data to plot.
        :param title: The title of the plot.
        """
        super().__init__(title=title)
        self.bayesian_optimizer = bayesian_optimizer
        self.X_gt = X_gt
        self.n_grid_points = 200

    def _generate_grid(self) -> torch.Tensor:
        """Generate a dense grid over the input bounds for plotting."""
        bounds = self.bayesian_optimizer.objective.bounds
        device = self.bayesian_optimizer.device
        dtype = self.bayesian_optimizer.dtype
        dim = self.bayesian_optimizer.objective.dim

        if dim == 1:
            X_grid = torch.linspace(
                bounds[0, 0].item(),
                bounds[1, 0].item(),
                self.n_grid_points,
                device=device,
                dtype=dtype
            ).unsqueeze(-1)
        else:
            raise ValueError("GP mean/variance plotting only supported for 1D problems.")

        return X_grid

    def plot_ground_truth(self):
        Y_obj_gt = self.bayesian_optimizer.objective.evaluate_true_objective(self.X_gt)
        self.ax.scatter(x=self.X_gt, y=Y_obj_gt, c='r', s=1)

        # Y_con_gt = self.bayesian_optimizer.objective.evaluate_true_constraint(self.X_gt)

    def plot_objective(self):
        X, Y = self.bayesian_optimizer.compute_feasible()
        self.ax.scatter(x=X, y=Y, color="black")

        X, Y = self.bayesian_optimizer.compute_infeasible()
        self.ax.scatter(X, Y, color="red")

        # # === Update colormaps ===
        # if self.idx_color is not None:
        #     if self.cbar is None:
        #         self._add_colorbar()
        #     self._update_cmap_and_norm()

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
