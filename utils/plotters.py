import matplotlib.pyplot as plt
from botorch.utils.multi_objective import is_non_dominated
from torch.quasirandom import SobolEngine

from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from utils.io import *
from utils.types import OptimizationProblemType

pareto_kwargs = {'color': "tab:orange", 'marker': "x", 's': 100}
observation_kwargs = {'color': "tab:blue", 'marker': "o", 's': 70, "alpha":0.8, "edgecolors": "black"}
ground_truth_kwargs = {'color': "black", 'marker': "o", 's': 10, "alpha": 0.1}
posterior_kwargs = {'color': "tab:green", 'marker': "o", 's': 70, "alpha": 0.5, "edgecolors": "black"}


# def _plot(self):
#     # Handle single-objective cases
#     if self._n_objectives == 1:
#
#         if len(self._bounds) == 1:
#             self._plot_from_R1_to_R1()
#         elif len(self._bounds) == 2:
#             self._plot_from_R2_to_R1()
#         else:
#             raise ValueError("Only 1D and 2D plots are supported.")
#
#     # Handle bi-objective cases
#     elif self._n_objectives == 2:
#         self.plot_multi_objective_from_RN_to_R2()
#
#     # Handle tri-objective cases
#     elif self._n_objectives == 3:
#         self.plot_multi_objective_from_RN_to_R3()
#
#     else:
#         raise ValueError("Cannot display pareto front for more than 3-objectives without PCA.")
#
#
# def _plot_from_R1_to_R1(self):
#     # Initialize figure
#     self._fig, self._ax = plt.subplots()
#     self._ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R} \rightarrow \mathbb{R}$')
#     self._ax.set_xlabel(r"$\mathcal{X}$")
#     self._ax.set_ylabel(r"$\mathcal{Y}$")
#     self._ax.set_xlim(self._bounds[0][0] * 1.1, self._bounds[0][1] * 1.1)
#
#     # Build domain
#     self._build_domain_from_bounds()
#
#     # Plot objective function if known
#     if self._f0:
#         self._ax.plot(self._domain, self._f0[0](self._domain), color="black", linestyle="--", label=r"True $f_0$")
#
#     # Plot posterior
#     self._predict_gaussian_process_on_domain()
#     self._ax.plot(self._domain, self._mu[0], label="Mean", zorder=2)
#     for i in range(1, 4):
#         self._ax.fill_between(x=self._domain.flatten(),
#                               y1=self._mu[0] - i * self._sigma[0],
#                               y2=self._mu[0] + i * self._sigma[0],
#                               alpha=0.2 / i,
#                               color="blue",
#                               label=rf"{i}$\sigma$")
#
#     # Plot new location
#     y_min, y_max = self._ax.get_ylim()
#     self._ax.vlines(self._new_X, ymin=y_min, ymax=y_max, color='red', alpha=0.3, linestyle="--", label="New X")
#
#     # Plot all observations except minimum
#     min_y_idx = np.argmin(self._Y)
#     mask = np.ones(len(self._Y), dtype=bool)
#     mask[min_y_idx] = False
#     self._ax.scatter(self._X[mask], self._Y[mask], marker="o", s=50, color='red', label='Observations')
#
#     # Plot minimum Y value
#     self._ax.scatter(self._X[min_y_idx], self._Y[min_y_idx], marker='*', s=200, color='green', label='Min Y')
#     self._ax.legend()
#
#
# def _plot_from_R2_to_R1(self):
#     # Initialize figure
#     self._fig = plt.figure()
#     self._ax = self._fig.add_subplot(111, projection='3d')
#     self._ax.set_title(r'Bayesian Optimization for $f_0:\mathbb{R}^2 \rightarrow \mathbb{R}$')
#     self._ax.set_xlabel('$x_1$')
#     self._ax.set_ylabel('$x_2$')
#     self._ax.set_zlabel('$f(\mathcal{X})$')
#     self._ax.set_xlim(self._bounds[0][0], self._bounds[0][1])
#     self._ax.set_ylim(self._bounds[1][0], self._bounds[1][1])
#
#     # Build domain
#     self._build_domain_from_bounds()
#
#     # Plot objective function if known
#     if self._f0 and all([isinstance(f, Callable) for f in self._f0]):
#         x_grid = np.linspace(self._bounds[0][0], self._bounds[0][1], 100)
#         y_grid = np.linspace(self._bounds[1][0], self._bounds[1][1], 100)
#         X, Y = np.meshgrid(x_grid, y_grid)
#         points = np.c_[X.ravel(), Y.ravel()]
#         Z = self._f0[0](points).reshape(X.shape)
#         self._ax.plot_wireframe(X, Y, Z, lw=0.5, alpha=0.4, color='black', label=r"True $f_0$")
#
#     # Plot new X location with vertical line
#     z_line = np.linspace(0, self._ax.get_zlim()[1], 100)
#     x_line = np.full_like(z_line, self._new_X[0, 0])
#     y_line = np.full_like(z_line, self._new_X[0, 1])
#     self._ax.plot(x_line, y_line, z_line, 'r--', alpha=0.5, label='New Location')
#
#     # Plot all observations except minimum
#     min_y_idx = np.argmin(self._Y)
#     mask = np.ones(len(self._Y), dtype=bool)
#     mask[min_y_idx] = False
#     self._ax.scatter(self._X[mask, 0], self._X[mask, 1], self._Y[mask, 0], c='red', marker='o', s=50,
#                      label='Observations')
#
#     # Plot minimum Y value
#     mask = np.invert(mask)
#     self._ax.scatter(self._X[mask, 0], self._X[mask, 1], self._Y[mask, 0], c='green', marker='*', s=200, label='Min Y')
#
#     # Plot posterior
#     self._predict_gaussian_process_on_domain()
#     x_grid = self._domain[:, 0]
#     y_grid = self._domain[:, 1]
#     X, Y = np.meshgrid(x_grid, y_grid)
#     for i in range(1, 4):
#         self._ax.plot_surface(X, Y, (self._mu[0] - i * self._sigma[0]).reshape(X.shape),
#                               lw=0.5, rstride=8, cstride=8,
#                               alpha=0.2, cmap='coolwarm',
#                               zorder=2)
#         self._ax.plot_surface(X, Y, (self._mu[0] + i * self._sigma[0]).reshape(X.shape),
#                               lw=0.5, rstride=8, cstride=8,
#                               alpha=0.2, cmap='coolwarm',
#                               zorder=3)


# TODO: the pareto front and posterior pareto should be a line in a 2d space
def plot_multi_objective_from_RN_to_R2(mobo: Mobo, ground_truth=False, posterior=False, show=True):
    """ X is an 'n x d' feature matrix, Y is an 'n x 2' objective matrix, whre n is the number of samples,
    and d is the number of dimensions. Pareto is a boolean array indicating which samples are Pareto optimal."""
    # Initialize figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    cm = plt.get_cmap('coolwarm')
    axes.set_xlabel('$f_{01}$')
    axes.set_ylabel('$f_{02}$')
    axes.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^2$')

    # Plot ground truth if requested
    if ground_truth:
        if not mobo.get_f0():
            raise ValueError("Ground truth not available.")

        # Let's draw a number of random samples to plot the ground truth
        dims = mobo.get_f0().num_objectives
        sobol = SobolEngine(dimension=dims, scramble=True)
        x = sobol.draw(n=int(1e4))
        # Evaluate the objective function
        y = mobo.get_f0()(x)
        # Plot points in 3D
        axes.scatter(y[:, 0], y[:, 1], **ground_truth_kwargs)

    if posterior:
        # Extract 1000 random samples
        sobol = SobolEngine(dimension=mobo.get_X().shape[1], scramble=True)
        x = sobol.draw(1000)  # 1000 candidates in [0, 1]^d
        lower_bounds = mobo.get_bounds()[0].cpu()
        upper_bounds = mobo.get_bounds()[1].cpu()
        x = lower_bounds + (upper_bounds - lower_bounds) * x
        x = x.to(mobo.get_X().device)  # Ensure that X_candidate is on the same device as X
        # Calculate posterior samples
        y = mobo.get_model().posterior(x).sample()
        # Calculate pareto front
        if mobo.get_optimization_problem_type() == OptimizationProblemType.Maximization:
            pareto_mask = is_non_dominated(Y=y, maximize=True)
        elif mobo.get_optimization_problem_type() == OptimizationProblemType.Minimization:
            pareto_mask = is_non_dominated(Y=y, maximize=False)
        else:
            raise ValueError("Unknown optimization problem type.")
        pareto = y[pareto_mask]
        pareto = pareto.cpu().numpy()
        axes.scatter(pareto[:, 0], pareto[:, 1], **posterior_kwargs, label="Posterior Pareto Front")

    # Bring inputs to CPU
    Y = mobo.get_Y().cpu().numpy()
    pareto = mobo.get_pareto().cpu().numpy()

    # Plot observations except Pareto front
    mask = np.ones(len(Y), dtype=bool)
    mask[pareto] = False
    axes.scatter(Y[mask, 0], Y[mask, 1], **observation_kwargs, label='Observations')

    # Plot Pareto Front
    mask = np.invert(mask)
    axes.scatter(Y[mask, 0], Y[mask, 1], **pareto_kwargs, label='Observed Pareto Front')

    # Add legend
    plt.legend()

    filepath = compose_figure_filename()
    fig.savefig(filepath + ".png", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)


# TODO: the pareto front should be a surface in a 3d Space
def plot_multi_objective_from_RN_to_R3(mobo: Mobo, ground_truth=False, posterior=False, show=True):
    """ X is an 'n x d' feature matrix, Y is an 'n x 3' objective matrix, where n is the number of samples,
    and d is the number of dimensions. Pareto is a boolean array indicating which samples are Pareto optimal."""
    # Initialize figure with subplots
    fig = plt.figure(figsize=(24, 8))

    # 3D plot
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.set_xlabel('$f_{01}$')
    ax1.set_ylabel('$f_{02}$')
    ax1.set_zlabel('$f_{03}$')
    ax1.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^3$')

    # 2D projections
    ax2 = fig.add_subplot(142)
    ax2.set_xlabel('$f_{01}$')
    ax2.set_ylabel('$f_{02}$')
    ax2.set_title('f1 vs f2 Projection')

    ax3 = fig.add_subplot(143)
    ax3.set_xlabel('$f_{02}$')
    ax3.set_ylabel('$f_{03}$')
    ax3.set_title('f2 vs f3 Projection')

    ax4 = fig.add_subplot(144)
    ax4.set_xlabel('$f_{01}$')
    ax4.set_ylabel('$f_{03}$')
    ax4.set_title('f1 vs f3 Projection')

    # Plot ground truth if requested
    if ground_truth:
        if not mobo.get_f0():
            raise ValueError("Ground truth not available.")

        # Let's draw a number of random samples to plot the ground truth
        dims = mobo.get_f0().num_objectives
        sobol = SobolEngine(dimension=dims, scramble=True)
        x = sobol.draw(n=int(1e4))
        # Evaluate the objective function
        y = mobo.get_f0()(x)
        # Plot points in 3D
        ax1.scatter(y[:, 0], y[:, 1], y[:, 2], **ground_truth_kwargs)
        ax2.scatter(y[:, 0], y[:, 1], **ground_truth_kwargs)
        ax3.scatter(y[:, 1], y[:, 2], **ground_truth_kwargs)
        ax4.scatter(y[:, 0], y[:, 2], **ground_truth_kwargs)

    if posterior:
        # Extract 1000 random samples
        sobol = SobolEngine(dimension=mobo.get_X().shape[1], scramble=True)
        x = sobol.draw(1000)  # 1000 candidates in [0, 1]^d
        lower_bounds = mobo.get_bounds()[0].cpu()
        upper_bounds = mobo.get_bounds()[1].cpu()
        x = lower_bounds + (upper_bounds - lower_bounds) * x
        x = x.to(mobo.get_X().device)  # Ensure that X_candidate is on the same device as X
        # Calculate posterior samples
        y = mobo.get_model().posterior(x).sample()
        # Calculate pareto front
        if mobo.get_optimization_problem_type() == OptimizationProblemType.Maximization:
            pareto_mask = is_non_dominated(Y=y, maximize=True)
        elif mobo.get_optimization_problem_type() == OptimizationProblemType.Minimization:
            pareto_mask = is_non_dominated(Y=y, maximize=False)
        else:
            raise ValueError("Unknown optimization problem type.")
        pareto = y[pareto_mask]
        pareto = pareto.cpu().numpy()
        ax1.scatter(pareto[:, 0], pareto[:, 1], pareto[:, 2], **posterior_kwargs, label="Posterior Pareto Front")
        ax2.scatter(pareto[:, 0], pareto[:, 1], **posterior_kwargs)
        ax3.scatter(pareto[:, 1], pareto[:, 2], **posterior_kwargs)
        ax4.scatter(pareto[:, 0], pareto[:, 2], **posterior_kwargs)


    # Bring inputs to CPU
    Y = mobo.get_Y().cpu().numpy()
    pareto = mobo.get_pareto().cpu().numpy()

    # Plot observations except Pareto front
    mask = np.ones(len(Y), dtype=bool)
    mask[pareto] = False
    ax1.scatter(Y[mask, 0], Y[mask, 1], Y[mask, 2], **observation_kwargs, label='Observations')
    ax2.scatter(Y[mask, 0], Y[mask, 1], **observation_kwargs)
    ax3.scatter(Y[mask, 1], Y[mask, 2], **observation_kwargs)
    ax4.scatter(Y[mask, 0], Y[mask, 2], **observation_kwargs)

    # Plot Pareto Front
    mask = np.invert(mask)
    ax1.scatter(Y[mask, 0], Y[mask, 1], Y[mask, 2], **pareto_kwargs, label='Pareto Front')
    ax2.scatter(Y[mask, 0], Y[mask, 1], **pareto_kwargs)
    ax3.scatter(Y[mask, 1], Y[mask, 2], **pareto_kwargs)
    ax4.scatter(Y[mask, 0], Y[mask, 1], **pareto_kwargs)

    # Add legend to main 3D plot
    ax1.legend()

    plt.tight_layout()
    filepath = compose_figure_filename()
    fig.savefig(filepath + ".png", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)
