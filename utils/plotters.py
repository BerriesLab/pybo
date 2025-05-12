import matplotlib.pyplot as plt
from botorch.utils.multi_objective import is_non_dominated
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from bayesian_optimization_for_gpu.samplers import draw_samples
from utils.io import *
from utils.types import OptimizationProblemType, SamplerType

observation_kwargs = {
    'color': "tab:orange",
    'marker': "o",
    's': 70,
    "alpha":0.6,
    "edgecolors": "black",
    'label': 'Observations'
}
observed_pareto_kwargs = {
    'color': 'black',
    'linestyle': '-',
    'linewidth': 2,
    'marker': 'o',
    'markersize': 8,
    'markerfacecolor': 'tab:orange',
    'markeredgecolor': 'black',
    'alpha': 0.9,
    'label': 'Observed Pareto Front'
}

ref_point_kwargs = {
    'color': 'red',
    'marker': 'x',
    's': 70,
    'alpha': 0.8,
    'label': 'Reference Point'
}
ground_truth_kwargs = {
    'color': "black",
    'marker': "o",
    's': 10,
    "alpha": 0.1
}
posterior_pareto_kwargs = {
    'color': 'tab:green',
    'linestyle': '-',
    'linewidth':2,
    'marker': 'o',
    'markersize': 8,
    'markerfacecolor': 'tab:green',
    'markeredgecolor': 'black',
    'alpha': 0.6,
    'label': 'Posterior Pareto Front'
}
pareto_surface_kwargs = {
    'alpha':0.4,
    'edgecolors':"black",
    'linewidth':0.1
}
posterior_surface_kwargs = {
    'alpha': 0.4,
    'edgecolors': 'black',
    'linewidths':0.1
}


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

#TODO: add 1st 2nd 3rd std dev to posterior plot
def plot_multi_objective_from_RN_to_R2(
        mobo: Mobo,
        f1_lims=None,
        f2_lims=None,
        ground_truth=False,
        posterior=False,
        ref_point=False,
        show=True):
    """ X is an 'n x d' feature matrix, Y is an 'n x 2' objective matrix, where n is the number of samples,
    and d is the number of dimensions. Pareto is a boolean array indicating which samples are Pareto optimal."""
    # Initialize figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.set_xlabel('$f_{01}$')
    axes.set_ylabel('$f_{02}$')
    axes.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^2$')
    if f1_lims is not None and isinstance(f1_lims, tuple):
        axes.set_xlim(f1_lims[0], f1_lims[1])
    if f2_lims is not None and isinstance(f2_lims, tuple):
        axes.set_ylim(f2_lims[0], f2_lims[1])

    # Plot ground truth if requested
    if ground_truth:
        if not mobo.get_true_objective():
            raise ValueError("Ground truth not available.")

        # Draw a number of random samples to plot the ground truth
        dims = mobo.get_X().shape[-1]
        x = draw_samples(
            sampler_type=SamplerType.Sobol,
            bounds=mobo.get_bounds().cpu(),
            n_samples=int(1e3),
            n_dimensions=dims
        )
        x = unnormalize(x, mobo.get_bounds().cpu())
        # Evaluate the objective function
        y = mobo.get_true_objective()(x)
        # Plot points in 3D
        axes.scatter(y[:, 0], y[:, 1], **ground_truth_kwargs)

    if ref_point:
        ref_point = mobo.get_ref_point().cpu().numpy()
        plt.scatter(ref_point[0], ref_point[1], **ref_point_kwargs)

    if posterior:
        # Extract 1000 random samples
        dims = mobo.get_X().shape[-1]
        x = draw_samples(
            sampler_type=SamplerType.Sobol,
            bounds=mobo.get_bounds().cpu(),
            n_samples=int(1e3),
            n_dimensions=dims
        )
        x = unnormalize(x, mobo.get_bounds().cpu())
        x = x.to(mobo.get_X().device)  # Ensure that X_candidate is on the same device as X
        # Calculate posterior samples
        posterior = mobo.get_model().posterior(x)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        # samples = posterior.sample()

        # Calculate pareto front for mean and samples
        if mobo.get_optimization_problem_type() == OptimizationProblemType.Maximization:
            mean_mask = is_non_dominated(Y=mean, maximize=True)
            mean_pareto = mean[mean_mask].detach().cpu().numpy()
            std_pareto = std[mean_mask].detach().cpu().numpy()
            mean_sorted = np.argsort(-mean_pareto[:, 0])

        elif mobo.get_optimization_problem_type() == OptimizationProblemType.Minimization:
            mean_mask = is_non_dominated(Y=mean, maximize=False)
            mean_pareto = mean[mean_mask].detach().cpu().numpy()
            std_pareto = std[mean_mask].detach().cpu().numpy()
            mean_sorted = np.argsort(mean_pareto[:, 0])
        else:
            raise ValueError("Unknown optimization true_objective type.")

        mean_pareto = mean_pareto[mean_sorted]
        std_pareto = std_pareto[mean_sorted]

        # Plot std dev bands
        for i in range(3, 0, -1):
            axes.fill_between(mean_pareto[:, 0],
                              mean_pareto[:, 1] - i * std_pareto[:, 1],
                              mean_pareto[:, 1] + i * std_pareto[:, 1],
                              alpha=0.2,
                              color='tab:green',
                              label=f'{i}σ' if i == 1 else None)

        axes.plot(mean_pareto[:, 0], mean_pareto[:, 1], **posterior_pareto_kwargs)


    # Plot observations except Pareto front
    y = mobo.get_Yobj().cpu().numpy()
    mask = mobo.get_pareto_mask().cpu().numpy()
    inverted_mask = np.invert(mask)
    if len(inverted_mask) > 0:
        axes.scatter(y[inverted_mask, 0], y[inverted_mask, 1], **observation_kwargs)

    # Plot Pareto Front
    if len(mask) > 0:
        pareto = y[mask]
        # Sort points by x-coordinate (f1)
        if mobo.get_optimization_problem_type() == OptimizationProblemType.Minimization:
            sorted_indices = np.argsort(pareto[:, 0])
        elif mobo.get_optimization_problem_type() == OptimizationProblemType.Maximization:
            sorted_indices = np.argsort(-pareto[:, 0])
        else:
            raise ValueError("Unknown optimization true_objective type.")
        pareto = pareto[sorted_indices]
        axes.plot(pareto[:, 0], pareto[:, 1], **observed_pareto_kwargs)

    # Add legend
    plt.legend()

    filepath = compose_figure_filename()
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)


# TODO: implement a parallel coordinate plot
def plot_multi_objective_from_RN_to_R3(mobo: Mobo, ground_truth=False, posterior=False, show=True):
    """ X is an 'n x d' feature matrix, Y is an 'n x 3' objective matrix, where n is the number of samples,
    and d is the number of dimensions. Pareto is a boolean array indicating which samples are Pareto optimal."""
    # Initialize figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={'projection': '3d'})
    fig.suptitle(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^3$')
    # Define the views
    views = {
        'Iso': (30, 30, 'z'),
        'XY': (90, -90, 'z'),
        'XZ': (0, -90, 'y'),
        'YZ': (0, 0, 'x'),
    }

    # Titles and positions for each subplot
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]



    for title, (i, j) in zip(views.keys(), positions):
        ax = axes[i, j]
        ax.set_xlabel('$f_{01}$')
        ax.set_ylabel('$f_{02}$')
        ax.set_zlabel('$f_{03}$')
        ax.set_title(title)

        # Set the view angle
        elev, azim, vertical_ax = views[title]
        ax.view_init(elev=elev, azim=azim, vertical_axis=vertical_ax)

        # Plot ground truth if requested
        if ground_truth:
            if not mobo.get_true_objective():
                raise ValueError("Ground truth not available.")

            # Let's draw a number of random samples to plot the ground truth
            dims = mobo.get_true_objective().num_objectives
            sobol = SobolEngine(dimension=dims, scramble=True)
            x = sobol.draw(n=int(1e4))
            # Evaluate the objective function
            y = mobo.get_true_objective()(x)
            # Plot points in 3D
            ax.scatter(y[:, 0], y[:, 1], y[:, 2], **ground_truth_kwargs)


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
                raise ValueError("Unknown optimization true_objective type.")
            pareto = y[pareto_mask]
            pareto = pareto.cpu().numpy()
            hull = ConvexHull(pareto)
            surface_simplices = hull.simplices
            # Light source direction (normalized)
            light_dir = np.array([1, 1, 2])
            light_dir = light_dir / np.linalg.norm(light_dir)
            vertices = [pareto[simplex] for simplex in surface_simplices]
            face_colors = []
            for simplex in hull.simplices:
                v0, v1, v2 = pareto[simplex]  # Get the three vertices
                # Calculate normal vector
                normal = np.cross(v1 - v0, v2 - v0)
                normal = normal / np.linalg.norm(normal)
                # Calculate shading
                shade = np.dot(normal, light_dir)
                shade = np.clip(shade, 0.3, 0.7)
                color = plt.cm.Greens(shade)
                face_colors.append(color)
            poly_collection = Poly3DCollection(vertices, **posterior_surface_kwargs, facecolors=face_colors)
            ax.add_collection3d(poly_collection)
            ax.scatter(pareto[:, 0], pareto[:, 1], pareto[:, 2], **posterior_kwargs, label="Posterior Pareto Front")

        # Bring inputs to CPU
        Y = mobo.get_Yobj().cpu().numpy()
        pareto = mobo.get_pareto().cpu().numpy()

        # Plot observations except Pareto front
        mask = np.ones(len(Y), dtype=bool)
        mask[pareto] = False
        ax.scatter(Y[mask, 0], Y[mask, 1], Y[mask, 2], **observation_kwargs, label='Observations')

        # Plot Pareto Front
        mask = np.invert(mask)
        pareto = Y[mask]
        hull = ConvexHull(pareto)
        surface_simplices = hull.simplices
        # Light source direction (normalized)
        light_dir = np.array([1, 1, 2])
        light_dir = light_dir / np.linalg.norm(light_dir)
        vertices = [pareto[simplex] for simplex in surface_simplices]
        face_colors = []
        for simplex in hull.simplices:
            v0, v1, v2 = pareto[simplex]  # Get the three vertices
            # Calculate normal vector
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)
            # Calculate shading
            shade = np.dot(normal, light_dir)
            shade = np.clip(shade, 0.3, 0.7)
            color = plt.cm.Oranges(shade)
            face_colors.append(color)
        poly_collection = Poly3DCollection(vertices, **pareto_surface_kwargs, facecolors=face_colors)
        ax.add_collection3d(poly_collection)
        ax.scatter(Y[mask, 0], Y[mask, 1], Y[mask, 2], **pareto_kwargs, label='Pareto Front')

        # Add legend
        if i == 0 and j == 0:
            ax.legend()

    plt.tight_layout()
    filepath = compose_figure_filename(postfix='_multi_objective')
    fig.savefig(filepath + ".png", dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)


def plot_log_hypervolume_difference(mobo: Mobo, show=False):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("number of observations (beyond initial points)")
    ax.set_ylabel("Log Hypervolume Difference")
    x = np.array(range(len(mobo.get_hypervolume()))) * mobo.get_batch_size()
    hv = np.array(mobo.get_hypervolume())
    max_hv = mobo.get_true_objective().max_hv
    dy = np.log10(max_hv - hv)
    ax.errorbar(x, dy, linewidth=2,)

    plt.tight_layout()
    filepath = compose_figure_filename(postfix="_hv_diff")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)