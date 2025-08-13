import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.tri import Triangulation
import matplotlib as mpl
from pybo.mobo.mobo import Mobo
from matplotlib.lines import Line2D


ms = 7

feasible_non_pareto_observations_kwargs = {
    'color': "tab:green",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Obs. Non-Pareto'
}

infeasible_observations_kwargs = {
    'color': "tab:red",
    'marker': "x",
    's': ms ** 2,
    "alpha": 0.7,
    # "edgecolors": "black",
    'label': 'Obs. Infeasible'
}

feasible_pareto_observations_kwargs = {
    'marker': 'D',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolors': 'black',
    'alpha': 0.7,
    'label': 'Obs. Pareto Front'
}

ref_point_kwargs = {
    'color': 'tab:red',
    "edgecolors": "black",
    'marker': 's',
    's': ms ** 2,
    'alpha': 0.7,
    'label': 'Ref. Point'
}

ground_truth_feas_kwargs = {
    'color': "black",
    'marker': "o",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Ground truth feas.'
}

ground_truth_inf_kwargs = {
    'color': "red",
    'marker': "x",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Ground truth inf.'
}

posterior_pareto_kwargs = {
    'fmt': 'o',
    'edgecolors': 'tab:blue',
    'alpha': 0.3,
    'label': r'Post. $\mu \pm 3 \sigma$',
    'capsize': 3,
}

xy_plot_kwargs = {
    'marker': 'o',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolors': 'black',
    'alpha': 1,
}


def make_grid(size: int, bounds: torch.Tensor, dtype=torch.float64, device='cpu'):
    """
    Creates a grid of points within the given bounds.

    Args:
        size (int): Number of points per dimension.
        bounds (torch.Tensor): Tensor of shape (2, d) containing min and max for each dimension.
        dtype: Torch dtype.
        device: Torch device.

    Returns:
        torch.Tensor: Grid of shape (size**d, d).
    """
    if bounds.ndim != 2 or bounds.shape[0] != 2:
        raise ValueError(f"Bounds must be of shape (2, d), but got {bounds.shape}")

    bounds = bounds.transpose(0, 1)  # Convert to (d, 2)
    dim = bounds.shape[0]

    axes = [
        torch.linspace(bounds[i, 0], bounds[i, 1], size, dtype=dtype, device=device)
        for i in range(dim)
    ]
    mesh = torch.meshgrid(*axes, indexing='ij')
    grid = torch.stack(mesh, dim=-1).reshape(-1, dim)
    return grid


def build_custom_legend_elements(
    show_ref_point=False,
    show_observations=False,
    feas_non_par=False,
    feas_par=False,
    infeas=False,
    show_ground_truth=False,
    gt_feas=False,
    gt_infeas=False
):
    legend_elements = []

    if show_observations:
        if infeas:
            legend_elements.append(Line2D(
                [0], [0], marker=infeasible_observations_kwargs['marker'],
                label=infeasible_observations_kwargs['label'],
                color=infeasible_observations_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

        if feas_non_par:
            legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_observations_kwargs['marker'],
                label=feasible_non_pareto_observations_kwargs['label'],
                color=feasible_non_pareto_observations_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

        if feas_par:
            legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_observations_kwargs['marker'],
                label=feasible_pareto_observations_kwargs['label'],
                color=feasible_pareto_observations_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

    if show_ref_point:
        legend_elements.append(Line2D(
            [0], [0], marker=ref_point_kwargs['marker'],
            label=ref_point_kwargs['label'],
            color=ref_point_kwargs['edgecolors'],
            markerfacecolor='none', markersize=ms, linestyle='None'))

    if show_ground_truth:
        if gt_feas:
            legend_elements.append(Line2D(
                [0], [0], marker=ground_truth_feas_kwargs['marker'],
                label=ground_truth_feas_kwargs['label'],
                color=ground_truth_feas_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

        if gt_infeas:
            legend_elements.append(Line2D(
                [0], [0], marker=ground_truth_inf_kwargs['marker'],
                label=ground_truth_inf_kwargs['label'],
                color=ground_truth_inf_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

    return legend_elements

def plot_objective_from_R1_to_R1():
    raise NotImplementedError("Function not yet implemented.")


def plot_objective_from_R2_to_R1():
    raise NotImplementedError("Function not yet implemented.")


def plot_multi_objective_from_RN_to_R2(
        mobo: Mobo,
        ground_truth_X: torch.Tensor or None = None,
        title: str or None = None,
        f1_label: str or None = None,
        f2_label: str or None = None,
        f1_lims=None,
        f2_lims=None,
        show_ref_point=True,
        show_observations=True,
        display_figures=True,
        show_ground_truth=False,
        show_posterior=False,
        output_path=None,
):

    # Initialize the figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.set_title(r'MOBO for $\mathbf{f}:\mathbb{R}^N \rightarrow \mathbb{R}^2$') if not title else axes.set_title(title)
    axes.set_xlabel("$f_{1}$") if not f1_label else axes.set_xlabel(f1_label)
    axes.set_ylabel("$f_{2}$") if not f2_label else axes.set_ylabel(f2_label)
    axes.set_xlim(f1_lims[0], f1_lims[1]) if f1_lims is not None else axes.autoscale(enable=True, axis='x')
    axes.set_ylim(f2_lims[0], f2_lims[1]) if f2_lims is not None else axes.autoscale(enable=True, axis='y')

    """ Plot ground truth """
    if show_ground_truth:
        if not hasattr(mobo.get_objective(), "evaluate_true"):
            raise ValueError("Ground truth not available.")

        ground_truth_X = ground_truth_X.to(mobo.get_device(), mobo.get_dtype())
        ground_truth_obj = mobo.get_objective().evaluate_true(ground_truth_X)

        # Apply constraint mask
        if mobo.get_output_constraints() is None:
            ground_truth_feas_mask = torch.ones_like(ground_truth_obj, dtype=torch.bool).all(dim=-1)
        else:
            ground_truth_con = mobo.get_objective().evaluate_slack_true(ground_truth_X)
            ground_truth_feas_mask = (ground_truth_con <= 0).all(dim=-1)

        # Plot feasible ground truth
        ground_truth_feas_f1 = ground_truth_obj[ground_truth_feas_mask, 0].detach().cpu().numpy()
        ground_truth_feas_f2 = ground_truth_obj[ground_truth_feas_mask, 1].detach().cpu().numpy()
        axes.scatter(ground_truth_feas_f1, ground_truth_feas_f2, **ground_truth_feas_kwargs)

        # Plot infeasible points - Could be enabled
        ground_truth_inf_mask = torch.logical_not(ground_truth_feas_mask)
        ground_truth_inf_f1 = ground_truth_obj[ground_truth_inf_mask, 0].detach().cpu().numpy()
        ground_truth_inf_f2 = ground_truth_obj[ground_truth_inf_mask, 1].detach().cpu().numpy()
        axes.scatter(ground_truth_inf_f1, ground_truth_inf_f2, **ground_truth_inf_kwargs)

    """ Plot posterior pareto """
    if show_posterior:
        raise NotImplementedError("Function not yet implemented.")
        # # Predict over test grid
        # ground_truth_X = ground_truth_X.to(mobo.get_device(), mobo.get_dtype())
        # posterior = mobo.get_model().posterior(ground_truth_X)
        # mean = posterior.mean.detach().cpu().numpy()
        # std = posterior.variance.sqrt().detach().cpu().numpy()
        #
        # # Plot mean with error bars
        # axes.errorbar(
        #     mean[:, 0],
        #     mean[:, 1],
        #     xerr=3 * std[:, 0],
        #     yerr=3 * std[:, 1],
        #     **posterior_pareto_kwargs,
        # )

    """ Plot reference point """
    if show_ref_point is True:
        ref_point = mobo.get_objective().ref_point.detach().cpu().numpy()
        plt.scatter(ref_point[0], ref_point[1], **ref_point_kwargs)

    """ Plot observations """
    if show_observations is True:
        y_obj = mobo.get_Yobj().clone()

        # Compute masks
        obs_feas_mask = mobo.get_feasible_observations_mask()
        obs_par_mask = mobo.get_pareto_front_mask()
        obs_inf_mask = torch.logical_not(obs_feas_mask)
        obs_feas_and_par_mask = torch.logical_and(obs_feas_mask, obs_par_mask)
        obs_feas_and_not_par_mask = torch.logical_and(obs_feas_mask, torch.logical_not(obs_par_mask))

        # Plot infeasible observations
        if mobo.get_Ycon() is not None and torch.any(obs_inf_mask):
            obs_inf = y_obj[obs_inf_mask].detach().cpu().numpy()
            axes.scatter(obs_inf[:, 0], obs_inf[:, 1], **infeasible_observations_kwargs)

        # Plot feasible non-pareto-front observations
        if torch.any(obs_feas_and_not_par_mask):
            obs_feas_non_par = y_obj[obs_feas_and_not_par_mask].detach().cpu().numpy()
            axes.scatter(obs_feas_non_par[:, 0], obs_feas_non_par[:, 1], **feasible_non_pareto_observations_kwargs)

        # Plot feasible pareto-front observations
        if torch.any(obs_feas_and_par_mask):
            obs_feas_par = y_obj[obs_feas_and_par_mask].detach().cpu().numpy()
            axes.scatter(obs_feas_par[:, 0], obs_feas_par[:, 1], **feasible_pareto_observations_kwargs)

    axes.autoscale_view(tight=False)  # Update the view limits if limits are not set
    # Add legend
    plt.legend()
    if output_path is None:
        output_path = Path.cwd() / "plot_multi_objective_from_RN_to_R2.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figures:
        plt.show()
    plt.close(fig)


def plot_multi_objective_from_RN_to_R3_with_color_coded_R3(
        mobo,
        X: torch.Tensor or None = None,
        title: str | None = None,
        f_labels: list[str] = ("$f_{01}$", "$f_{02}$", "$f_{03}$"),
        f_lims: list[tuple] = None,
        x_idx: int = 0,
        y_idx: int = 1,
        z_idx: int = 2,
        show_ref_point: bool = True,
        show_observations: bool = True,
        show_ground_truth: bool = False,
        show_posterior: bool = False,
        display_figure: bool = True,
        output_path=None,
):
    """
    General-purpose plotter for 2D and 3D objectives with optional face-color color coding.
    x_inx is the index of the objective plotted on the x-axis.
    y_idx is the index of the objective plotted on the y-axis.
    z_idx is the index of the objective used for color coding.    """

    # Check indexes consistency
    assert x_idx != y_idx, "x_idx and y_idx must be different"
    if z_idx is not None:
        assert z_idx not in (x_idx, y_idx), "color_idx must be different from x_idx and y_idx"

    # Initialize plot. If the objective limits are not passed, set the axes to autoscale, otherwise
    # fix the x- and y-limits. If all labels are not passed explicitly, set them to "$f_1$", "$f_2$"
    # and "$f_3$".
    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    axes.set_title(title or 'Pareto Front')
    if f_labels is None or len(f_labels) != 3:
        f_labels = [f"$f_{{{i}}}$" for i in range(3)]
    axes.set_xlabel(f_labels[x_idx])
    axes.set_ylabel(f_labels[y_idx])
    if f_lims is None:
        axes.autoscale(enable=True, axis='both')
    else:
        axes.set_xlim(*f_lims[x_idx])
        axes.set_ylim(*f_lims[y_idx])

    # Define colormap. If the argument "x" is provided and the objective includes an "evaluate_true"
    # function, then the ground truth is calculated as "evaluate_true(x)" and the colormap normalized
    # between the minimum and maximum value of the "color_values = evaluate_true(x)" and the colormap
    # normalized between the minimum and maximum values of "color_value". Otherwise, the colormap is
    # normalized between the minimum and maximum values of the objective used for color-coding.
    if hasattr(mobo.get_objective(), "evaluate_true") and X is not None:
        X = X.to(mobo.get_device(), mobo.get_dtype())
        ground_truth = mobo.get_objective().evaluate_true(X)
        color_values = ground_truth[:, z_idx].cpu().numpy()
    else:
        color_values = mobo.get_Yobj()[:, z_idx].cpu().numpy()
    norm = mpl.colors.Normalize(vmin=color_values.min(), vmax=color_values.max())
    cmap = mpl.cm.coolwarm

    """ Plot ground truth """
    if show_ground_truth:
        # Check that the objective includes an "evaluate_true" method and that the argument X is not note.
        if not hasattr(mobo.get_objective(), "evaluate_true"):
            raise ValueError("Ground truth not available: the objective must include an evaluate_true function.")
        if X is None:
            raise ValueError("Ground truth not available: X must be provided.")

        # Compute the ground truth. First, calculate the ground truth without considering the constraints
        # on the objectives, then compute the feasibility mask to select the values that satisfy the constraints.
        X = X.to(mobo.get_device(), mobo.get_dtype())
        ground_truth = mobo.get_objective().evaluate_true(X)
        if mobo.get_output_constraints() is None:
            ground_truth_feas_mask = torch.ones_like(ground_truth, dtype=torch.bool).all(dim=-1)
        else:
            ground_truth_con = mobo.get_objective().evaluate_slack_true(X)
            ground_truth_feas_mask = (ground_truth_con <= 0).all(dim=-1)
        ground_truth_inf_mask = torch.logical_not(ground_truth_feas_mask)

        # Plot feasible ground truth
        ground_truth_feas_vals = ground_truth[ground_truth_feas_mask]
        if ground_truth_feas_vals.numel() > 0:
            feas_X = ground_truth_feas_vals[:, x_idx].detach().cpu().numpy()
            feas_Y = ground_truth_feas_vals[:, y_idx].detach().cpu().numpy()
            color_values = ground_truth_feas_vals[:, z_idx].detach().cpu().numpy()
            kwargs = dict(ground_truth_feas_kwargs)  # make a local copy
            kwargs.pop("color", None)
            c = cmap(norm(color_values))
            axes.scatter(x=feas_X, y=feas_Y, c=c, **kwargs)

        # Plot infeasible ground truth
        ground_truth_inf_vals = ground_truth[ground_truth_inf_mask]
        if ground_truth_inf_vals.numel() > 0:
            inf_X = ground_truth_inf_vals[:, x_idx].detach().cpu().numpy()
            inf_Y = ground_truth_inf_vals[:, y_idx].detach().cpu().numpy()
            color_values = ground_truth_inf_vals[:, z_idx].detach().cpu().numpy()
            kwargs = dict(ground_truth_inf_kwargs)  # make a local copy
            kwargs.pop("color", None)
            c = cmap(norm(color_values))
            axes.scatter(x=inf_X, y=inf_Y, c=c, **kwargs)

    """ Plot posterior (not implemented) """
    if show_posterior:
        raise NotImplementedError("Posterior plotting is not yet implemented.")

    """ Plot reference point """
    if show_ref_point:
        # Plot the reference point, color-coded.
        ref_point = mobo.get_objective().ref_point.detach().cpu().numpy()
        color = ref_point[z_idx]
        kwargs = ref_point_kwargs
        kwargs.pop("color", None)
        c = cmap(norm(color))
        axes.scatter(x=ref_point[x_idx], y=ref_point[y_idx], color=c, **ref_point_kwargs)

    """ Plot observations """
    if show_observations:
        y_obj = mobo.get_Yobj().clone()

        # Compute masks
        ground_truth_feas_mask = mobo.get_feasible_observations_mask()
        par_mask = mobo.get_pareto_front_mask()
        inf_mask = torch.logical_not(ground_truth_feas_mask)
        feas_non_par_mask = torch.logical_and(ground_truth_feas_mask, torch.logical_not(par_mask))
        feas_par_mask = torch.logical_and(ground_truth_feas_mask, par_mask)

        # Plot infeasible observations
        if mobo.get_Ycon() is not None and torch.any(inf_mask):
            obs_inf = y_obj[inf_mask].detach().cpu().numpy()
            color_values = obs_inf[:, z_idx]
            c = cmap(norm(color_values))
            kwargs = dict(infeasible_observations_kwargs)  # make a local copy
            kwargs.pop("color", None)
            axes.scatter(
                obs_inf[:, x_idx],
                obs_inf[:, y_idx],
                c=c,
                **kwargs
            )

        # Plot feasible non-pareto-front observations
        if torch.any(feas_non_par_mask):
            obs_feas_non_par = y_obj[feas_non_par_mask].detach().cpu().numpy()
            color_values = obs_feas_non_par[:, z_idx]
            c = cmap(norm(color_values))
            kwargs = dict(feasible_non_pareto_observations_kwargs)  # make a local copy
            kwargs.pop("color", None)
            axes.scatter(x=obs_feas_non_par[:, x_idx], y=obs_feas_non_par[:, y_idx], c=c, **kwargs)

        # Plot feasible pareto-front observations
        if torch.any(feas_par_mask):
            obs_feas_par = y_obj[feas_par_mask].detach().cpu().numpy()
            color_values = obs_feas_par[:, z_idx]
            c = cmap(norm(color_values))
            kwargs = dict(feasible_pareto_observations_kwargs)  # make a local copy
            kwargs.pop("color", None)
            axes.scatter(x=obs_feas_par[:, x_idx], y=obs_feas_par[:, y_idx], c=c, **kwargs)

    """ Colorbar """
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, pad=0.01)
    cbar.set_label(f_labels[z_idx])

    axes.autoscale_view(tight=False)    # Compute flags for what was actually plotted
    legend_elements = build_custom_legend_elements(
        show_ref_point=show_ref_point,
        show_observations=show_observations,
        feas_non_par=torch.any(feas_non_par_mask).item() if show_observations else False,
        feas_par=torch.any(feas_par_mask).item() if show_observations else False,
        infeas=torch.any(inf_mask).item() if show_observations and mobo.get_Ycon() is not None else False,
        show_ground_truth=show_ground_truth,
        gt_feas=torch.any(ground_truth_feas_mask).item() if show_ground_truth else False,
        gt_infeas=torch.any(torch.logical_not(ground_truth_feas_mask)).item() if show_ground_truth else False,
    )
    axes.legend(handles=legend_elements, loc='best')

    if output_path is None:
        output_path = Path.cwd() / "plot_multi_objective_general.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figure:
        plt.show()
    plt.close(fig)


def plot_multi_objective_from_RN_to_R3(
        mobo: Mobo,
        x: torch.Tensor,
        f_lims=None,
        f_labels=("$f_{01}$", "$f_{02}$", "$f_{03}$"),
        show_ref_point=True,
        show_observations=True,
        display_figure=True,
        show_ground_truth=False,
        show_posterior=False,
        output_path=None,
):
    fig = plt.figure(figsize=(8, 6))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_title(r'Multi-objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^3$')

    # Set axis labels
    axes.set_xlabel(f_labels[0])
    axes.set_ylabel(f_labels[1])
    axes.set_zlabel(f_labels[2])

    # Set limits if provided
    if f_lims is not None and isinstance(f_lims, tuple) and len(f_lims) == 3:
        axes.set_xlim(f_lims[0])
        axes.set_ylim(f_lims[1])
        axes.set_zlim(f_lims[2])

    """ Plot ground truth """
    if show_ground_truth:
        if mobo.get_true_objective() is None:
            raise ValueError("Ground truth not available.")

        x = x.to(mobo.get_device(), mobo.get_dtype())
        ground_truth_obj = mobo.get_true_objective()(x)

        if mobo.get_output_constraints() is None:
            feas_mask = torch.ones_like(ground_truth_obj, dtype=torch.bool).all(dim=-1)
        else:
            con = -mobo.get_true_objective().evaluate_slack(x)
            feas_mask = (con <= 0).all(dim=-1)

        feas = ground_truth_obj[feas_mask].detach().cpu().numpy()
        axes.scatter(feas[:, 0], feas[:, 1], feas[:, 2], **ground_truth_feas_kwargs)

    """ Plot posterior """
    if show_posterior:
        x = x.to(mobo.get_device(), mobo.get_dtype())
        posterior = mobo.get_model().posterior(x)
        mean = posterior.mean.detach().cpu().numpy()
        std = posterior.variance.sqrt().detach().cpu().numpy()

        axes.errorbar(
            mean[:, 0], mean[:, 1], mean[:, 2],
            xerr=3 * std[:, 0],
            yerr=3 * std[:, 1],
            zerr=3 * std[:, 2],
            fmt='o', **posterior_pareto_kwargs,
        )

    """ Plot reference point """
    if show_ref_point is True:
        ref_point = mobo.get_objective().ref_point.detach().cpu().numpy()
        axes.scatter(ref_point[0], ref_point[1], ref_point[2], **ref_point_kwargs)

    """ Plot observations """
    if show_observations:
        y_obj = mobo.get_Yobj()

        # Compute masks
        obs_feas_mask = mobo.get_feasible_observations_mask()
        obs_par_mask = mobo.get_pareto_front_mask()
        obs_inf_mask = torch.logical_not(mobo.get_feasible_observations_mask())
        obs_feas_and_par_mask = torch.logical_and(obs_feas_mask, obs_par_mask)
        obs_feas_and_not_par_mask = torch.logical_and(obs_feas_mask, torch.logical_not(obs_par_mask))

        # Plot Pareto surface
        pareto_points = y_obj[obs_feas_and_par_mask].detach().cpu().numpy()
        if pareto_points.shape[0] >= 3:
            # Reduce 3D points to 2D parameters
            pca = PCA(n_components=2)
            params_2d = pca.fit_transform(pareto_points)  # shape (num_points, 2)
            # Triangulate in param space
            tri = Triangulation(params_2d[:, 0], params_2d[:, 1])
            # Plot trisurf in 3D using original coordinates and the 2D triangulation
            axes.plot_trisurf(
                pareto_points[:, 0],
                pareto_points[:, 1],
                pareto_points[:, 2],
                triangles=tri.triangles,
                cmap='Blues',
                alpha=0.4,
                edgecolor='gray'
            )

        # Plot infeasible observations
        if mobo.get_Ycon() is not None and torch.any(obs_inf_mask):
            obs_inf = y_obj[obs_inf_mask].detach().cpu().numpy()
            axes.scatter(obs_inf[:, 0], obs_inf[:, 1], obs_inf[:, 2], **infeasible_observations_kwargs)

        # Plot feasible non-pareto-front observations
        if torch.any(obs_feas_and_not_par_mask):
            obs = y_obj[obs_feas_and_not_par_mask].detach().cpu().numpy()
            axes.scatter(obs[:, 0], obs[:, 1], obs[:, 2], **feasible_non_pareto_observations_kwargs)

        # Plot feasible pareto-front observations
        if torch.any(obs_feas_and_par_mask):
            obs = y_obj[obs_feas_and_par_mask].detach().cpu().numpy()
            axes.scatter(obs[:, 0], obs[:, 1], obs[:, 2], **feasible_pareto_observations_kwargs)

    # Add legend and save
    plt.legend()
    if output_path is None:
        output_path = Path.cwd() / "plot_multi_objective_from_RN_to_R3.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figure:
        plt.show()
    plt.close(fig)


def plot_log_hypervolume_improvement(
        mobo: Mobo,
        output_path: Path = None,
        batch_size: int = 1,
        display_figure: bool = False,
        epsilon: float = 1e-6,
):
    """
    Plots two scatter plots:
    1. Cumulative hypervolume over iterations.
    2. log10 of relative hypervolume improvement.

    Args:
        hv (list[float]): List of cumulative hypervolume values.
        output_path (Path, optional): Where to save the figure.
        batch_size (int): Number of new samples per BO iteration.
        display_figure (bool): If True, displays plot interactively.
        epsilon (float): Minimum improvement to avoid log10(0).
        color (str): Color used in scatter plots.
    """

    hv = mobo.get_hypervolume()

    if len(hv) <= 1:
        print("Not enough data to plot.")
        return

    hv = np.array(hv)
    hvi = np.diff(hv, prepend=np.nan)  # first element has no diff

    # Clip small values to avoid log10(0)
    hvi_log = np.log10(np.clip(hvi, epsilon, None))

    # X-axis: number of observations beyond initial front
    x = np.arange(len(hv)) * batch_size

    if output_path is None:
        output_path = Path.cwd() / "hv_and_log_hvi.png"

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Subplot 1: Cumulative HV ---
    axs[0].scatter(x, hv, **xy_plot_kwargs)
    axs[0].set_title("Hypervolume")
    axs[0].set_ylabel("HV")
    axs[0].grid(True)
    # axs[0].legend()
    if hasattr(mobo.get_objective(), "max_hv") and mobo.get_objective().max_hv is not None:
        axs[0].axhline(y=mobo.get_objective().max_hv, linestyle='--', color='black', label='Max HV')

    # --- Subplot 2: log10(HVI) ---
    mask = ~np.isnan(hvi_log)
    axs[1].scatter(x[mask], hvi_log[mask], **xy_plot_kwargs)
    axs[1].set_title("Hypervolume Improvement")
    axs[1].set_xlabel("Number of observations")
    axs[1].set_ylabel(r"$\log_{10}(\mathrm{HVI})$")
    axs[1].grid(True)
    # axs[1].legend()

    # Save or show
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format="png")

    if display_figure:
        plt.show()
    plt.close(fig)


def plot_elapsed_time(
        mobo: Mobo,
        output_path: Path or None = None,
        batch_size: int = 1,  # Number of new X drawn per iteration
        show=False):

    elapsed_time = mobo.get_elapsed_time()

    if output_path is None:
        output_path = Path.cwd() / "plot_elapsed_time.png"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("Number of observations (beyond initial points)")
    ax.set_ylabel("Elapsed Time (s)")
    x = np.array(range(len(elapsed_time))) * batch_size
    y = elapsed_time
    ax.scatter(x, y, **xy_plot_kwargs)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)

