import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from matplotlib.tri import Triangulation
import matplotlib as mpl
from mobo.mobo import Mobo
from matplotlib.lines import Line2D
from botorch.utils.multi_objective import is_non_dominated

ms = 7

feasible_pareto_observations_kwargs = {
    'marker': 'D',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolors': 'black',
    'alpha': 0.7,
    'label': 'Pareto Obs.'
}

feasible_non_pareto_observations_kwargs = {
    'color': "tab:green",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Non-Pareto Obs.'
}

infeasible_observations_kwargs = {
    'color': "tab:red",
    'marker': "x",
    's': ms ** 2,
    "alpha": 0.7,
    # "edgecolors": "black",
    'label': 'Inf. Obs.'
}

ref_point_kwargs = {
    'color': 'tab:red',
    "edgecolors": "black",
    'marker': 's',
    's': ms ** 2,
    'alpha': 0.7,
    'label': 'Ref. Point'
}

feasible_pareto_ground_truth_kwargs = {
    'color': "black",
    'marker': "D",
    's': ms ** 2 / 5,
    "alpha": 1,
    'label': 'Pareto GT'
}

feasible_non_pareto_ground_truth_kwargs = {
    'color': "black",
    'marker': "o",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Non-Pareto GT'
}

infeasible_ground_truth_kwargs = {
    'color': "red",
    'marker': "x",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Inf. GT'
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


def plot_objective_from_RN_to_R1():
    raise NotImplementedError("Function not yet implemented.")


def plot_multi_objective_from_RN_to_R2(
        mobo: Mobo,
        X: torch.Tensor or None = None,
        title: str or None = None,
        f1_label: str or None = None,
        f2_label: str or None = None,
        f1_lims=None,
        f2_lims=None,
        show_ref_point=True,
        show_ground_truth=False,
        show_posterior=False,
        show_observations=True,
        display_figure=True,
        output_path=None,
):
    # Initialize the figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.set_title(title or "Pareto Front")
    axes.set_xlabel("$f_{1}$") if not f1_label else axes.set_xlabel(f1_label)
    axes.set_ylabel("$f_{2}$") if not f2_label else axes.set_ylabel(f2_label)
    axes.set_xlim(f1_lims[0], f1_lims[1]) if f1_lims is not None else axes.autoscale(enable=True, axis='x')
    axes.set_ylim(f2_lims[0], f2_lims[1]) if f2_lims is not None else axes.autoscale(enable=True, axis='y')
    if output_path is None:
        output_path = Path.cwd() / "pareto_front.png"

    """ Plot ground truth """
    if show_ground_truth:

        if mobo.objective.evaluate_true_objective(X) is None:
            raise ValueError("Cannot plot ground truth: ground truth is not available.")
        Y_gt = mobo.objective.evaluate_true_objective(X)

        # === Compute feasible and infeasible masks ===
        if mobo.objective.output_constraints is None:
            feasible_mask = torch.ones(Y_gt.shape[-2], dtype=torch.bool)
            infeasible_mask = torch.zeros_like(Y_gt, dtype=torch.bool)
        else:
            ground_truth_con = mobo.objective.evaluate_true_slack(X)
            feasible_mask = (ground_truth_con <= 0).all(dim=-1)
            infeasible_mask = torch.logical_not(feasible_mask)

        # === Compute pareto masks ===
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)
        if feasible_mask.any():
            Y_par = Y_gt.clone()
            Y_par[..., mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[feasible_mask]
            pareto_mask[feasible_mask] = is_non_dominated(Y_par)
        feasible_pareto_mask = torch.logical_and(feasible_mask, pareto_mask)
        feasible_non_pareto_mask = torch.logical_and(feasible_mask, torch.logical_not(pareto_mask))

        # === Plot infeasible ground truth ===
        if mobo.objective.output_constraints is not None:
            if torch.any(infeasible_mask):
                Y_gt_inf = Y_gt[infeasible_mask].detach().cpu().numpy()
                axes.scatter(Y_gt_inf[:, 0], Y_gt_inf[:, 1], **infeasible_ground_truth_kwargs)

        # === Plot feasible non-pareto-front ground truth ===
        if torch.any(feasible_non_pareto_mask):
            Y_gt_feas_non_par = Y_gt[feasible_non_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_gt_feas_non_par[:, 0], Y_gt_feas_non_par[:, 1], **feasible_non_pareto_ground_truth_kwargs)

        # === Plot feasible pareto-front ground truth ===
        if torch.any(feasible_mask):
            Y_gt_feas_par = Y_gt[feasible_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_gt_feas_par[:, 0], Y_gt_feas_par[:, 1], **feasible_pareto_ground_truth_kwargs)

    """ Plot posterior pareto """
    if show_posterior:
        raise NotImplementedError("Function not yet implemented.")
        # # Predict over test grid
        # X = X.to(mobo.get_device(), mobo.get_dtype())
        # posterior = mobo.get_model().posterior(X)
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
        ref_point = mobo.objective.ref_point.detach().cpu().numpy()
        axes.scatter(ref_point[0], ref_point[1], **ref_point_kwargs)

    """ Plot observations """
    if show_observations is True:

        Y_obj = mobo.Y_obj.clone()

        # === Compute feasible and infeasible masks ===
        if mobo.objective.output_constraints is None:
            feasible_mask = torch.ones(mobo.Y_obj.shape[-2], dtype=torch.bool)
            infeasible_mask = torch.zeros_like(mobo.Y_obj, dtype=torch.bool)
        else:
            Y_full = torch.cat([mobo.Y_obj, mobo.Y_con], dim=-1)
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in mobo.objective.output_constraints]).all(dim=-2)
            infeasible_mask = torch.logical_not(feasible_mask)

        # === Compute pareto masks ===
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)
        if feasible_mask.any():
            Y_par = Y_obj.clone()
            Y_par[..., mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[feasible_mask]
            pareto_mask[feasible_mask] = is_non_dominated(Y_par)
        feasible_pareto_mask = torch.logical_and(feasible_mask, pareto_mask)
        feasible_non_pareto_mask = torch.logical_and(feasible_mask, torch.logical_not(pareto_mask))

        # === Plot infeasible observations ===
        if mobo.objective.output_constraints is not None:
            if torch.any(infeasible_mask):
                Y_obj_inf = Y_obj[infeasible_mask].detach().cpu().numpy()
                axes.scatter(Y_obj_inf[:, 0], Y_obj_inf[:, 1], **infeasible_observations_kwargs)

        # === Plot feasible non-pareto-front observations ===
        if torch.any(feasible_non_pareto_mask):
            Y_obj_feas_non_par = Y_obj[feasible_non_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_non_par[:, 0], Y_obj_feas_non_par[:, 1], **feasible_non_pareto_observations_kwargs)

        # === Plot feasible pareto-front observations ===
        if torch.any(feasible_mask):
            Y_obj_feas_par = Y_obj[feasible_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_par[:, 0], Y_obj_feas_par[:, 1], **feasible_pareto_observations_kwargs)

    axes.autoscale_view(tight=False)  # Update the view limits if limits are not set
    plt.legend()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figure:
        plt.show()
    plt.close(fig)


def plot_multi_objective_from_RN_to_R3_with_color_coded_R3(
        mobo,
        X: torch.Tensor | None = None,
        title: str | None = None,
        f1_label: str = "$f_{1}",
        f2_label: str = "$f_{2}$",
        f3_label: str = "$f_{3}$",
        f1_lims: tuple[float, float] = None,
        f2_lims: tuple[float, float] = None,
        f3_lims: tuple[float, float] = None,
        f1_idx: int = 0,
        f2_idx: int = 1,
        f3_idx: int = 2,
        show_ref_point: bool = True,
        show_ground_truth: bool = False,
        show_posterior: bool = False,
        show_observations: bool = True,
        display_figure: bool = True,
        output_path=None,
):
    """
    General-purpose plotter for 3D objectives with optional face-color color coding.
    f1_idx is the index of the objective plotted on the x-axis.
    f2_idx is the index of the objective plotted on the y-axis.
    f3_idx is the index of the objective used for color coding.    """

    # Check indexes consistency
    assert f1_idx != f2_idx, "f1_idx and f2_idx must be different"
    if f3_idx is not None:
        assert f3_idx not in (f1_idx, f2_idx), "color_idx must be different from f1_idx and f2_idx"

    # Initialize the figure
    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    axes.set_title(title or 'Pareto Front')
    axes.set_xlabel("$f_{1}$") if not f1_label else axes.set_xlabel(f1_label)
    axes.set_ylabel("$f_{2}$") if not f2_label else axes.set_ylabel(f2_label)
    axes.set_xlim(f1_lims[0], f1_lims[1]) if f1_lims is not None else axes.autoscale(enable=True, axis='x')
    axes.set_ylim(f2_lims[0], f2_lims[1]) if f2_lims is not None else axes.autoscale(enable=True, axis='y')
    legend_elements = []
    if output_path is None:
        output_path = Path.cwd() / "pareto_front.png"

    # If the limits for the third observable are given, use them to define the colormap
    # norm. Otherwise, collect the ground truth and the observations, and find the
    # overall min and max.
    if f3_lims is not None:
        norm = mpl.colors.Normalize(vmin=f3_lims[0], vmax=f3_lims[1])
    else:
        Y_colors = None
        if X is not None:
            if callable(mobo.objective.evaluate_true_objective):
                X = X.to(mobo.device, mobo.dtype)
                Y_colors = mobo.objective.evaluate_true_objective(X)
        if mobo.Y_obj is not None:
            if Y_colors is not None:
                Y_colors = torch.cat([Y_colors, mobo.Y_obj], dim=0)
            else:
                Y_colors = mobo.Y_obj
        if Y_colors is not None:
            Y_colors = Y_colors.detach().cpu().numpy()[:, f3_idx]
            norm = mpl.colors.Normalize(vmin=Y_colors.min().item(), vmax=Y_colors.max().item())
        else:
            raise ValueError("Cannot compute color coding.")
    cmap = mpl.cm.coolwarm

    """ Plot ground truth """
    if show_ground_truth:
        Y_gt = mobo.objective.evaluate_true_objective(X)

        # === Compute feasible and infeasible masks ===
        if mobo.objective.output_constraints is None:
            feasible_mask = torch.ones(Y_gt.shape[-2], dtype=torch.bool)
            infeasible_mask = torch.zeros_like(Y_gt, dtype=torch.bool)
        else:
            ground_truth_con = mobo.objective.evaluate_true_slack(X)
            feasible_mask = (ground_truth_con <= 0).all(dim=-1)
            infeasible_mask = torch.logical_not(feasible_mask)

        # === Compute pareto masks ===
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)
        if feasible_mask.any():
            Y_par = Y_gt.clone()
            Y_par[..., mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[feasible_mask]
            pareto_mask[feasible_mask] = is_non_dominated(Y_par)
        feasible_pareto_mask = torch.logical_and(feasible_mask, pareto_mask)
        feasible_non_pareto_mask = torch.logical_and(feasible_mask, torch.logical_not(pareto_mask))

        # === Plot infeasible ground truth ===
        if mobo.objective.output_constraints is not None:
            if torch.any(infeasible_mask):
                Y_gt_inf = Y_gt[infeasible_mask].detach().cpu().numpy()
                color_values = Y_gt_inf[:, f3_idx]
                kwargs = dict(infeasible_ground_truth_kwargs)  # make a local copy
                kwargs.pop("color", None)
                c = cmap(norm(color_values))
                axes.scatter(x=Y_gt_inf[:, f1_idx], y=Y_gt_inf[:, f2_idx], c=c, **kwargs)
                legend_elements.append(Line2D(
                    [0], [0], marker=infeasible_ground_truth_kwargs['marker'],
                    label=infeasible_ground_truth_kwargs['label'],
                    color=infeasible_ground_truth_kwargs['color'],
                    markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

        # === Plot feasible non-pareto-front ground truth ===
        if torch.any(feasible_non_pareto_mask):
            Y_gt_feas_non_par = Y_gt[feasible_non_pareto_mask].detach().cpu().numpy()
            color_values = Y_gt_feas_non_par[:, f3_idx]
            kwargs = dict(feasible_non_pareto_ground_truth_kwargs)  # make a local copy
            kwargs.pop("color", None)
            c = cmap(norm(color_values))
            axes.scatter(x=Y_gt_feas_non_par[:, 0], y=Y_gt_feas_non_par[:, 1], c=c, **kwargs)
            legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_ground_truth_kwargs['marker'],
                label=feasible_non_pareto_ground_truth_kwargs['label'],
                color=feasible_non_pareto_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

        # === Plot feasible pareto-front ground truth ===
        if torch.any(feasible_pareto_mask):
            Y_gt_feas_par = Y_gt[feasible_pareto_mask].detach().cpu().numpy()
            color_values = Y_gt_feas_par[:, f3_idx]
            kwargs = dict(feasible_non_pareto_ground_truth_kwargs)  # make a local copy
            kwargs.pop("color", None)
            c = cmap(norm(color_values))
            axes.scatter(x=Y_gt_feas_par[:, 0], y=Y_gt_feas_par[:, 1], c=c, **kwargs)
            legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_ground_truth_kwargs['marker'],
                label=feasible_pareto_ground_truth_kwargs['label'],
                color=feasible_pareto_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

    """ Plot posterior (not implemented) """
    if show_posterior:
        raise NotImplementedError("Posterior plotting is not yet implemented.")

    """ Plot reference point """
    if show_ref_point:
        # Plot the reference point, color-coded.
        ref_point = mobo.objective.ref_point.detach().cpu().numpy()
        color = ref_point[f3_idx]
        kwargs = ref_point_kwargs
        kwargs.pop("color", None)
        c = cmap(norm(color))
        axes.scatter(x=ref_point[f1_idx], y=ref_point[f2_idx], color=c, **ref_point_kwargs)
        legend_elements.append(Line2D(
            [0], [0], marker=ref_point_kwargs['marker'],
            label=ref_point_kwargs['label'],
            color=ref_point_kwargs['edgecolors'],
            markerfacecolor='none', markersize=ms, linestyle='None'))

    """ Plot observations """
    if show_observations:

        Y_obj = mobo.Y_obj.clone()

        # === Compute feasible and infeasible masks ===
        if mobo.objective.output_constraints is None:
            feasible_mask = torch.ones(mobo.Y_obj.shape[-2], dtype=torch.bool)
            infeasible_mask = torch.zeros_like(mobo.Y_obj, dtype=torch.bool)
        else:
            Y_full = torch.cat([mobo.Y_obj, mobo.Y_con], dim=-1)
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in mobo.objective.output_constraints]).all(dim=-2)
            infeasible_mask = torch.logical_not(feasible_mask)

        # === Compute pareto masks ===
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)
        if feasible_mask.any():
            Y_par = Y_obj.clone()
            Y_par[..., mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[feasible_mask]
            pareto_mask[feasible_mask] = is_non_dominated(Y_par)
        feasible_pareto_mask = torch.logical_and(feasible_mask, pareto_mask)
        feasible_non_pareto_mask = torch.logical_and(feasible_mask, torch.logical_not(pareto_mask))

        # === Plot infeasible observations ===
        if mobo.objective.output_constraints is not None:
            if torch.any(infeasible_mask):
                Y_obj_inf = Y_obj[infeasible_mask].detach().cpu().numpy()
                color_values = Y_obj_inf[:, f3_idx]
                c = cmap(norm(color_values))
                kwargs = dict(infeasible_observations_kwargs)  # make a local copy
                kwargs.pop("color", None)
                axes.scatter(x=Y_obj_inf[:, f1_idx], y=Y_obj_inf[:, f2_idx], c=c, **kwargs)
                legend_elements.append(Line2D(
                    [0], [0], marker=infeasible_observations_kwargs['marker'],
                    label=infeasible_observations_kwargs['label'],
                    color=infeasible_observations_kwargs['edgecolors'],
                    markerfacecolor='none', markersize=ms, linestyle='None'))

        # === Plot feasible non-pareto-front observations ===
        if torch.any(feasible_non_pareto_mask):
            Y_obj_feas_non_par = Y_obj[feasible_non_pareto_mask].detach().cpu().numpy()
            color_values = Y_obj_feas_non_par[:, f3_idx]
            c = cmap(norm(color_values))
            kwargs = dict(feasible_non_pareto_observations_kwargs)  # make a local copy
            kwargs.pop("color", None)
            axes.scatter(x=Y_obj_feas_non_par[:, f1_idx], y=Y_obj_feas_non_par[:, f2_idx], c=c, **kwargs)
            legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_observations_kwargs['marker'],
                label=feasible_non_pareto_observations_kwargs['label'],
                color=feasible_non_pareto_observations_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

        # === Plot feasible pareto-front observations ===
        if torch.any(feasible_pareto_mask):
            Y_obj_feas_par = Y_obj[feasible_pareto_mask].detach().cpu().numpy()
            color_values = Y_obj_feas_par[:, f3_idx]
            c = cmap(norm(color_values))
            kwargs = dict(feasible_pareto_observations_kwargs)  # make a local copy
            kwargs.pop("color", None)
            axes.scatter(x=Y_obj_feas_par[:, f1_idx], y=Y_obj_feas_par[:, f2_idx], c=c, **kwargs)
            legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_observations_kwargs['marker'],
                label=feasible_pareto_observations_kwargs['label'],
                color=feasible_pareto_observations_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

    """ Add Colorbar """
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, pad=0.01)
    cbar.set_label(f3_label)

    """ Add Legend """
    axes.legend(handles=legend_elements, loc='best')

    axes.autoscale_view(tight=False)  # Compute flags for what was actually plotted
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figure:
        plt.show()
    plt.close(fig)


def plot_multi_objective_from_RN_to_R3(
        mobo: Mobo,
        X: torch.Tensor,
        title: str | None = None,
        f1_label: str = "$f_{1}",
        f2_label: str = "$f_{2}$",
        f3_label: str = "$f_{3}$",
        f1_lims: tuple[float, float] = None,
        f2_lims: tuple[float, float] = None,
        f3_lims: tuple[float, float] = None,
        show_ref_point: bool = True,
        show_ground_truth: bool = False,
        show_posterior: bool = False,
        show_observations: bool = True,
        display_figure: bool = True,
        output_path=None,
):
    # Initialize the figure
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    axes.set_title(title or 'Pareto Front')
    axes.set_xlabel("$f_{1}$") if not f1_label else axes.set_xlabel(f1_label)
    axes.set_ylabel("$f_{2}$") if not f2_label else axes.set_ylabel(f2_label)
    axes.set_zlabel("$f_{3}$") if not f3_label else axes.set_ylabel(f3_label)
    axes.set_xlim(f1_lims[0], f1_lims[1]) if f1_lims is not None else axes.autoscale(enable=True, axis='x')
    axes.set_ylim(f2_lims[0], f2_lims[1]) if f2_lims is not None else axes.autoscale(enable=True, axis='y')
    axes.set_zlim(f3_lims[0], f3_lims[1]) if f3_lims is not None else axes.autoscale(enable=True, axis='z')
    if output_path is None:
        output_path = Path.cwd() / "pareto_front_3d.png"

    """ Plot ground truth """
    if show_ground_truth:

        if mobo.objective.evaluate_true_objective(X) is None:
            raise ValueError("Cannot plot ground truth: ground truth is not available.")
        Y_gt = mobo.objective.evaluate_true_objective(X)

        # === Compute feasible and infeasible masks ===
        if mobo.objective.output_constraints is None:
            feasible_mask = torch.ones(Y_gt.shape[-2], dtype=torch.bool)
            infeasible_mask = torch.zeros_like(Y_gt, dtype=torch.bool)
        else:
            ground_truth_con = mobo.objective.evaluate_true_slack(X)
            feasible_mask = (ground_truth_con <= 0).all(dim=-1)
            infeasible_mask = torch.logical_not(feasible_mask)

        # === Compute pareto masks ===
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)
        if feasible_mask.any():
            Y_par = Y_gt.clone()
            Y_par[..., mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[feasible_mask]
            pareto_mask[feasible_mask] = is_non_dominated(Y_par)
        feasible_pareto_mask = torch.logical_and(feasible_mask, pareto_mask)
        feasible_non_pareto_mask = torch.logical_and(feasible_mask, torch.logical_not(pareto_mask))

        # === Plot infeasible ground truth ===
        if mobo.objective.output_constraints is not None:
            if torch.any(infeasible_mask):
                Y_gt_inf = Y_gt[infeasible_mask].detach().cpu().numpy()
                axes.scatter(Y_gt_inf[:, 0], Y_gt_inf[:, 1], Y_gt_inf[:, 2], **infeasible_ground_truth_kwargs)

        # === Plot feasible non-pareto-front ground truth ===
        if torch.any(feasible_non_pareto_mask):
            Y_gt_feas_non_par = Y_gt[feasible_non_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_gt_feas_non_par[:, 0], Y_gt_feas_non_par[:, 1], Y_gt_feas_non_par[:, 2],
                         **feasible_non_pareto_ground_truth_kwargs)

        # === Plot feasible pareto-front ground truth ===
        if torch.any(feasible_mask):
            Y_gt_feas_par = Y_gt[feasible_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_gt_feas_par[:, 0], Y_gt_feas_par[:, 1], Y_gt_feas_par[:, 2],
                         **feasible_pareto_ground_truth_kwargs)

    """ Plot posterior """
    if show_posterior:
        # TODO: implement posterior plot
        raise NotImplementedError("Posterior not yet implemented.")
        # x = x.to(mobo.device, mobo.dtype)
        # posterior = mobo.model.posterior(x)
        # mean = posterior.mean.detach().cpu().numpy()
        # std = posterior.variance.sqrt().detach().cpu().numpy()
        #
        # axes.errorbar(
        #     mean[:, 0], mean[:, 1], mean[:, 2],
        #     xerr=3 * std[:, 0],
        #     yerr=3 * std[:, 1],
        #     zerr=3 * std[:, 2],
        #     fmt='o', **posterior_pareto_kwargs,
        # )

    """ Plot reference point """
    if show_ref_point is True:
        ref_point = mobo.objective.ref_point.detach().cpu().numpy()
        axes.scatter(ref_point[0], ref_point[1], ref_point[2], **ref_point_kwargs)

    """ Plot observations """
    if show_observations:

        Y_obj = mobo.Y_obj.clone()

        # === Compute feasible and infeasible masks ===
        if mobo.objective.output_constraints is None:
            feasible_mask = torch.ones(mobo.Y_obj.shape[-2], dtype=torch.bool)
            infeasible_mask = torch.zeros_like(mobo.Y_obj, dtype=torch.bool)
        else:
            Y_full = torch.cat([mobo.Y_obj, mobo.Y_con], dim=-1)
            feasible_mask = torch.stack([c(Y_full) <= 0 for c in mobo.objective.output_constraints]).all(dim=-2)
            infeasible_mask = torch.logical_not(feasible_mask)

        # === Compute pareto masks ===
        pareto_mask = torch.zeros_like(feasible_mask, dtype=torch.bool)
        if feasible_mask.any():
            Y_par = Y_obj.clone()
            Y_par[..., mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[feasible_mask]
            pareto_mask[feasible_mask] = is_non_dominated(Y_par)
        feasible_pareto_mask = torch.logical_and(feasible_mask, pareto_mask)
        feasible_non_pareto_mask = torch.logical_and(feasible_mask, torch.logical_not(pareto_mask))

        # === Plot infeasible observations ===
        if mobo.objective.output_constraints is not None:
            if torch.any(infeasible_mask):
                Y_obj_inf = Y_obj[infeasible_mask].detach().cpu().numpy()
                axes.scatter(Y_obj_inf[:, 0], Y_obj_inf[:, 1], Y_obj_inf[:, 2], **infeasible_observations_kwargs)

        # === Plot feasible non-pareto-front observations ===
        if torch.any(feasible_non_pareto_mask):
            Y_obj_feas_non_par = Y_obj[feasible_non_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_non_par[:, 0], Y_obj_feas_non_par[:, 1], Y_obj_feas_non_par[:, 2],
                         **feasible_non_pareto_observations_kwargs)

        # === Plot feasible pareto-front observations ===
        if torch.any(feasible_mask):
            Y_obj_feas_par = Y_obj[feasible_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_par[:, 0], Y_obj_feas_par[:, 1], Y_obj_feas_par[:, 2],
                         **feasible_pareto_observations_kwargs)

            # Plot Pareto surface
            if Y_obj_feas_par.shape[0] >= 3:
                # Reduce 3D points to 2D parameters
                pca = PCA(n_components=2)
                params_2d = pca.fit_transform(Y_obj_feas_par)  # shape (num_points, 2)
                # Triangulate in param space
                tri = Triangulation(params_2d[:, 0], params_2d[:, 1])
                # Plot trisurf in 3D using original coordinates and the 2D triangulation
                axes.plot_trisurf(
                    Y_obj_feas_par[:, 0],
                    Y_obj_feas_par[:, 1],
                    Y_obj_feas_par[:, 2],
                    triangles=tri.triangles,
                    cmap='Blues',
                    alpha=0.4,
                    edgecolor='gray'
                )

    axes.autoscale_view(tight=False)  # Update the view limits if limits are not set
    plt.legend()
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
    """

    hv = mobo.hypervolume

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
        output_path = Path.cwd() / "hv.png"

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Subplot 1: Cumulative HV ---
    axs[0].scatter(x, hv, **xy_plot_kwargs)
    axs[0].set_title("Hypervolume")
    axs[0].set_ylabel("HV")
    axs[0].grid(True)
    # axs[0].legend()
    if mobo.objective.max_hv is not None:
        axs[0].axhline(y=mobo.objective.max_hv, linestyle='--', color='black', label='Max HV')

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
        display_figure=False
):
    elapsed_time = mobo.elapsed_time

    if output_path is None:
        output_path = Path.cwd() / "elapsed_time.png"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("Number of observations (beyond initial points)")
    ax.set_ylabel("Elapsed Time (s)")
    x = np.array(range(len(elapsed_time))) * batch_size
    y = elapsed_time
    ax.scatter(x, y, **xy_plot_kwargs)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figure:
        plt.show()
    plt.close(fig)


def plot_parameters_evolution(
        mobo,
        parameter_names: list[str] | None = None,
        display_figure: bool = False,
        output_path: Path | None = None,
):
    X = mobo.X.clone().cpu().numpy()
    n = mobo.objective.dim
    bounds = mobo.objective.bounds.clone().cpu().numpy()
    if parameter_names is None:
        if mobo.objective.parameter_names is not None:
            parameter_names = mobo.objective.parameter_names
        else:
            parameter_names = [f"p{i}" for i in range(n)]

    for i in range(n):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(X.shape[0]), X[:, i], marker="o", label=parameter_names[i])
        plt.axhline(y=bounds[0][i], linestyle='--', color='black')
        plt.axhline(y=bounds[1][i], linestyle='--', color='black')
        plt.xlabel("Iteration")
        plt.ylabel(parameter_names[i])
        plt.title(f"Evolution of {parameter_names[i]}")
        plt.legend()
        plt.tight_layout()

        save_path = output_path or (
                Path.cwd() / f"parameter_evolution_{parameter_names[i].replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=300)

        if display_figure:
            plt.show()
        plt.close(fig)


def plot_objectives_evolution(
        mobo,
        objective_names: list[str] | None = None,
        display_figure: bool = False,
        output_path: Path | None = None,
):
    Y_obj = mobo.Y_obj.clone().cpu().numpy()
    n = mobo.objective.num_objectives
    if objective_names is None:
        if mobo.objective.objective_names is not None:
            objective_names = mobo.objective.objective_names
        else:
            objective_names = [f"obj{i}" for i in range(n)]

    for i in range(n):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(Y_obj.shape[0]), Y_obj[:, i], marker="o", label=objective_names[i])
        plt.xlabel("Iteration")
        plt.ylabel(objective_names[i])
        plt.title(f"Evolution of {objective_names[i]}")
        plt.legend()
        plt.tight_layout()

        save_path = output_path or (
                Path.cwd() / f"objective_evolution_{objective_names[i].replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=300)

        if display_figure:
            plt.show()
        plt.close(fig)


def plot_constraints_evolution(
        mobo,
        constraint_names: list[str] | None = None,
        display_figure: bool = False,
        output_path: Path | None = None,
):
    if mobo.objective.num_constraints == 0:
        return
    Y_con = mobo.Y_con.clone().cpu().numpy()  # assuming constraints stored in mobo.C
    n = mobo.objective.num_constraints
    if constraint_names is None:
        if mobo.objective.constraint_names is not None:
            constraint_names = mobo.objective.constraint_names
        else:
            constraint_names = [f"c{i}" for i in range(n)]

    for i in range(n):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(Y_con.shape[0]), Y_con[:, i], marker="o", label=constraint_names[i])
        plt.axhline(0.0, color="red", linestyle="--", label="feasibility boundary")
        plt.xlabel("Iteration")
        plt.ylabel(constraint_names[i])
        plt.title(f"Evolution of {constraint_names[i]}")
        plt.legend()
        plt.tight_layout()

        save_path = output_path or (
                Path.cwd() / f"constraint_evolution_{constraint_names[i].replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=300)

        if display_figure:
            plt.show()
        plt.close(fig)
