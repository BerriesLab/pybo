from collections.abc import Callable
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from fontTools.merge import cmap
from sklearn.decomposition import PCA
from matplotlib.tri import Triangulation
import matplotlib as mpl
from torch import Tensor
from torch.utils.data.datapipes.iter.routeddecoder import RoutedDecoderIterDataPipe

from constraints import output_constraints
from mobo.mobo import Mobo
from matplotlib.lines import Line2D
from botorch.utils.multi_objective import is_non_dominated

from objectives.base_class import MCMultiOutputBase

ms = 7

feasible_pareto_objectives_kwargs = {
    'marker': 'D',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolors': 'black',
    'alpha': 0.7,
    'label': 'Pareto Obs.'
}

feasible_non_pareto_objectives_kwargs = {
    'color': "tab:green",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Non-Pareto Obs.'
}

infeasible_objectives_kwargs = {
    'color': "tab:red",
    'marker': "x",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
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
                axes.scatter(Y_obj_inf[:, 0], Y_obj_inf[:, 1], **infeasible_objectives_kwargs)

        # === Plot feasible non-pareto-front observations ===
        if torch.any(feasible_non_pareto_mask):
            Y_obj_feas_non_par = Y_obj[feasible_non_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_non_par[:, 0], Y_obj_feas_non_par[:, 1], **feasible_non_pareto_objectives_kwargs)

        # === Plot feasible pareto-front observations ===
        if torch.any(feasible_mask):
            Y_obj_feas_par = Y_obj[feasible_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_par[:, 0], Y_obj_feas_par[:, 1], **feasible_pareto_objectives_kwargs)

    axes.autoscale_view(tight=False)  # Update the view limits if limits are not set
    plt.legend()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', format='png')

    if display_figure:
        plt.show()
    plt.close(fig)


class Plotter:
    def __init__(
            self,
            title: str,
            labels: List[str],
            lims: List[Tuple[float, float]] | None = None,
            figsize: Tuple[int, int] = (8, 7),
    ):
        self.title = title
        self.labels = labels
        self.lims = lims
        self.figsize = figsize
        self.fig, self.ax = self._initialize_figure()
        self.legend_elements = []
        self.cbar: plt.Colorbar | None = None

    def _initialize_figure(self):
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        ax.set_title(self.title)
        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])

        # X limits
        if self.lims is not None and self.lims[0] is not None:
            ax.set_xlim(self.lims[0][0], self.lims[0][1])
        else:
            ax.autoscale(enable=True, axis='x')

        # Y limits
        if self.lims is not None and self.lims[1] is not None:
            ax.set_ylim(self.lims[1][0], self.lims[1][1])
        else:
            ax.autoscale(enable=True, axis='y')

        return fig, ax


class ObjectivesPlotter(Plotter):
    def __init__(
            self,
            title: str,
            mobo: Mobo,
            f1_idx: int,
            f2_idx: int,
            f3_idx: int,
            pareto_idxs: list[int] | None = None,
            X_gt: torch.Tensor | None = None,
    ):
        super().__init__(
            title=title,
            labels=mobo.objective.objective_names
        )
        self.mobo = mobo
        # TODO: indexes can be a list
        self.f1_idx = f1_idx
        self.f2_idx = f2_idx
        self.f3_idx = f3_idx
        # TODO: the pareto indexes defines the objectives that must be used for the calculation of the pareto
        self.pareto_idxs = pareto_idxs
        self.X_gt = X_gt
        self.Y_obj_gt: torch.Tensor | None = None
        self.Y_con_gt: torch.Tensor | None = None
        self.feasible_obj_mask: torch.Tensor | None = None
        self.infeasible_obj_mask: torch.Tensor | None = None
        self.feasible_pareto_obj_mask: torch.Tensor | None = None
        self.feasible_non_pareto_obj_mask: torch.Tensor | None = None
        self.feasible_gt_mask: torch.Tensor | None = None
        self.infeasible_gt_mask: torch.Tensor | None = None
        self.feasible_pareto_gt_mask: torch.Tensor | None = None
        self.feasible_non_pareto_gt_mask: torch.Tensor | None = None

    def _add_colorbar(self):
        """
        Add a colorbar to the current figure using the existing colormap and normalization.
        Uses self.labels[2] as the label if available.
        """
        # Remove existing colorbar if present
        if self.cbar is not None:
            self.cbar.remove()
        # Create new colorbar
        self.cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
            ax=self.ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04
        )
        # Set label if available
        self.cbar.set_label(self.labels[self.f3_idx])
        # Refresh the figure
        self.fig.canvas.draw_idle()

    def _update_colormap(self):
        """Automatically recompute normalization after data changes."""
        vmin, vmax = 0.0, 1.0  # defaults

        if self.mobo.Y_obj is not None:
            Y_colors = self.mobo.Y_obj[..., self.f3_idx].detach().cpu().numpy()
            vmin, vmax = Y_colors.min(), Y_colors.max()

        if self.Y_obj_gt is not None:
            Y_colors_gt = self.Y_obj_gt[..., self.f3_idx].detach().cpu().numpy()
            vmin = min(vmin, Y_colors_gt.min())
            vmax = max(vmax, Y_colors_gt.max())

        # Update normalization and colormap
        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.cmap = mpl.cm.coolwarm

        # Update colorbar if it exists
        if self.cbar is not None:
            self.cbar.mappable.set_norm(self.norm)  # <-- update the mappable's norm
            self.cbar._draw_all()  # redraw the colorbar
            if self.fig is not None:
                self.fig.canvas.draw_idle()

    def plot_observations(self):
        # TODO: since at the end of the method the colorbar is updated, the colormap
        #  may be limited by default between 0 and 1
        self._compute_objectives_colormap()

        # === Compute feasible and infeasible masks ===
        self._compute_feasible_objectives_mask()
        self._compute_infeasible_objectives_mask()

        # === Compute objectives' pareto front masks ===
        self._compute_feasible_pareto_objectives_mask()
        self._compute_feasible_non_pareto_objectives_mask()

        # === Plot objectives ===
        self._plot_infeasible_objectives()
        self._plot_feasible_non_pareto_objectives()
        self._plot_feasible_pareto_objectives()
        if self.cbar is None:
            self._add_colorbar()
        self._update_colormap()

    def plot_ground_truth(self):
        if self.X_gt is None:
            raise ValueError("Must provide X for ground truth.")

        # === Compute ground truth ===
        self.Y_obj_gt = self.mobo.objective.evaluate_true_objective(self.X_gt)
        self.Y_con_gt = self.mobo.objective.evaluate_true_constraint(self.X_gt)
        # TODO: since at the end of the method the colorbar is updated, the colormap
        #  may be limited by default to between 0 and 1
        self._compute_ground_truth_colormap()

        # === Compute feasible and infeasible masks ===
        self._compute_feasible_ground_truth_mask()
        self._compute_infeasible_ground_truth_mask()

        # === Compute pareto masks ===
        self._compute_feasible_pareto_ground_truth_mask()
        self._compute_feasible_non_pareto_ground_truth_mask()

        # === Plot infeasible ground truth ===
        self._plot_infeasible_ground_truth()
        self._plot_feasible_non_pareto_ground_truth()
        self._plot_feasible_pareto_ground_truth()
        if self.cbar is None:
            self._add_colorbar()
        self._update_colormap()

    def _compute_objectives_colormap(self):
        if self.lims is not None and self.lims[2] is not None:
            self.norm = mpl.colors.Normalize(
                vmin=lims[self.f3_idx][0],
                vmax=lims[self.f3_idx][1]
            )
        else:
            Y_colors = self.mobo.Y_obj[..., self.f3_idx]
            self.norm = mpl.colors.Normalize(
                vmin=Y_colors.min().item(),
                vmax=Y_colors.max().item()
            )
        self.cmap = mpl.cm.coolwarm

    def _compute_ground_truth_colormap(self):
        if self.lims is not None and self.lims[2] is not None:
            self.norm = mpl.colors.Normalize(
                vmin=lims[self.f3_idx][0],
                vmax=lims[self.f3_idx][1]
            )
        else:
            self.norm = mpl.colors.Normalize(
                vmin=self.Y_obj_gt[..., self.f3_idx].min().item(),
                vmax=self.Y_obj_gt[..., self.f3_idx].max().item()
            )
        self.cmap = mpl.cm.coolwarm

    def update_colorbar_limits(self):
        """Update the color limits of the plot and the colorbar."""
        vmin = np.min(self.data)
        vmax = np.max(self.data)
        self.im.set_clim(vmin, vmax)
        self.cbar.set_clim(vmin, vmax)
        self.cbar.draw_all()  # Ensure the colorbar is redrawn
        self.fig.canvas.draw_idle()  # Refresh the figure

    def _compute_feasible_objectives_mask(self):
        """
        A function to compute a mask for the feasible observations. The mask is
        calculated for the objectives and constraints used in the optimization
        problem.
        """
        if self.mobo.Y_con is None:
            self.feasible_obj_mask = torch.ones(self.mobo.Y_obj.shape[-2], dtype=torch.bool)
        else:
            Y_full = torch.cat([self.mobo.Y_obj, self.mobo.Y_con], dim=-1)
            self.feasible_obj_mask = torch.stack([c(Y_full) <= 0 for c in self.mobo.objective.output_constraints]).all(
                dim=-2)

    def _compute_feasible_ground_truth_mask(self):
        if self.Y_con_gt is None:
            self.feasible_gt_mask = torch.ones(self.X_gt.shape[-2], dtype=torch.bool)
        else:
            Y_full = torch.cat([self.Y_obj_gt, self.Y_con_gt], dim=-1)
            self.feasible_gt_mask = torch.stack([c(Y_full) <= 0 for c in self.mobo.objective.output_constraints]).all(
                dim=-2)

    def _compute_infeasible_objectives_mask(self):
        """
        A function to compute a mask for the infeasible observations.
        """
        self.infeasible_obj_mask = torch.logical_not(self.feasible_obj_mask)

    def _compute_infeasible_ground_truth_mask(self):
        self.infeasible_gt_mask = torch.logical_not(self.feasible_gt_mask)

    def _compute_feasible_pareto_objectives_mask(self):
        """
        Compute the mask for feasible pareto points in the maximization space.
        """
        pareto_mask = torch.zeros_like(self.feasible_obj_mask, dtype=torch.bool)
        if self.feasible_obj_mask.any():
            Y_par = self.mobo.Y_obj.clone()
            Y_par[..., self.mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[self.feasible_obj_mask][..., self.pareto_idxs]
            pareto_mask[self.feasible_obj_mask] = is_non_dominated(Y_par)
        self.feasible_pareto_obj_mask = torch.logical_and(self.feasible_obj_mask, pareto_mask)

    def _compute_feasible_pareto_ground_truth_mask(self):
        pareto_mask = torch.zeros_like(self.feasible_gt_mask, dtype=torch.bool)
        if self.feasible_gt_mask.any():
            Y_par = self.Y_obj_gt.clone()
            Y_par[..., self.mobo.objective.obj_to_minimize] *= -1
            Y_par = Y_par[self.feasible_gt_mask][..., self.pareto_idxs]
            pareto_mask[self.feasible_gt_mask] = is_non_dominated(Y_par)
        self.feasible_pareto_gt_mask = torch.logical_and(self.feasible_gt_mask, pareto_mask)

    def _compute_feasible_non_pareto_objectives_mask(self):
        """
        Compute the mask for feasible non-pareto points.
        """
        self.feasible_non_pareto_obj_mask = torch.logical_not(self.feasible_pareto_obj_mask)

    def _compute_feasible_non_pareto_ground_truth_mask(self):
        self.feasible_non_pareto_gt_mask = torch.logical_not(self.feasible_pareto_gt_mask)

    def _plot_infeasible_objectives(self):
        """
        Plot the infeasible observations. Y is a torch.Tensor with the objective values.
        """
        if torch.any(self.infeasible_obj_mask):
            Y = self.mobo.Y_obj[mask].detach().cpu().numpy()
            color_values = Y[:, self.f3_idx]
            c = self.cmap(self.norm(color_values))
            kwargs = dict(infeasible_objectives_kwargs)  # make a local copy
            kwargs.pop("color", None)
            self.ax.scatter(x=Y[:, self.f1_idx], y=Y[:, self.f2_idx], c=c, **kwargs)
            self.legend_elements.append(Line2D(
                [0], [0], marker=infeasible_objectives_kwargs['marker'],
                label=infeasible_objectives_kwargs['label'],
                color=infeasible_objectives_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None')
            )

    def _plot_infeasible_ground_truth(self):
        """
        Plot the infeasible observations. Y is a torch.Tensor with the objective values.
        """
        if torch.any(self.infeasible_gt_mask):
            Y = self.mobo.Y_obj_gt[mask].detach().cpu().numpy()
            color_values = Y[:, self.f3_idx]
            c = self.cmap(self.norm(color_values))
            kwargs = dict(infeasible_ground_truth_kwargs)  # make a local copy
            kwargs.pop("color", None)
            self.ax.scatter(x=Y[:, self.f1_idx], y=Y[:, self.f2_idx], c=c, **kwargs)
            self.legend_elements.append(Line2D(
                [0], [0], marker=infeasible_ground_truth_kwargs['marker'],
                label=infeasible_ground_truth_kwargs['label'],
                color=infeasible_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None')
            )

    def _plot_feasible_non_pareto_objectives(self):
        """
        Plot the feasible non-pareto observations. Y_obj is a torch.Tensor with the objective values.
        """
        if torch.any(self.feasible_non_pareto_obj_mask):
            Y = self.mobo.Y_obj[self.feasible_non_pareto_obj_mask].detach().cpu().numpy()
            color_values = Y[:, self.f3_idx]
            kwargs = dict(feasible_non_pareto_objectives_kwargs)  # make a local copy
            kwargs.pop("color", None)
            c = self.cmap(self.norm(color_values))
            self.ax.scatter(x=Y[:, self.f1_idx], y=Y[:, self.f2_idx], c=c, **kwargs)
            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_objectives_kwargs['marker'],
                label=feasible_non_pareto_objectives_kwargs['label'],
                color=feasible_non_pareto_objectives_kwargs['color'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

    def _plot_feasible_non_pareto_ground_truth(self):
        if torch.any(self.feasible_non_pareto_gt_mask):
            Y = self.Y_obj_gt[self.feasible_non_pareto_gt_mask].detach().cpu().numpy()
            color_values = Y[:, self.f3_idx]
            kwargs = dict(feasible_non_pareto_ground_truth_kwargs)  # make a local copy
            kwargs.pop("color", None)
            c = self.cmap(self.norm(color_values))
            self.ax.scatter(x=Y[:, 0], y=Y[:, 1], c=c, **kwargs)
            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_ground_truth_kwargs['marker'],
                label=feasible_non_pareto_ground_truth_kwargs['label'],
                color=feasible_non_pareto_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

    def _plot_feasible_pareto_objectives(self):
        """
        Plot the feasible pareto observations. Y_obj is a torch.Tensor with the objective values.
        """
        if torch.any(self.feasible_pareto_obj_mask):
            Y = self.mobo.Y_obj[self.feasible_pareto_obj_mask].detach().cpu().numpy()
            color_values = Y[:, 2]
            c = self.cmap(self.norm(color_values))
            kwargs = dict(feasible_pareto_objectives_kwargs)  # make a local copy
            kwargs.pop("color", None)
            self.ax.scatter(x=Y[:, self.f1_idx], y=Y[:, self.f2_idx], c=c, **kwargs)
            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_objectives_kwargs['marker'],
                label=feasible_pareto_objectives_kwargs['label'],
                color=feasible_pareto_objectives_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None')
            )

    def _plot_feasible_pareto_ground_truth(self):
        if torch.any(self.feasible_pareto_gt_mask):
            Y = self.Y_obj_gt[self.feasible_pareto_gt_mask].detach().cpu().numpy()
            color_values = Y[:, 2]
            c = self.cmap(self.norm(color_values))
            kwargs = dict(feasible_pareto_ground_truth_kwargs)  # make a local copy
            kwargs.pop("color", None)
            self.ax.scatter(x=Y[:, self.f1_idx], y=Y[:, self.f2_idx], c=c, **kwargs)
            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_ground_truth_kwargs['marker'],
                label=feasible_pareto_ground_truth_kwargs['label'],
                color=feasible_pareto_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None')
            )

    @staticmethod
    def show():
        plt.show()

    def save_figure(self, fname: str | Path | None = None):
        if fname is None:
            fname = self.title.replace(" ", "_").lower() + ".png"
        self.fig.savefig(fname=Path.cwd() / fname)


class TrackersPlotter(Plotter):
    def __init__(
            self,
    ):
        super().__init__(title, labels, lims)


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
                axes.scatter(Y_obj_inf[:, 0], Y_obj_inf[:, 1], Y_obj_inf[:, 2], **infeasible_objectives_kwargs)

        # === Plot feasible non-pareto-front observations ===
        if torch.any(feasible_non_pareto_mask):
            Y_obj_feas_non_par = Y_obj[feasible_non_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_non_par[:, 0], Y_obj_feas_non_par[:, 1], Y_obj_feas_non_par[:, 2],
                         **feasible_non_pareto_objectives_kwargs)

        # === Plot feasible pareto-front observations ===
        if torch.any(feasible_mask):
            Y_obj_feas_par = Y_obj[feasible_pareto_mask].detach().cpu().numpy()
            axes.scatter(Y_obj_feas_par[:, 0], Y_obj_feas_par[:, 1], Y_obj_feas_par[:, 2],
                         **feasible_pareto_objectives_kwargs)

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


def plot_trackers_evolution(
        mobo,
        tracker_names: list[str] | None = None,
        display_figure: bool = False,
        output_path: Path | None = None,
):
    if mobo.objective.num_trackers == 0:
        return
    Y_track = mobo.Y_track.clone().cpu().numpy()  # assuming constraints stored in mobo.C
    n = mobo.objective.num_trackers
    if tracker_names is None:
        if mobo.objective.tracker_names is not None:
            tracker_names = mobo.objective.tracker_names
        else:
            tracker_names = [f"c{i}" for i in range(n)]

    for i in range(n):
        fig = plt.figure(figsize=(10, 6))
        plt.plot(range(Y_track.shape[0]), Y_track[:, i], marker="o", label=tracker_names[i])
        plt.xlabel("Iteration")
        plt.ylabel(tracker_names[i])
        plt.title(f"Evolution of {tracker_names[i]}")
        plt.legend()
        plt.tight_layout()

        save_path = output_path or (
                Path.cwd() / f"tracker_evolution_{tracker_names[i].replace(' ', '_').lower()}.png")
        plt.savefig(save_path, dpi=300)

        if display_figure:
            plt.show()
        plt.close(fig)
