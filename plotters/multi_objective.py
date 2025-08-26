import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.decomposition import PCA
from matplotlib.tri import Triangulation
import matplotlib as mpl

from mobo.mobo import Mobo
from matplotlib.lines import Line2D
from botorch.utils.multi_objective import is_non_dominated

from plotters.base_class import PlotterBase
from plotters.utils import *


class MultiObjectivePlotter(PlotterBase):
    """
    A class for visualizing multi-objective functions from a Mobo object in 2D.
    The class includes methods requiring only Minimal user input; most settings
    are automatically inferred from the `Mobo` object.

    Key features:
    - Maps N-dimensional objective data to a 2D scatter plot.
    - Supports color-coding based on a selected objective or tracker.
    - Can highlight Pareto fronts for selected objectives.
    - Can include ground-truth evaluations if provided.
    """

    def __init__(
            self,
            mobo: Mobo,
            title: str = "Pareto front",
            idx_x=0,
            idx_y=1,
            idx_color: int | None = None,
            pareto_idxs: list[int] | None = None,
            use_tracker: bool = False,
            X_gt: torch.Tensor | None = None,
    ):
        """

        Args:
            - title (str): The title for the plot.
            - mobo (Mobo): The `Mobo` object containing objective data to plot.
            - idx_x (int, optional): Index of the objective to plot on the x-axis. Default is 0.
            - idx_y (int, optional): Index of the objective to plot on the y-axis. Default is 1.
            - idx_color (int, optional): Index of the objective used for color-coding. Default is 2.
            - pareto_idxs (list[int] | None, optional): Indices of objectives used to determine the Pareto front.
            - use_tracker (bool, optional): Whether to use tracker data for color-coding. If True,
                `idx_color` is applied to tracker values. Default is False.
            - X_gt (torch.Tensor | None, optional): Input parameters used to compute ground-truth objectives.
                Default is None.
        """
        super().__init__(
            title=title,
            labels=mobo.objective.objective_names
        )
        self.mobo = mobo
        self.idx_x = idx_x
        self.idx_y = idx_y
        self.idx_color = idx_color
        self.pareto_idxs = pareto_idxs
        self.use_tracker = use_tracker
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

    def plot_objectives(self):
        self._initialize_norm()
        self._initialize_colormap()

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

        # === Update colormaps ===
        if self.idx_color is not None:
            if self.cbar is None:
                self._add_colorbar()
            self._update_cmap_and_norm()

        # === Update legend ===
        self.ax.legend()

        return self

    def plot_ground_truth(self):
        if self.X_gt is None:
            raise ValueError("Must provide X for ground truth.")

        self._initialize_norm()
        self._initialize_colormap()

        # === Compute ground truth ===
        self.Y_obj_gt = self.mobo.objective.evaluate_true_objective(self.X_gt)
        self.Y_con_gt = self.mobo.objective.evaluate_true_constraint(self.X_gt)

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

        # === Update colormaps ===
        if self.idx_color is not None:
            if self.cbar is None:
                self._add_colorbar()
            self._update_cmap_and_norm()

        # === Update legend ===
        self.ax.legend()

        return self

    def _compute_feasible_objectives_mask(self):
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
        self.infeasible_obj_mask = torch.logical_not(self.feasible_obj_mask)

    def _compute_infeasible_ground_truth_mask(self):
        self.infeasible_gt_mask = torch.logical_not(self.feasible_gt_mask)

    def _compute_feasible_pareto_objectives_mask(self):
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
        self.feasible_non_pareto_obj_mask = torch.logical_and(
            self.feasible_obj_mask, torch.logical_not(
                self.feasible_pareto_obj_mask
            )
        )

    def _compute_feasible_non_pareto_ground_truth_mask(self):
        self.feasible_non_pareto_gt_mask = torch.logical_and(
            self.feasible_gt_mask, torch.logical_not(
                self.feasible_pareto_gt_mask
            )
        )

    def _plot_infeasible_objectives(self):
        if torch.any(self.infeasible_obj_mask):
            Y = self.mobo.Y_obj[self.infeasible_obj_mask].detach().cpu().numpy()
            kwargs = dict(infeasible_objectives_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.mobo.objective.evaluate_trackers(X=self.mobo.X)[..., self.idx_color]
                    color_values = color_values[self.infeasible_obj_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=infeasible_objectives_kwargs['marker'],
                label=infeasible_objectives_kwargs['label'],
                color=infeasible_objectives_kwargs['color'],
                markerfacecolor='none', markersize=ms, linestyle='None')
            )

    def _plot_infeasible_ground_truth(self):
        if torch.any(self.infeasible_gt_mask):
            Y = self.Y_obj_gt[self.infeasible_gt_mask].detach().cpu().numpy()
            kwargs = dict(infeasible_ground_truth_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.mobo.objective.evaluate_trackers(X=self.X_gt)[..., self.idx_color]
                    color_values = color_values[self.infeasible_gt_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=infeasible_ground_truth_kwargs['marker'],
                label=infeasible_ground_truth_kwargs['label'],
                color=infeasible_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None')
            )

    def _plot_feasible_non_pareto_objectives(self):
        if torch.any(self.feasible_non_pareto_obj_mask):
            Y = self.mobo.Y_obj[self.feasible_non_pareto_obj_mask].detach().cpu().numpy()
            kwargs = dict(feasible_non_pareto_objectives_kwargs)  # make a local copy

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.mobo.objective.evaluate_trackers(X=self.mobo.X)[..., self.idx_color]
                    color_values = color_values[self.feasible_non_pareto_obj_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_objectives_kwargs['marker'],
                label=feasible_non_pareto_objectives_kwargs['label'],
                color=feasible_non_pareto_objectives_kwargs['color'],
                markerfacecolor='none', markersize=ms, linestyle='None'))

    def _plot_feasible_non_pareto_ground_truth(self):
        if torch.any(self.feasible_non_pareto_gt_mask):
            Y = self.Y_obj_gt[self.feasible_non_pareto_gt_mask].detach().cpu().numpy()
            kwargs = dict(feasible_non_pareto_ground_truth_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.mobo.objective.evaluate_trackers(X=self.X_gt)[..., self.idx_color]
                    color_values = color_values[self.feasible_non_pareto_gt_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_ground_truth_kwargs['marker'],
                label=feasible_non_pareto_ground_truth_kwargs['label'],
                color=feasible_non_pareto_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None'))

    def _plot_feasible_pareto_objectives(self):
        if torch.any(self.feasible_pareto_obj_mask):
            Y = self.mobo.Y_obj[self.feasible_pareto_obj_mask].detach().cpu().numpy()
            kwargs = dict(feasible_pareto_objectives_kwargs)  # make a local copy

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.mobo.objective.evaluate_trackers(X=self.mobo.X)[..., self.idx_color]
                    color_values = color_values[self.feasible_pareto_obj_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_objectives_kwargs['marker'],
                label=feasible_pareto_objectives_kwargs['label'],
                color=feasible_pareto_objectives_kwargs['edgecolors'],
                markerfacecolor='none', markersize=ms, linestyle='None')
            )

    def _plot_feasible_pareto_ground_truth(self):
        if torch.any(self.feasible_pareto_gt_mask):
            Y = self.Y_obj_gt[self.feasible_pareto_gt_mask].detach().cpu().numpy()
            kwargs = dict(feasible_pareto_ground_truth_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.mobo.objective.evaluate_trackers(X=self.X_gt)[..., self.idx_color]
                    color_values = color_values[self.feasible_pareto_gt_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_ground_truth_kwargs['marker'],
                label=feasible_pareto_ground_truth_kwargs['label'],
                color=feasible_pareto_ground_truth_kwargs['color'],
                markerfacecolor='none', markersize=ms * 0.45, linestyle='None')
            )

    def _initialize_colormap(self):
        self.cmap = mpl.cm.get_cmap("coolwarm")

    def _initialize_norm(self):
        if self.lims is not None and self.lims[2] is not None:
            self.norm = mpl.colors.Normalize(
                vmin=self.lims[self.idx_color][0],
                vmax=self.lims[self.idx_color][1],
            )
        else:
            self.norm = mpl.colors.Normalize(
                vmin=0,
                vmax=1,
            )

    def _add_colorbar(self):
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
        self.cbar.set_label(self.labels[self.idx_color])

    def _update_cmap_and_norm(self):
        if self.lims is not None and self.lims[2] is not None:
            return

        vmin, vmax = 0.0, 1.0  # defaults

        if self.use_tracker and self.mobo.Y_track is not None:
            Y_colors = self.mobo.Y_track[..., self.idx_color]
            vmin, vmax = Y_colors.min(), Y_colors.max()
        else:
            if self.mobo.Y_obj is not None:
                Y_colors = self.mobo.Y_obj[..., self.idx_color]
                vmin, vmax = Y_colors.min(), Y_colors.max()

            if self.Y_obj_gt is not None:
                Y_colors_gt = self.Y_obj_gt[..., self.idx_color]
                vmin = min(vmin, Y_colors_gt.min())
                vmax = max(vmax, Y_colors_gt.max())

        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        for ax in self.fig.axes:
            for coll in ax.collections:
                coll.set_norm(self.norm)
                coll.set_cmap(self.cmap)
                coll.changed()

        if self.cbar is not None:
            self.cbar.mappable.set_norm(self.norm)  # Explicitly update the mappable's norm
            self.cbar.update_ticks()  # Update the ticks

        self.fig.canvas.draw()


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
    axs[0].plot(x, hv, **xy_plot_kwargs)
    axs[0].set_title("Hypervolume")
    axs[0].set_ylabel("HV")
    axs[0].grid(True)
    # axs[0].legend()
    if mobo.objective.max_hv is not None:
        axs[0].axhline(y=mobo.objective.max_hv, linestyle='--', color='black', label='Max HV')

    # --- Subplot 2: log10(HVI) ---
    mask = ~np.isnan(hvi_log)
    axs[1].plot(x[mask], hvi_log[mask], **xy_plot_kwargs)
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
    ax.plot(x, y, **xy_plot_kwargs)
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
        plt.plot(range(X.shape[0]), X[:, i], **xy_plot_kwargs, label=parameter_names[i])
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
        plt.plot(range(Y_obj.shape[0]), Y_obj[:, i], **xy_plot_kwargs, label=objective_names[i])
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
        plt.plot(range(Y_con.shape[0]), Y_con[:, i], **xy_plot_kwargs, label=constraint_names[i])
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
        plt.plot(range(Y_track.shape[0]), Y_track[:, i], **xy_plot_kwargs, label=tracker_names[i])
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
