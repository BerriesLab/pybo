import numpy as np
import matplotlib as mpl
from bayesian_optimizer.optimizer import BayesianOptimizer
from matplotlib.lines import Line2D
from botorch.utils.multi_objective import is_non_dominated
from plotters.base_class import PlotterBase
from plotters.utils import *


class MultiObjectivePlotter(PlotterBase):
    """
    A class for visualizing bi- and three-objective optimization problems.
    The class includes methods requiring only Minimal user input: most settings
    are automatically inferred from the passed BayesianOptimizer.

    Key features:
    - Maps 2 and 3-dimensional objective data to a 2D scatter plot.
    - Supports color-coding based on a selected objective or tracker.
    - Can highlight Pareto fronts for selected objectives.
    - Can include ground-truth evaluations if provided.
    """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            title: str | None = "Pareto front",
            idx_x: int = 0,
            idx_y: int = 1,
            idx_color: int | None = None,
            pareto_idxs: list[int] | None = None,
            use_tracker: bool = False,
            X_gt: torch.Tensor | None = None,
    ):
        """
        Args:
            - title (str): The title for the plot.
            - bayesian_optimizer: The object containing objective data to plot.
            - idx_x (int, optional): Index of the objective to plot on the x-axis. Default is 0.
            - idx_y (int, optional): Index of the objective to plot on the y-axis. Default is 1.
            - idx_color (int, optional): Index of the objective used for color-coding. Default is 2.
            - pareto_idxs (list[int] | None, optional): Indices of objectives used to determine the Pareto front.
            - use_tracker (bool, optional): Whether to use tracker data for color-coding. If True,
                `idx_color` is applied to tracker values. Default is False.
            - X_gt (torch.Tensor | None, optional): Input parameters used to compute ground-truth objectives.
                Default is None.
        """
        super().__init__(title=title)
        self.bayesian_optimizer = bayesian_optimizer
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
        self.update_labels()

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
        self.ax.legend(handles=self.legend_elements, loc="best")

        return self

    def plot_ground_truth(self):
        if self.X_gt is None:
            raise ValueError("Must provide X for ground truth.")

        self._initialize_norm()
        self._initialize_colormap()
        self.update_labels()

        # === Compute ground truth ===
        self.Y_obj_gt = self.bayesian_optimizer.objective.evaluate_true_objective(self.X_gt)
        self.Y_con_gt = self.bayesian_optimizer.objective.evaluate_true_constraint(self.X_gt)

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
        self.ax.legend(handles=self.legend_elements, loc="best")

        return self

    def _compute_feasible_objectives_mask(self):
        if self.bayesian_optimizer.Y_con is None:
            self.feasible_obj_mask = torch.ones(
                self.bayesian_optimizer.Y_obj.shape[-2],
                device=self.bayesian_optimizer.device, dtype=torch.bool
            )
        else:
            Y_full = torch.cat(
                [self.bayesian_optimizer.Y_obj,
                 self.bayesian_optimizer.Y_con],
                dim=-1
            )
            self.feasible_obj_mask = torch.stack(
                [c(Y_full) <= 0 for c in self.bayesian_optimizer.objective.output_constraints]
            ).all(dim=-2)

    def _compute_feasible_ground_truth_mask(self):
        if self.Y_con_gt is None:
            self.feasible_gt_mask = torch.ones(self.X_gt.shape[-2], device=self.bayesian_optimizer.device,
                                               dtype=torch.bool)
        else:
            Y_full = torch.cat([self.Y_obj_gt, self.Y_con_gt], dim=-1)
            self.feasible_gt_mask = torch.stack(
                [c(Y_full) <= 0 for c in self.bayesian_optimizer.objective.output_constraints]).all(
                dim=-2)

    def _compute_infeasible_objectives_mask(self):
        self.infeasible_obj_mask = torch.logical_not(self.feasible_obj_mask)

    def _compute_infeasible_ground_truth_mask(self):
        self.infeasible_gt_mask = torch.logical_not(self.feasible_gt_mask)

    def _compute_feasible_pareto_objectives_mask(self):
        pareto_mask = torch.zeros_like(self.feasible_obj_mask, device=self.bayesian_optimizer.device, dtype=torch.bool)
        if self.feasible_obj_mask.any():
            Y_par = self.bayesian_optimizer.Y_obj.clone()
            Y_par[..., self.bayesian_optimizer.objective.obj_to_minimize] *= -1
            Y_par = Y_par[self.feasible_obj_mask][..., self.pareto_idxs]
            pareto_mask[self.feasible_obj_mask] = is_non_dominated(Y_par)
        self.feasible_pareto_obj_mask = torch.logical_and(self.feasible_obj_mask, pareto_mask)

    def _compute_feasible_pareto_ground_truth_mask(self):
        pareto_mask = torch.zeros_like(self.feasible_gt_mask, device=self.bayesian_optimizer.device, dtype=torch.bool)
        if self.feasible_gt_mask.any():
            Y_par = self.Y_obj_gt.clone()
            Y_par[..., self.bayesian_optimizer.objective.obj_to_minimize] *= -1
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
            Y = self.bayesian_optimizer.Y_obj[self.infeasible_obj_mask].detach().cpu().numpy()
            kwargs = dict(infeasible_objectives_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.bayesian_optimizer.objective.evaluate_trackers(X=self.bayesian_optimizer.X)[
                        ..., self.idx_color]
                    color_values = color_values[self.infeasible_obj_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=infeasible_objectives_kwargs['marker'],
                label=infeasible_objectives_kwargs['label'],
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=np.sqrt(infeasible_objectives_kwargs["s"]),
                linestyle='None')
            )

    def _plot_infeasible_ground_truth(self):
        if torch.any(self.infeasible_gt_mask):
            Y = self.Y_obj_gt[self.infeasible_gt_mask].detach().cpu().numpy()
            kwargs = dict(infeasible_ground_truth_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.bayesian_optimizer.objective.evaluate_trackers(X=self.X_gt)[..., self.idx_color]
                    color_values = color_values[self.infeasible_gt_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=infeasible_ground_truth_kwargs['marker'],
                label=infeasible_ground_truth_kwargs['label'],
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=np.sqrt(infeasible_ground_truth_kwargs["s"]),
                linestyle='None')
            )

    def _plot_feasible_non_pareto_objectives(self):
        if torch.any(self.feasible_non_pareto_obj_mask):
            Y = self.bayesian_optimizer.Y_obj[self.feasible_non_pareto_obj_mask].detach().cpu().numpy()
            kwargs = dict(feasible_non_pareto_objectives_kwargs)  # make a local copy

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.bayesian_optimizer.objective.evaluate_trackers(X=self.bayesian_optimizer.X)[
                        ..., self.idx_color]
                    color_values = color_values[self.feasible_non_pareto_obj_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_objectives_kwargs['marker'],
                label=feasible_non_pareto_objectives_kwargs['label'],
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=np.sqrt(feasible_non_pareto_objectives_kwargs["s"]),
                linestyle='None')
            )

    def _plot_feasible_non_pareto_ground_truth(self):
        if torch.any(self.feasible_non_pareto_gt_mask):
            Y = self.Y_obj_gt[self.feasible_non_pareto_gt_mask].detach().cpu().numpy()
            kwargs = dict(feasible_non_pareto_ground_truth_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.bayesian_optimizer.objective.evaluate_trackers(X=self.X_gt)[..., self.idx_color]
                    color_values = color_values[self.feasible_non_pareto_gt_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_non_pareto_ground_truth_kwargs['marker'],
                label=feasible_non_pareto_ground_truth_kwargs['label'],
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=np.sqrt(feasible_non_pareto_ground_truth_kwargs["s"]),
                linestyle='None'))

    def _plot_feasible_pareto_objectives(self):
        if torch.any(self.feasible_pareto_obj_mask):
            Y = self.bayesian_optimizer.Y_obj[self.feasible_pareto_obj_mask].detach().cpu().numpy()
            kwargs = dict(feasible_pareto_objectives_kwargs)  # make a local copy

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.bayesian_optimizer.objective.evaluate_trackers(X=self.bayesian_optimizer.X)[
                        ..., self.idx_color]
                    color_values = color_values[self.feasible_pareto_obj_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_objectives_kwargs['marker'],
                label=feasible_pareto_objectives_kwargs['label'],
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=np.sqrt(feasible_pareto_objectives_kwargs["s"]),
                linestyle='None')
            )

    def _plot_feasible_pareto_ground_truth(self):
        if torch.any(self.feasible_pareto_gt_mask):
            Y = self.Y_obj_gt[self.feasible_pareto_gt_mask].detach().cpu().numpy()
            kwargs = dict(feasible_pareto_ground_truth_kwargs)

            if self.idx_color is None:
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], **kwargs)
            else:
                if self.use_tracker:
                    color_values = self.bayesian_optimizer.objective.evaluate_trackers(X=self.X_gt)[..., self.idx_color]
                    color_values = color_values[self.feasible_pareto_gt_mask].detach().cpu().numpy()
                else:
                    color_values = Y[:, self.idx_color]
                kwargs.pop("color", None)
                self.ax.scatter(x=Y[:, self.idx_x], y=Y[:, self.idx_y], c=color_values, **kwargs)

            self.legend_elements.append(Line2D(
                [0], [0], marker=feasible_pareto_ground_truth_kwargs['marker'],
                label=feasible_pareto_ground_truth_kwargs['label'],
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=np.sqrt(feasible_pareto_ground_truth_kwargs["s"]),
                linestyle='None')
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

        vmin, vmax = 0.0, 1.0

        if self.use_tracker and self.bayesian_optimizer.Y_track is not None:
            Y_colors = self.bayesian_optimizer.Y_track[..., self.idx_color]
            vmin, vmax = Y_colors.min(), Y_colors.max()
        else:
            if self.bayesian_optimizer.Y_obj is not None:
                Y_colors = self.bayesian_optimizer.Y_obj[..., self.idx_color]
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
            self.cbar.mappable.set_norm(self.norm)
            self.cbar.update_ticks()

        self.fig.canvas.draw()

    def update_labels(self, labels: list[str] | None = None):
        """ Use a list of strings if provided, otherwise infer the labels from the objective.
        If it cannot infer, use default labels: f_1, f_2, and optionally f_3 if idx_color is set. """
        if labels is None:
            if self.bayesian_optimizer.objective.objective_names is not None:
                self.labels = [
                    self.bayesian_optimizer.objective.objective_names[self.idx_x],
                    self.bayesian_optimizer.objective.objective_names[self.idx_y],
                ]
                if self.idx_color is not None:
                    self.labels.append(
                        self.bayesian_optimizer.objective.tracker_names[self.idx_color]
                        if self.use_tracker
                        else self.bayesian_optimizer.objective.objective_names[self.idx_color]
                    )
            else:
                self.labels = ["$f_1$", "$f_2$"]
                if self.idx_color is not None:
                    self.labels.append("$f_3$")
        else:
            self.labels = labels

        self._set_labels()
