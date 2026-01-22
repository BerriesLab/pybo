from pathlib import Path
from bayesian_optimizer.optimizer import BayesianOptimizer
from objectives.base_class import MCSingleObjectiveBase
from plotters.base_class import PlotterBase
import numpy as np
import matplotlib.pyplot as plt
from plotters.styles import *


class Acqf1DPlotter(PlotterBase):
    """ A class for visualizing acquisition function values. """

    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

        if not isinstance(bo.objective, MCSingleObjectiveBase):
            raise TypeError("Objective must be of type MCSingleObjectiveBase")

        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.ax.set_xlabel(bo.objective.objective_names[0] if bo.objective.objective_names is not None else r"$x$")
        self.ax.set_ylabel(bo.objective.objective_names[1] if bo.objective.objective_names is not None else r"$f(x)$")

    def plot_acquisition(self):
        if self.bo.acqf_instance is None:
            raise ValueError("Acquisition function must be set before plotting.")

        X_grid = self._generate_uniform_grid()

        with torch.no_grad():
            acq_values = self.bo.acqf_instance(X_grid.unsqueeze(1))

        X_np = X_grid.squeeze().detach().cpu().numpy()
        acq_np = acq_values.squeeze().detach().cpu().numpy()

        if getattr(self.bo.acqf, "_log"):
            log_abs_acqf = np.log(np.abs(acq_np))
            self.ax.plot(X_np, -log_abs_acqf, **acqf_1d)
            self.ax.set_ylabel(r'$-\log \left( | \mathrm{Acquisition\ Value} | \right) $')
        else:
            self.ax.plot(X_np, acq_np, **acqf_1d)
            self.ax.set_ylabel(r'$\mathrm{Acquisition\ Value}$')

        return self

    def plot_next_X(self):
        X = self.bo.new_X.detach().cpu().numpy()
        if X is not None:
            if X.ndim == 0:
                X = [X.item()]
            for i, x in enumerate(X):
                self.ax.axvline(x=x, **new_X_1d)
        return self

    def plot(self):
        self.plot_acquisition()
        self.plot_next_X()
        self.ax.legend(loc='upper right', fontsize='small', frameon=True)
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acqf.png"
        return super().save_figure(filename=filename)


class Acqf2DPlotter(PlotterBase):

    def __init__(self, bo: BayesianOptimizer):
        super().__init__(bo=bo)

    def plot_acquisition(self, n_grid_points: int = 50, cmap: str = 'viridis', levels: int = 50):
        """Plot the acquisition landscape using filled contours."""
        if self.bo.acqf_instance is None:
            raise ValueError("Acquisition function must be set before plotting.")

        X_grid = self._generate_uniform_grid(n_points_per_dim=n_grid_points)

        with torch.no_grad():
            acq_values = self.bo.acqf_instance(X_grid.unsqueeze(1))

        # Reshape for contour plotting: (N*N) -> (N, N)
        acq_np = acq_values.reshape(n_grid_points, n_grid_points).detach().cpu().numpy()

        # Get bounds for the extent of the plot
        b = self.bo.objective.bounds.cpu().numpy()

        if getattr(self.bo.acqf, "_log", False):
            acq_np = -np.log(np.abs(acq_np) + 1e-9)
            label = r'$-\log(|Acq|)$'
        else:
            label = 'Acquisition Value'

        x = np.linspace(b[0, 0], b[1, 0], n_grid_points)
        y = np.linspace(b[0, 1], b[1, 1], n_grid_points)

        # Plotting
        contour = self.ax.contourf(x, y, acq_np.T, levels=50, cmap=cmap)
        # Adding the Colorbar
        cbar = self.ax.figure.colorbar(contour, ax=self.ax)

        cbar.set_label(label)
        self.ax.set_xlabel('$x_1$')
        self.ax.set_ylabel('$x_2$')
        return self

    def plot_next_X(self):
        """Mark the next suggested point with a prominent star."""
        if self.bo.new_X is not None:
            X = self.bo.new_X.detach().cpu().numpy()
            self.ax.scatter(
                X[:, 0], X[:, 1],
                color='red', marker='*', s=200,
                edgecolor='white', label="Next X", zorder=10
            )
        return self

    def plot(self):
        self.plot_acquisition()
        self.plot_next_X()
        self.ax.legend(loc="best")
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acquisition.png"
        return super().save_figure(filename=filename)
