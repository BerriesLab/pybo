from pathlib import Path
import torch
from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.base_class import PlotterBase
import numpy as np


class AcquisitionPlotter(PlotterBase):
    """ A class for visualizing acquisition function values. """

    def __init__(
            self,
            bayesian_optimizer: BayesianOptimizer,
            lims: list[tuple[float, float]] | None = None,
    ):
        super().__init__(bayesian_optimizer=bayesian_optimizer, lims=lims)

    def plot_acquisition(self, color: str = 'green', linewidth: float = 1.5,
                         label: str = 'Acquisition Function'):
        """Plot the acquisition function values."""
        if self.bo.acquisition_function_instance is None:
            raise ValueError("Acquisition function must be set before plotting.")

        X_grid = self._generate_grid()

        with torch.no_grad():
            acq_values = self.bo.acquisition_function_instance(X_grid.unsqueeze(1))

        X_np = X_grid.squeeze().detach().cpu().numpy()
        acq_np = acq_values.squeeze().detach().cpu().numpy()

        if self.bo.acquisition_function_factory.acquisition_function_type.is_log():
            log_abs_acqf = np.log(np.abs(acq_np))
            self.ax.plot(X_np, -log_abs_acqf, color=color, linewidth=linewidth, label=label)
            self.ax.set_ylabel(r'$-\log \left( | \mathrm{Acquisition\ Value} | \right) $')
        else:
            self.ax.plot(X_np, acq_np, color=color, linewidth=linewidth, label=label)
            self.ax.set_ylabel(r'$\mathrm{Acquisition\ Value}$')

        return self

    def plot_next_X(self):
        """Mark the next suggested point."""
        X = self.bo.new_X.detach().cpu().numpy()
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

    def plot_legend(self, loc: str = "best"):
        self.ax.legend(loc=loc)
        return self

    def plot(self):
        self.plot_acquisition()
        self.plot_next_X()
        self.plot_legend()
        return self

    def save_figure(self, filename: str | Path | None = None):
        filename = filename or "acquisition.png"
        return super().save_figure(filename=filename)
