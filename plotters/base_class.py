from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt


class PlotterBase:
    """ A base class for plotting. """

    def __init__(
            self,
            title: str | None = None,
            labels: List[str] | None = None,
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
        if self.title is not None:
            ax.set_title(self.title)
        if self.labels is not None:
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

    def _set_labels(self):
        self.ax.set_xlabel(self.labels[0])
        self.ax.set_ylabel(self.labels[1])
        if self.cbar is not None:
            self.cbar.set_label(self.labels[2])

    def save_figure(self, filename: str | Path | None = None):
        if filename is None:
            filename = self.title.replace(" ", "_").lower() + ".png"
        self.fig.savefig(fname=Path.cwd() / filename, dpi=600)
        return self

    def close_figure(self):
        plt.close(self.fig)

    def show_figure(self):
        plt.show(self.fig)
