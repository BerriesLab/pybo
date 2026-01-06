from pathlib import Path
from typing import List, Tuple
import matplotlib

matplotlib.use("Agg")  # A pure renderer backend, not compatible with plt.show().
print(f"Matplotlib backend: {matplotlib.get_backend()}")
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
        self.fig, self.ax = plt.subplots(1, 1, figsize=self.figsize, dpi=600)
        self.cbar: plt.Colorbar | None = None
        self._initialize_figure()
        self._set_labels()
        self._set_x_lims()
        self._set_y_lims()
        self.legend_elements = []

    def _initialize_figure(self):
        if self.title is not None:
            self.ax.set_title(self.title)

    def _set_x_lims(self):
        if self.lims is not None and self.lims[0] is not None:
            self.ax.set_xlim(self.lims[0][0], self.lims[0][1])
        else:
            self.ax.autoscale(enable=True, axis='x')

    def _set_y_lims(self):
        if self.lims is not None and self.lims[1] is not None:
            self.ax.set_ylim(self.lims[1][0], self.lims[1][1])
        else:
            self.ax.autoscale(enable=True, axis='y')

    def _set_labels(self):
        if self.labels is not None:
            self.ax.set_xlabel(self.labels[0])
            self.ax.set_ylabel(self.labels[1])
        if self.cbar is not None:
            self.cbar.set_label(self.labels[2])

    def save_figure(self, filename: str | Path | None = None):
        if filename is None:
            filename = self.title.replace(" ", "_").lower() + ".png"

        path = Path(filename)
        save_path = Path.cwd() / path

        if save_path.exists():
            stem = path.stem
            suffix = path.suffix
            i = 1
            while True:
                candidate = save_path.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    save_path = candidate
                    break
                i += 1

        self.fig.savefig(fname=save_path, dpi=600)
        return self

    def close_figure(self):
        plt.close(self.fig)

    def show_figure(self):
        plt.show(self.fig)
