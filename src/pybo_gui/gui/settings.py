"""GUI-wide settings, held for the session.

One object is built by the app and handed to every tab, so a choice made in Settings is
visible to the tab that launches the plots. Nothing is written to disk: the analysis
scripts take these as flags, so a scripted run stays entirely described by its command
line rather than by what the GUI last remembered.
"""
from dataclasses import dataclass

from pybo.plotters.style import DEFAULT_STYLE


@dataclass
class Settings:
    """What the tabs share. Fields map one-to-one onto analysis-script flags."""

    plot_style: str = DEFAULT_STYLE

    def plot_args(self) -> list:
        """The flags every launched plot should carry."""
        return ["--plot-style", self.plot_style]