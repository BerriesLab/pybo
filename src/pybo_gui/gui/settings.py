"""GUI-wide settings shared by the tabs.

The figure style is not held in memory here: a plot runs as its own process and resolves
the style itself at startup, from configs/figure_settings_app/state.json. So the Settings
tab writes that file and this object only proxies the store - which also means a style
chosen in the GUI survives a restart, and applies to a script run straight from a
terminal.
"""
from dataclasses import dataclass

from pybo_gui.configs.figure_settings import store


@dataclass
class Settings:
    """What the tabs share. Style access is delegated to the figure-settings store."""

    @property
    def plot_style(self):
        """The active publisher style, or None when the campaign uses the bare defaults."""
        return store.get_active()["publisher"]

    @plot_style.setter
    def plot_style(self, name) -> None:
        store.set_active_publisher(name or None)
