"""The settings tab.

Holds what applies to every plot rather than to one of them. A change here is written to
configs/figure_settings_app/state.json and takes effect on the next plot launched, since
each plot is a fresh process that resolves the style from that file.
"""
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from pybo_gui.configs.figure_settings import store

_NONE = "(package defaults)"


def _describe(style) -> str:
    """One line on what picking this style actually changes.

    Reads the style's YAML rather than resolving it, so the GUI never imports the
    matplotlib-side assembler just to describe a choice.
    """
    if not style:
        return "No publisher style: the package and app defaults are used as they are."
    try:
        data = store.load_style("publisher", style)
    except (OSError, ValueError) as exc:  # noqa: BLE001 - a bad file must not kill the tab
        return f"Could not read that style: {exc}"
    parts = []
    if data.get("description"):
        parts.append(str(data["description"]))
    if data.get("column_width_in"):
        parts.append(f"column width {data['column_width_in']} in")
    if data.get("dpi"):
        parts.append(f"{data['dpi']} dpi")
    if data.get("rcparams"):
        parts.append(f"{len(data['rcparams'])} rcParams")
    return "; ".join(parts) if parts else "A partial override of the defaults."


def build(settings) -> QWidget:
    """Construct the tab. `settings` is the shared object the other tabs read."""
    page = QWidget()
    layout = QVBoxLayout(page)

    box = QGroupBox("Figures")
    box_layout = QVBoxLayout(box)

    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel("Plot style:")
    label.setFixedWidth(90)
    combo = QComboBox()
    combo.setFixedWidth(220)
    combo.addItems([_NONE] + store.list_publisher_styles())
    combo.setCurrentText(settings.plot_style or _NONE)
    row_layout.addWidget(label)
    row_layout.addWidget(combo)
    row_layout.addStretch()
    box_layout.addWidget(row)

    description = QLabel(_describe(settings.plot_style))
    description.setStyleSheet("color: grey;")
    description.setWordWrap(True)
    box_layout.addWidget(description)

    note = QLabel("Written to state.json, so it applies to the next plot launched and "
                  "survives a restart. Open windows keep the style they were drawn with.")
    note.setStyleSheet("color: grey;")
    note.setWordWrap(True)
    box_layout.addWidget(note)

    def _on_change(text: str) -> None:
        style = None if text == _NONE else text
        settings.plot_style = style
        description.setText(_describe(style))

    combo.currentTextChanged.connect(_on_change)

    layout.addWidget(box)
    layout.addStretch()
    return page
