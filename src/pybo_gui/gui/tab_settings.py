"""The settings tab.

Holds what applies to every plot rather than to one of them. A change here takes effect
on the next plot launched, since each plot is a fresh process that receives the settings
as flags.
"""
from PySide6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from pybo.plotters.style import list_styles, resolve


def _describe(style: str) -> str:
    """One line on what picking this style actually changes.

    Resolving here only touches this process's copy of the style module - the plots run
    as subprocesses that resolve it again from the flag - so it is safe to call just to
    read the numbers back out.
    """
    try:
        cfg = resolve(style)
    except Exception as exc:  # noqa: BLE001 - a broken style file must not kill the tab
        return f"Could not read that style: {exc}"
    width, height = cfg["figsize"]["convergence"]
    return (f"Column width {width} in, so a convergence plot is {width} x {height} in "
            f"at {cfg['dpi']} dpi.")


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
    combo.addItems(list_styles())
    combo.setCurrentText(settings.plot_style)
    row_layout.addWidget(label)
    row_layout.addWidget(combo)
    row_layout.addStretch()
    box_layout.addWidget(row)

    description = QLabel(_describe(settings.plot_style))
    description.setStyleSheet("color: grey;")
    box_layout.addWidget(description)

    note = QLabel("Applies to the next plot launched; open windows keep the style they "
                  "were drawn with.")
    note.setStyleSheet("color: grey;")
    note.setWordWrap(True)
    box_layout.addWidget(note)

    def _on_change(style: str) -> None:
        settings.plot_style = style
        description.setText(_describe(style))

    combo.currentTextChanged.connect(_on_change)

    layout.addWidget(box)
    layout.addStretch()
    return page