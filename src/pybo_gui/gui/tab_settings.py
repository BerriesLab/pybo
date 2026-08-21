"""The settings tab.

Holds what applies to every plot rather than to one of them. A change here is written to
configs/figure_settings_app/state.json and takes effect on the next plot launched, since
each plot is a fresh process that resolves the style from that file.
"""
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

from pybo_gui.configs import workspace as ws_store
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

    # ---- Workspace -----------------------------------------------------------
    ws_box = QGroupBox("Workspace")
    ws_layout = QVBoxLayout(ws_box)

    ws_row = QWidget()
    ws_row_layout = QHBoxLayout(ws_row)
    ws_row_layout.setContentsMargins(0, 0, 0, 0)
    ws_label = QLabel("Folder:")
    ws_label.setFixedWidth(90)
    ws_path = QLineEdit()
    ws_path.setReadOnly(True)
    ws_path.setMinimumWidth(320)
    browse = QPushButton("Browse…")
    clear = QPushButton("Use a temporary folder")
    ws_row_layout.addWidget(ws_label)
    ws_row_layout.addWidget(ws_path)
    ws_row_layout.addWidget(browse)
    ws_row_layout.addWidget(clear)
    ws_row_layout.addStretch()
    ws_layout.addWidget(ws_row)

    usage_row = QWidget()
    usage_layout = QHBoxLayout(usage_row)
    usage_layout.setContentsMargins(0, 0, 0, 0)
    usage_label = QLabel()
    usage_label.setStyleSheet("color: grey;")
    clear_cache = QPushButton("Clear cache")
    spacer = QLabel()
    spacer.setFixedWidth(90)   # allinea sotto "Folder:"
    usage_layout.addWidget(spacer)
    usage_layout.addWidget(usage_label)
    usage_layout.addWidget(clear_cache)
    usage_layout.addStretch()
    ws_layout.addWidget(usage_row)

    ws_note = QLabel(
        "Where each session writes its experiment_map.json and group_map.json, one "
        "directory per session. Left empty they go to a temporary folder and are gone "
        "when the session ends, so the map is rebuilt from scratch next time — which on "
        "a large campaign is minutes. A session already open keeps the folder it started "
        "with. Clearing the cache only deletes the built maps, which are rebuilt "
        "from the records on demand; the session folders are left alone.")
    ws_note.setStyleSheet("color: grey;")
    ws_note.setWordWrap(True)
    ws_layout.addWidget(ws_note)

    def _mb(value: int) -> str:
        return f"{value / (1024 * 1024):.1f} MB"

    def _show_workspace() -> None:
        current = settings.workspace
        ws_path.setText(str(current) if current else "")
        ws_path.setPlaceholderText("temporary — not kept between sessions")
        usage = ws_store.usage()
        if usage is None:
            usage_label.setText("No workspace, so nothing is kept.")
            clear_cache.setEnabled(False)
            return
        # The two are split because only one of them is safe to delete.
        usage_label.setText(
            f"{_mb(usage['total'])} in use — {_mb(usage['cache'])} of cached maps "
            f"({usage['entries']} selection{'' if usage['entries'] == 1 else 's'}), "
            f"{_mb(usage['sessions'])} of session folders.")
        clear_cache.setEnabled(usage["cache"] > 0)

    def _clear_cache() -> None:
        freed = ws_store.clear_cache()
        _show_workspace()
        usage_label.setText(usage_label.text() + f"  Freed {_mb(freed)}.")

    def _browse() -> None:
        chosen = QFileDialog.getExistingDirectory(page, "Choose a workspace folder",
                                                  str(settings.workspace or ""))
        if chosen:
            settings.workspace = chosen
            _show_workspace()

    browse.clicked.connect(_browse)
    clear_cache.clicked.connect(_clear_cache)
    clear.clicked.connect(lambda: (setattr(settings, "workspace", None), _show_workspace()))
    _show_workspace()

    layout.addWidget(ws_box)
    layout.addStretch()
    return page
