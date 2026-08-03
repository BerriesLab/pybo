"""Reusable widget builders for the GUI tabs."""
from PySide6.QtCore import Qt, QCoreApplication, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

# Operator symbols shown in the constraint rows, mapped to what the expression parser
# in pybo_gui.modules.bayesian_campaign_analysis.constraints understands.
_OPS = ["<=", "<", ">=", ">", "=="]


def set_text_async(label: QLabel, text: str) -> None:
    """Set a label's text safely from any thread, by marshalling onto the main one."""
    app = QCoreApplication.instance()
    QTimer.singleShot(0, app, lambda: label.setText(text))


def set_enabled_async(widget: QWidget, enabled: bool) -> None:
    """Set a widget's enabled state safely from any thread."""
    app = QCoreApplication.instance()
    QTimer.singleShot(0, app, lambda: widget.setEnabled(enabled))


def repopulate(combo: QComboBox, items, blank_first: bool = False) -> None:
    """Replace a combo's items, keeping the current selection if it survives."""
    current = combo.currentText()
    combo.clear()
    combo.addItems(([""] if blank_first else []) + list(items))
    if current in items:
        combo.setCurrentText(current)
    elif combo.count():
        combo.setCurrentIndex(0)


def set_combo_default(combo: QComboBox, preferred: str) -> None:
    """Leave the current selection if it is still valid, else prefer `preferred`."""
    items = [combo.itemText(i) for i in range(combo.count())]
    if combo.currentText() in items:
        return
    if preferred in items:
        combo.setCurrentText(preferred)
    elif items:
        combo.setCurrentIndex(0)


def make_sense_toggle() -> QPushButton:
    """A checkable Min/Max button for an objective's optimisation sense.

    Unchecked = minimise, checked = maximise; the text follows the state. Read
    `btn.isChecked()` for the maximise flag.
    """
    btn = QPushButton("Min")
    btn.setCheckable(True)
    btn.setFixedWidth(52)
    btn.setToolTip("Click to toggle between minimise and maximise")
    btn.toggled.connect(lambda on: btn.setText("Max" if on else "Min"))
    return btn


def bind_label_entry(combo: QComboBox, entry: QLineEdit) -> None:
    """Mirror the combo's text into the entry, so an axis label defaults to its key."""
    entry.setText(combo.currentText())
    combo.currentTextChanged.connect(entry.setText)


def make_constraints_widget():
    """A 'Constraints' box with add/removable expression rows.

    Each row is an editable key combo, an operator and a right-hand value, collecting to
    one expression string. The left side is editable so richer expressions (sums,
    ``abs(...)``, powers) can be typed where a bare key is not enough.

    Returns (box, collect, set_keys):
      - collect() -> expression strings for rows with both sides filled in
      - set_keys(keys) repopulates every row's key combo
    """
    box = QGroupBox("Constraints")
    outer = QVBoxLayout(box)
    host = QWidget()
    rows_layout = QVBoxLayout(host)
    rows_layout.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(host)

    rows = []
    keys = [""]

    def _add_row() -> None:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        lhs = QComboBox()
        lhs.setEditable(True)
        lhs.setFixedWidth(240)
        lhs.addItems(keys)
        lhs.setCurrentText("")
        op = QComboBox()
        op.setFixedWidth(64)
        op.addItems(_OPS)
        rhs = QLineEdit()
        rhs.setFixedWidth(140)
        rhs.setPlaceholderText("value or expr")
        remove = QPushButton("-")
        remove.setFixedWidth(28)
        for widget in (lhs, op, rhs, remove):
            h.addWidget(widget)
        h.addStretch()

        entry = (row, lhs, op, rhs)
        rows.append(entry)
        rows_layout.addWidget(row)

        def _remove() -> None:
            rows.remove(entry)
            rows_layout.removeWidget(row)
            row.deleteLater()

        remove.clicked.connect(_remove)

    add = QPushButton("+ Add constraint")
    add.clicked.connect(_add_row)
    outer.addWidget(add)

    def set_keys(new_keys) -> None:
        keys.clear()
        keys.append("")
        keys.extend(new_keys)
        for _row, lhs, _op, _rhs in rows:
            current = lhs.currentText()
            lhs.clear()
            lhs.addItems(keys)
            lhs.setCurrentText(current)

    def collect() -> list:
        specs = []
        for _row, lhs, op, rhs in rows:
            left, right = lhs.currentText().strip(), rhs.text().strip()
            if left and right:
                specs.append(f"{left} {op.currentText()} {right}")
        return specs

    return box, collect, set_keys


def make_objective_checklist():
    """A scrollable checklist of labels with a sense toggle each, for N-D hypervolume.

    Returns (box, collect, set_keys):
      - collect() -> [(label, maximize)] for the ticked rows
      - set_keys(keys) rebuilds the rows, preserving ticks and senses where the label
        survives, so reloading an objective does not silently reset the selection
    """
    box = QGroupBox("Objectives (N-D hypervolume)")
    outer = QVBoxLayout(box)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFixedHeight(130)
    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    inner_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    scroll.setWidget(inner)
    outer.addWidget(scroll)

    rows: dict = {}

    def set_keys(keys, senses: dict | None = None) -> None:
        previous = {k: (cb.isChecked(), tog.isChecked()) for k, (_r, cb, tog) in rows.items()}
        for row, _cb, _tog in rows.values():
            inner_layout.removeWidget(row)
            row.deleteLater()
        rows.clear()
        for key in keys:
            was_checked, was_max = previous.get(key, (True, False))
            if senses is not None and key not in previous:
                was_max = not senses.get(key, True)
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            check = QCheckBox(key)
            check.setChecked(was_checked)
            toggle = make_sense_toggle()
            toggle.setChecked(was_max)
            h.addWidget(check)
            h.addStretch()
            h.addWidget(toggle)
            inner_layout.addWidget(row)
            rows[key] = (row, check, toggle)

    def collect() -> list:
        return [(key, tog.isChecked()) for key, (_r, cb, tog) in rows.items() if cb.isChecked()]

    return box, collect, set_keys
