"""Reusable widget builders for the GUI tabs."""
from PySide6.QtCore import Qt, QCoreApplication, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLineEdit, QPushButton, QScrollArea, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)


def _variable_tree(headers, height: int) -> QTreeWidget:
    """A flat, header-labelled table of one variable per row.

    The same shape the group map is shown in, and for the same reason: a bare column of
    numbers needs a heading to say which number it is, and the widths belong to the
    contents rather than to a pixel count picked by hand.
    """
    tree = QTreeWidget()
    tree.setColumnCount(len(headers))
    tree.setHeaderLabels(headers)
    tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    tree.header().setStretchLastSection(True)
    # A flat list, not a hierarchy: nothing nests, so the expand arrows would be a column
    # of blanks. Selection means nothing here either - the rows are not a choice.
    tree.setRootIsDecorated(False)
    tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    tree.setFixedHeight(height)
    return tree

# Operator symbols shown in the constraint rows, mapped to what the expression parser
# in pybo_gui.modules.bayesian_campaign_analysis.constraints understands.
_OPS = ["<=", "<", ">=", ">", "=="]


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


def make_parameters_widget(on_change=None):
    """A 'Parameters' box listing each parameter with the rig's resolution for it.

    The resolution is the smallest step the rig can actually distinguish, in the
    parameter's own units, and it is what decides whether two records are the same
    setting. It is per parameter because one number cannot serve volts and nanoseconds
    at once. Blank means unknown, and the caller falls back to its own default.

    A row reads `label | bounds | resolution | unit`. Bounds and unit come from the
    objective and are shown, not edited: they say what the number in the box means, so a
    resolution is typed against the range and the unit it belongs to rather than blind.
    The objective supplies the resolution too; typing over one is an override for this
    session, the same way a sense stays editable after an objective is loaded.

    Returns (box, collect, set_keys):
      - collect() -> {label: resolution} for rows holding a positive number
      - set_keys(keys, specs) rebuilds the rows, keeping whatever was typed for a label
        that survives, so reloading an objective does not discard an override. `specs`
        maps a label to its {"bounds", "unit", "resolution"}, all optional.

    `on_change` is called when an edit is committed, not on every keystroke: a resolution
    is only meaningful once it is finished being typed, and regrouping on each digit would
    fire on the `0.` of `0.1`.
    """
    box = QGroupBox("Parameters")
    outer = QVBoxLayout(box)
    tree = _variable_tree(["Parameter", "Bounds", "Resolution", "Unit"], 130)
    outer.addWidget(tree)

    rows: dict = {}

    def set_keys(keys, specs: dict | None = None) -> None:
        # Read the overrides before clearing: clear() destroys the embedded editors with
        # their items, so anything typed has to be lifted out first.
        previous = {key: edit.text() for key, edit in rows.items()}
        tree.clear()
        rows.clear()
        for key in keys:
            spec = (specs or {}).get(key) or {}
            if previous.get(key):
                # A blank box is not an override - it is what a row looks like before
                # any objective has spoken for it, and letting it win would mean the
                # objective's own resolution can never reach the box once the map has
                # been built once.
                text = previous[key]
            else:
                value = spec.get("resolution")
                text = "" if value is None else f"{value:g}"
            bounds = spec.get("bounds")

            # Nothing to show before an objective is loaded: the map names the parameters
            # but carries no bounds or units, so those cells stay empty rather than absent.
            item = QTreeWidgetItem([key,
                                    f"{bounds[0]:g} – {bounds[1]:g}" if bounds else "",
                                    "", spec.get("unit") or ""])
            tree.addTopLevelItem(item)

            # A real editor in the cell rather than an editable item: that keeps the
            # placeholder, the tooltip and the commit-on-finish behaviour, none of which a
            # plain editable item gives.
            edit = QLineEdit(text)
            edit.setPlaceholderText("resolution")
            edit.setToolTip(f"Smallest step the rig resolves for {key}, in the unit shown. "
                            f"Two records closer than this are the same setting. Blank "
                            f"means unknown, and the parameter is compared as measured.")
            if on_change is not None:
                edit.editingFinished.connect(on_change)
            tree.setItemWidget(item, 2, edit)
            rows[key] = edit

    def collect() -> dict:
        found = {}
        for key, edit in rows.items():
            try:
                value = float(edit.text().strip())
            except ValueError:
                # Blank is the normal way to say "unknown"; anything unparseable means the
                # same thing here rather than killing the click that read it.
                continue
            if value > 0:
                found[key] = value
        return found

    return box, collect, set_keys


def make_objectives_widget():
    """An 'Objectives' box listing what the campaign is actually optimising.

    Read-only, and deliberately so: which way an objective runs, what it is measured in
    and where its hypervolume is measured from are the objective's own statements, not
    settings. The senses next to the axes on the plot tab are seeded from these and stay
    editable there, for a figure that wants to read an axis the other way round; this box
    is what they were seeded from.

    A bound is shown where the objective declares one - most do not, an objective being a
    measured outcome rather than something dialled in - and the reference point is what a
    hypervolume is measured against.

    Empty until an objective is loaded: the step records name their columns but say
    nothing about direction, units or reference points.

    Returns (box, set_keys), where set_keys(specs) takes the objectives as
    [{"label", "to_minimize", "unit", "bounds", "ref_point"}], all but the label optional.
    """
    box = QGroupBox("Objectives")
    outer = QVBoxLayout(box)
    tree = _variable_tree(["Objective", "Direction", "Bounds", "Ref. point", "Unit"], 110)
    outer.addWidget(tree)

    def set_keys(specs) -> None:
        tree.clear()
        if not specs:
            # A row rather than a hidden table: an empty grid with headers reads as
            # "there are none", which is a different claim from "none are knowable yet".
            note = QTreeWidgetItem(["No objective loaded — direction, units and reference "
                                    "point are only nameable from one."])
            note.setFirstColumnSpanned(True)
            note.setDisabled(True)
            tree.addTopLevelItem(note)
            return
        for spec in specs:
            bounds = spec.get("bounds")
            ref = spec.get("ref_point")
            tree.addTopLevelItem(QTreeWidgetItem([
                spec.get("label") or "",
                "minimize" if spec.get("to_minimize") else "maximize",
                f"{bounds[0]:g} – {bounds[1]:g}" if bounds else "",
                f"{ref:g}" if isinstance(ref, (int, float)) else "",
                spec.get("unit") or "",
            ]))

    set_keys([])
    return box, set_keys


def make_trackers_widget():
    """A 'Trackers' box listing what the objective measures without optimising it.

    Read-only: a tracker carries no sense, no reference point and nothing to tune, so the
    box says what is being recorded alongside the objectives - and, when a tracker turns
    out to be what a constraint is really derived from, which quantity that is.

    Empty until an objective is loaded: the step records name their columns but never say
    which of them is a tracker.

    Returns (box, set_keys), where set_keys(specs) takes the trackers as
    [{"label", "unit", "bounds"}], unit and bounds optional.
    """
    box = QGroupBox("Trackers")
    outer = QVBoxLayout(box)
    tree = _variable_tree(["Tracker", "Bounds", "Unit"], 110)
    outer.addWidget(tree)

    def set_keys(specs) -> None:
        tree.clear()
        if not specs:
            # A row rather than a hidden table: an empty grid with headers reads as
            # "there are none", which is a different claim from "none are knowable yet".
            note = QTreeWidgetItem(["No objective loaded — a tracker is only "
                                    "nameable from one."])
            note.setFirstColumnSpanned(True)
            note.setDisabled(True)
            tree.addTopLevelItem(note)
            return
        for spec in specs:
            bounds = spec.get("bounds")
            tree.addTopLevelItem(QTreeWidgetItem([
                spec.get("label") or "",
                f"{bounds[0]:g} – {bounds[1]:g}" if bounds else "",
                spec.get("unit") or "",
            ]))

    set_keys([])
    return box, set_keys


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
