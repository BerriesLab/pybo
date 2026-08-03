"""The campaign-analysis tab.

Turns a step selection plus a problem definition into a pybo_gui.modules.bayesian_campaign_analysis command line.
The tab draws nothing itself: every button launches the matching analysis module, which
stays runnable from a terminal with the same flags.

Load objective is what fills the tab in - the step records name their values but carry no
problem definition, so the axis keys, the min/max senses and the hypervolume reference
point all come from the run's objective.py. Every sense stays editable afterwards: the
objective is the default, not the last word.
"""
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QRadioButton, QVBoxLayout, QWidget,
)

from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective, problem_definition
from pybo_gui.modules.bayesian_campaign_analysis.steps import find_steps, group_labels
from pybo_gui.gui.launchers import launch_analysis, watch
from pybo_gui.gui.widgets import (
    bind_label_entry, make_constraints_widget, make_objective_checklist, make_sense_toggle,
    repopulate, set_text_async,
)

DIMENSIONS = ("1D", "2D", "3D", "N-D")


def _row(*widgets) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    return row


def _axis_row(prefix: str, combo: QComboBox, entry: QLineEdit):
    """One axis: prefix label + key combo + display label + Min/Max toggle."""
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    lead = QLabel(prefix)
    lead.setFixedWidth(78)
    combo.setFixedWidth(200)
    entry.setMinimumWidth(180)
    toggle = make_sense_toggle()
    for widget in (lead, combo, entry, toggle):
        h.addWidget(widget)
    h.addStretch()
    return row, lead, toggle


def build(step_list, settings) -> QWidget:
    """Construct the tab.

    `step_list` is the selector window it reads the selection from; `settings` is the
    shared object the Settings tab writes, read at launch so a style change applies to
    the next plot without rebuilding anything.
    """
    page = QWidget()
    layout = QVBoxLayout(page)
    state: dict = {"problem": None}

    # ---- Objective -----------------------------------------------------------
    obj_box = QGroupBox("Objective")
    obj_layout = QVBoxLayout(obj_box)
    obj_edit = QLineEdit()
    obj_edit.setPlaceholderText("path to the run's objective.py")
    obj_status = QLabel("No objective loaded — keys will come from the selected steps.")
    obj_status.setStyleSheet("color: grey;")
    browse = QPushButton("Browse")
    load_btn = QPushButton("Load objective")
    obj_layout.addWidget(_row(obj_edit, browse, load_btn))
    obj_layout.addWidget(obj_status)
    layout.addWidget(obj_box)

    # ---- Dimensionality ------------------------------------------------------
    dim_group = QButtonGroup(page)
    dim_buttons = {}
    dim_row = QWidget()
    dim_layout = QHBoxLayout(dim_row)
    dim_layout.setContentsMargins(0, 0, 0, 0)
    lead = QLabel("Dimensions:")
    lead.setFixedWidth(78)
    dim_layout.addWidget(lead)
    for name in DIMENSIONS:
        button = QRadioButton(name)
        dim_group.addButton(button)
        dim_buttons[name] = button
        dim_layout.addWidget(button)
    dim_layout.addStretch()
    dim_buttons["1D"].setEnabled(False)
    dim_buttons["1D"].setToolTip("Not yet implemented")
    dim_buttons["2D"].setChecked(True)
    layout.addWidget(dim_row)

    def _dimension() -> str:
        button = dim_group.checkedButton()
        return button.text() if button else "2D"

    # ---- Axes ----------------------------------------------------------------
    axes_box = QGroupBox("Objectives")
    axes_layout = QVBoxLayout(axes_box)
    x_combo, y_combo, z_combo = QComboBox(), QComboBox(), QComboBox()
    x_entry, y_entry, z_entry = QLineEdit(), QLineEdit(), QLineEdit()
    x_row, _x_lead, x_sense = _axis_row("x:", x_combo, x_entry)
    y_row, _y_lead, y_sense = _axis_row("y:", y_combo, y_entry)
    z_row, z_lead, z_sense = _axis_row("z (colour):", z_combo, z_entry)
    for row in (x_row, y_row, z_row):
        axes_layout.addWidget(row)
    for combo, entry in ((x_combo, x_entry), (y_combo, y_entry), (z_combo, z_entry)):
        bind_label_entry(combo, entry)
    layout.addWidget(axes_box)

    nd_box, nd_collect, nd_set_keys = make_objective_checklist()
    layout.addWidget(nd_box)

    con_box, con_collect, con_set_keys = make_constraints_widget()
    layout.addWidget(con_box)

    # ---- Plots ---------------------------------------------------------------
    plot_box = QGroupBox("Plots")
    plot_layout = QVBoxLayout(plot_box)
    btn_pareto = QPushButton("Plot Pareto")
    btn_hv = QPushButton("Plot hypervolume")
    btn_hvi = QPushButton("Plot HV improvement")
    btn_refresh = QPushButton("Refresh keys")
    cb_per_run = QCheckBox("One series per run")
    status = QLabel("")
    status.setStyleSheet("color: grey;")
    plot_layout.addWidget(_row(btn_pareto, btn_hv, btn_hvi, btn_refresh, cb_per_run))
    plot_layout.addWidget(status)
    layout.addWidget(plot_box)
    layout.addStretch()

    # ---- Key discovery -------------------------------------------------------

    def _refresh_keys() -> None:
        """Repopulate every key list.

        Objective labels when one is loaded, else whatever the selected steps happen to
        carry - so the tab is usable before an objective is picked, just without senses.
        """
        problem = state["problem"]
        if problem is not None:
            objectives = [o["label"] for o in problem["objectives"]]
            everything = objectives + [p["label"] for p in problem["parameters"]] \
                         + [c["label"] for c in problem["constraints"]] \
                         + [t["label"] for t in problem["trackers"]]
            senses = problem["minimized"]
        else:
            groups = group_labels(find_steps(step_list.checked_paths or [step_list.root])) \
                if (step_list.checked_paths or step_list.root) else {}
            objectives = groups.get("objectives", [])
            everything = [label for labels in groups.values() for label in labels]
            senses = {}

        repopulate(x_combo, objectives)
        repopulate(y_combo, objectives)
        repopulate(z_combo, everything, blank_first=True)
        if len(objectives) > 1:
            y_combo.setCurrentIndex(1)
        for combo, toggle in ((x_combo, x_sense), (y_combo, y_sense), (z_combo, z_sense)):
            key = combo.currentText()
            if key in senses:
                toggle.setChecked(not senses[key])
        nd_set_keys(objectives, senses)
        con_set_keys(everything)

    def _load_objective(path: str) -> None:
        try:
            problem = problem_definition(load_objective(path))
        except Exception as exc:  # noqa: BLE001 - a bad path must not kill the tab
            state["problem"] = None
            obj_status.setText(f"Could not load: {exc}")
            return
        state["problem"] = problem
        _refresh_keys()
        senses = ", ".join(f"{o['label']} ({'min' if o['to_minimize'] else 'max'})"
                           for o in problem["objectives"])
        obj_status.setText(f"{len(problem['objectives'])} objectives: {senses}. "
                           f"ref_point={problem['ref_point']}")

    def _browse_objective() -> None:
        chosen, _ = QFileDialog.getOpenFileName(page, "Choose an objective", obj_edit.text(),
                                                "Python (*.py)")
        if chosen:
            obj_edit.setText(chosen)
            _load_objective(chosen)

    browse.clicked.connect(_browse_objective)
    load_btn.clicked.connect(lambda: _load_objective(obj_edit.text()))
    btn_refresh.clicked.connect(_refresh_keys)

    # ---- Dimensionality wiring ----------------------------------------------

    def _on_dimension_change() -> None:
        dim = _dimension()
        is_nd = dim == "N-D"
        is_3d = dim == "3D"
        # N-D has no scatter to draw beyond three axes, so the axis frame gives way to
        # the checklist and the Pareto button goes with it.
        axes_box.setVisible(not is_nd)
        nd_box.setVisible(is_nd)
        z_row.setEnabled(not is_nd)
        # z is a real objective only in 3D; in 2D it colours the points, so its sense
        # means nothing.
        z_sense.setEnabled(is_3d)
        z_lead.setText("z:" if is_3d else "z (colour):")
        btn_pareto.setEnabled(not is_nd)
        cb_per_run.setEnabled(dim == "2D")

    dim_group.buttonToggled.connect(lambda _b, checked: checked and _on_dimension_change())

    # ---- Launching -----------------------------------------------------------

    def _shared_args() -> list | None:
        steps = step_list.checked_paths
        if not steps:
            status.setText("Select at least one step in the Steps window.")
            return None
        if state["problem"] is None:
            status.setText("Load an objective first: the senses and reference point come "
                           "from it.")
            return None
        args = ["--objective", obj_edit.text()]
        for step in steps:
            args += ["--step", step]
        for spec in con_collect():
            args += ["--constraint", spec]
        # Read at launch, not at build: the Settings tab may have changed it since.
        args += settings.plot_args()
        return args

    def _sense_args(*pairs) -> list:
        """--maximize/--minimize for each (key, toggle), stated explicitly.

        The scripts default to the objective, but the tab lets a sense be flipped, so it
        always says which it means rather than relying on the default.
        """
        args = []
        for key, toggle in pairs:
            if key:
                args += ["--maximize" if toggle.isChecked() else "--minimize", key]
        return args

    def _launch(module: str, *extra) -> None:
        args = _shared_args()
        if args is None:
            return
        proc = launch_analysis(module, *args, *extra)
        watch([proc],
              on_start=lambda: set_text_async(status, f"Running {module}..."),
              on_done=lambda: set_text_async(status, "Done."))

    def _plot_pareto() -> None:
        x, y, z = x_combo.currentText(), y_combo.currentText(), z_combo.currentText()
        if not x or not y:
            status.setText("Choose an x and a y objective.")
            return
        if _dimension() == "3D":
            if not z:
                status.setText("3D needs a z objective.")
                return
            extra = ["--x", x, "--y", y, "--z", z,
                     "--xlabel", x_entry.text(), "--ylabel", y_entry.text(),
                     "--zlabel", z_entry.text()]
            extra += _sense_args((x, x_sense), (y, y_sense), (z, z_sense))
            _launch("pybo_gui.modules.bayesian_campaign_analysis.campaign_pareto_3d", *extra)
            return
        extra = ["--x", x, "--y", y, "--xlabel", x_entry.text(), "--ylabel", y_entry.text()]
        if z:
            extra += ["--z", z, "--zlabel", z_entry.text()]
        extra += _sense_args((x, x_sense), (y, y_sense))
        if cb_per_run.isChecked():
            extra.append("--per-run")
        _launch("pybo_gui.modules.bayesian_campaign_analysis.campaign_pareto", *extra)

    def _plot_hypervolume(improvement: bool) -> None:
        extra = ["--improvement"] if improvement else []
        if _dimension() == "N-D":
            chosen = nd_collect()
            if len(chosen) < 2:
                status.setText("Tick at least two objectives for an N-D hypervolume.")
                return
            for key, _is_max in chosen:
                extra += ["--objective-label", key]
            for key, is_max in chosen:
                extra += ["--maximize" if is_max else "--minimize", key]
        else:
            x, y, z = x_combo.currentText(), y_combo.currentText(), z_combo.currentText()
            if not x or not y:
                status.setText("Choose an x and a y objective.")
                return
            pairs = [(x, x_sense), (y, y_sense)]
            extra += ["--objective-label", x, "--objective-label", y]
            # The 2D colour axis is not an objective, so it joins only in 3D.
            if _dimension() == "3D" and z:
                extra += ["--objective-label", z]
                pairs.append((z, z_sense))
            extra += _sense_args(*pairs)
        _launch("pybo_gui.modules.bayesian_campaign_analysis.campaign_hypervolume", *extra)

    btn_pareto.clicked.connect(_plot_pareto)
    btn_hv.clicked.connect(lambda: _plot_hypervolume(False))
    btn_hvi.clicked.connect(lambda: _plot_hypervolume(True))

    # An objective usually sits with the tutorial that produced the data, so offer the
    # first one found under the selected root rather than leaving the field blank.
    if step_list.root:
        guess = next(Path(step_list.root).glob("**/objective.py"), None)
        if guess is not None:
            obj_edit.setText(str(guess))

    _on_dimension_change()
    return page
