"""The campaign-analysis tab.

Turns a step selection plus a problem definition into a pybo_gui.modules.bayesian_campaign_analysis command line.
The tab draws nothing itself: every button launches the matching analysis module, which
stays runnable from a terminal with the same flags.

Load objective is what fills the tab in - the step records name their values but carry no
problem definition, so the axis keys, the min/max senses and the hypervolume reference
point all come from the run's objective.py. Every sense stays editable afterwards: the
objective is the default, not the last word.
"""
import json
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
    QDoubleSpinBox, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from pybo_gui.configs import settings as configs_settings
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import build_map
from pybo_gui.modules.bayesian_campaign_analysis.build_group_map import (
    DEFAULT_DECIMALS, build_groups)
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective, problem_definition
from pybo_gui.gui.launchers import launch_analysis, watch
from pybo_gui.gui.widgets import (
    bind_label_entry, make_constraints_widget, make_objective_checklist, make_sense_toggle,
    repopulate, set_text_async,
)

OBJECTIVE_COUNTS = ("1", "2", "3", "4+")
MODULES = "pybo_gui.modules.bayesian_campaign_analysis"
_NO_OBJECTIVE = "No objective loaded — keys will come from the selected steps."


def _result_keys(exp_map: dict) -> list:
    """Every column in the experiment map, or [] when nothing is built yet.

    Parameters count as columns too: with no objective loaded this is the only source
    of keys, and leaving them out is what kept a parameter off the colour axis.

    The <label>_var partners are left out: they are the uncertainty of a column, read by
    the error bars, not a quantity to put on an axis of its own.
    """
    keys = []
    for entry in (exp_map or {}).get("experiments", []):
        for key in list(entry.get("results", {})) + list(entry.get("parameters", {})):
            if key not in keys and not key.endswith("_var"):
                keys.append(key)
    return sorted(keys)


def _parameter_keys(exp_map: dict) -> list:
    """Every parameter column in the experiment map, or [] when nothing is built yet.

    _result_keys deliberately merges parameters in with the results, since an axis will
    take either. The single-objective landscape draws the objective *over* its
    parameters, so there they have to be told apart.
    """
    keys = []
    for entry in (exp_map or {}).get("experiments", []):
        for key in entry.get("parameters", {}):
            if key not in keys:
                keys.append(key)
    return sorted(keys)


def _view_json_dialog(parent: QWidget, payload, title: str) -> None:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(700, 600)
    v = QVBoxLayout(dlg)
    txt = QPlainTextEdit()
    txt.setReadOnly(True)
    txt.setPlainText(json.dumps(payload, indent=2) if payload else
                     "Nothing built yet — click “Rebuild map now” first.")
    v.addWidget(txt)
    dlg.show()


def _view_group_map_dialog(parent: QWidget, groups, exp_map) -> None:
    dlg = QDialog(parent)
    dlg.setWindowTitle("Group map")
    dlg.resize(700, 600)
    v = QVBoxLayout(dlg)

    if not groups:
        v.addWidget(QLabel("Nothing built yet — click “Rebuild map now” first."))
        dlg.show()
        return

    gid_to_exps = {}
    for e in (exp_map or {}).get("experiments", []):
        gid = e.get("group_id")
        # experiment_id, where the rig listed a folder: a pybo observation has no
        # directory of its own, it is one row of a step record.
        gid_to_exps.setdefault(gid, []).append(e["experiment_id"])

    cols = list(groups[0].keys())
    splitter = QSplitter(Qt.Orientation.Vertical)

    tree = QTreeWidget()
    tree.setColumnCount(len(cols))
    tree.setHeaderLabels(cols)
    tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    for g in groups:
        item = QTreeWidgetItem([str(g.get(c, "")) for c in cols])
        item.setData(0, Qt.ItemDataRole.UserRole, g.get("group_id"))
        tree.addTopLevelItem(item)
    splitter.addWidget(tree)

    detail_box = QWidget()
    detail_layout = QVBoxLayout(detail_box)
    detail_layout.setContentsMargins(0, 0, 0, 0)
    detail_lbl = QLabel("Select a group to see its experiments.")
    detail_lbl.setStyleSheet("color: grey;")
    detail_layout.addWidget(detail_lbl)
    detail_txt = QPlainTextEdit()
    detail_txt.setReadOnly(True)
    detail_layout.addWidget(detail_txt)
    splitter.addWidget(detail_box)

    def _on_select():
        sel = tree.selectedItems()
        if not sel:
            return
        gid = sel[0].data(0, Qt.ItemDataRole.UserRole)
        exps = gid_to_exps.get(gid, [])
        detail_lbl.setText(f"Group {gid} — {len(exps)} experiment(s):")
        detail_lbl.setStyleSheet("")
        detail_txt.setPlainText("\n".join(exps))

    tree.itemSelectionChanged.connect(_on_select)

    v.addWidget(splitter)
    dlg.show()


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
    # "source" records where the map came from: a selection rebuild, or a file the
    # user loaded. A loaded map must survive the next plot click rather than being
    # silently rebuilt from a selection it has nothing to do with.
    state: dict = {"problem": None, "map": None, "groups": None, "source": None}
    # One scratch directory per session, so a rebuild never writes into the
    # user's data tree. Save map is the only thing that puts a map where they chose.
    _scratch = Path(tempfile.mkdtemp(prefix="pybo_campaign_"))

    # ---- Experiment map ------------------------------------------------------
    # First, as in the original tab: it defines the rebuild every other action calls.
    map_box = QGroupBox("Experiment map")
    map_layout = QVBoxLayout(map_box)
    btn_rebuild = QPushButton("Rebuild map now")
    btn_view_exp = QPushButton("View experiment map")
    btn_view_grp = QPushButton("View group map")
    btn_save_map = QPushButton("Save map")
    btn_load_map = QPushButton("Load map")
    map_status = QLabel("")
    map_status.setStyleSheet("color: grey;")
    # What counts as the same setting: parameters are compared rounded to this many
    # decimals. The default compares them as measured, which splits a setting whose
    # repeats were logged at the rig's rounded setpoint and at the raw proposal; lowering
    # it is what makes those one group. Negative values round to tens and upwards.
    decimals_spin = QSpinBox()
    decimals_spin.setRange(-3, 9)
    decimals_spin.setValue(DEFAULT_DECIMALS)
    decimals_spin.setToolTip("Decimals a parameter is rounded to before two observations "
                             "count as the same setting.")
    map_layout.addWidget(_row(btn_rebuild, btn_view_exp, btn_view_grp, btn_save_map,
                              btn_load_map))
    map_layout.addWidget(_row(QLabel("Grouping decimals:"), decimals_spin))
    map_layout.addWidget(map_status)
    layout.addWidget(map_box)

    # ---- Objective -----------------------------------------------------------
    obj_box = QGroupBox("Objective")
    obj_layout = QVBoxLayout(obj_box)
    obj_edit = QLineEdit()
    obj_edit.setPlaceholderText("path to the run's objective.py")
    obj_status = QLabel(_NO_OBJECTIVE)
    obj_status.setStyleSheet("color: grey;")
    browse = QPushButton("Browse")
    load_btn = QPushButton("Load objective")
    unload_btn = QPushButton("Unload")
    unload_btn.setToolTip("Go back to the keys the selected steps carry, with no senses")
    unload_btn.setEnabled(False)
    obj_layout.addWidget(_row(obj_edit, browse, load_btn, unload_btn))
    obj_layout.addWidget(obj_status)
    layout.addWidget(obj_box)

    # ---- Objective count -------------------------------------------------------
    count_group = QButtonGroup(page)
    count_buttons = {}
    count_row = QWidget()
    count_layout = QHBoxLayout(count_row)
    count_layout.setContentsMargins(0, 0, 0, 0)
    lead = QLabel("Objectives:")
    lead.setFixedWidth(78)
    count_layout.addWidget(lead)
    for name in OBJECTIVE_COUNTS:
        button = QRadioButton(name)
        count_group.addButton(button)
        count_buttons[name] = button
        count_layout.addWidget(button)
    count_layout.addStretch()
    count_buttons["1"].setToolTip("A single-objective campaign: the objective over its "
                                  "parameters, and the best value it reached")
    count_buttons["2"].setChecked(True)
    layout.addWidget(count_row)

    def _n_objectives() -> str:
        button = count_group.checkedButton()
        return button.text() if button else "2"

    # ---- Axes ----------------------------------------------------------------
    axes_box = QGroupBox("Objectives")
    axes_layout = QVBoxLayout(axes_box)
    x_combo, y_combo, z_combo = QComboBox(), QComboBox(), QComboBox()
    x_entry, y_entry, z_entry = QLineEdit(), QLineEdit(), QLineEdit()
    x_row, x_lead, x_sense = _axis_row("x:", x_combo, x_entry)
    y_row, y_lead, y_sense = _axis_row("y:", y_combo, y_entry)
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
    # Grouped aggregates replicate runs at the same observation index; the error-bar mode
    # only means anything once it is on.
    cb_grouped = QCheckBox("Grouped")
    rb_sem = QRadioButton("Std. error")
    rb_sem.setChecked(True)
    rb_std = QRadioButton("Std. dev.")
    rb_minmax = QRadioButton("Min/max")
    # What the three bars mean, since they answer different questions and the difference
    # decides how a figure is read.
    ERRORBAR_TEXT = {
        rb_sem: "Std. error — how well the point's position is known. Repeats shrink it by √n.",
        rb_std: "Std. dev. — how much a single measurement scatters. Repeats do not shrink it.",
        rb_minmax: "Min/max — the plain range of the group's measurements.",
    }
    errorbar_help = QLabel("")
    errorbar_help.setWordWrap(True)
    errorbar_help.setStyleSheet("color: grey;")
    cb_numbers = QCheckBox("Show numbers")
    # The true objective behind the campaign. Only the objective knows it, so this stays
    # disabled until one is loaded.
    cb_ground = QCheckBox("Ground truth")
    cb_ground.setEnabled(False)
    cb_ground.setToolTip("Load an objective to draw the true front under the observations")
    gt_method = QComboBox()
    gt_method.addItems(["random", "grid"])
    gt_method.setFixedWidth(90)
    gt_samples = QSpinBox()
    gt_samples.setRange(16, 1_000_000)
    gt_samples.setValue(4096)
    gt_samples.setSingleStep(512)
    gt_samples.setPrefix("N = ")
    gt_spacing = QDoubleSpinBox()
    gt_spacing.setDecimals(4)
    gt_spacing.setRange(0.0001, 1.0)
    gt_spacing.setValue(0.05)
    gt_spacing.setSingleStep(0.01)
    gt_spacing.setPrefix("Δ = ")
    for widget in ERRORBAR_TEXT:
        widget.setEnabled(False)
        widget.toggled.connect(
            lambda checked, w=widget: checked and errorbar_help.setText(ERRORBAR_TEXT[w]))
    cb_grouped.stateChanged.connect(
        lambda _s: [w.setEnabled(cb_grouped.isChecked()) for w in ERRORBAR_TEXT])
    errorbar_help.setText(ERRORBAR_TEXT[rb_sem])
    status = QLabel("")
    status.setStyleSheet("color: grey;")
    plot_layout.addWidget(_row(btn_pareto, btn_hv, btn_hvi, btn_refresh))
    plot_layout.addWidget(_row(cb_grouped, rb_sem, rb_std, rb_minmax, cb_numbers))
    plot_layout.addWidget(errorbar_help)
    plot_layout.addWidget(_row(cb_ground, gt_method, gt_samples, gt_spacing))
    plot_layout.addWidget(status)
    layout.addWidget(plot_box)

    # ---- Diagnostics ---------------------------------------------------------
    # The campaign-wide plots that take no axis selection: one row each, as in the
    # original tab's diagnostic frame.
    diag_box = QGroupBox("Diagnostic tools")
    diag_layout = QVBoxLayout(diag_box)
    layout.addWidget(diag_box)
    layout.addStretch()

    # ---- Key discovery -------------------------------------------------------

    def _sync_ground_truth() -> None:
        """Only offer the ground truth when the objective that defines it is loaded."""
        loaded = state["problem"] is not None
        cb_ground.setEnabled(loaded)
        if not loaded:
            cb_ground.setChecked(False)
        on = loaded and cb_ground.isChecked()
        gt_method.setEnabled(on)
        gt_samples.setEnabled(on and gt_method.currentText() == "random")
        gt_spacing.setEnabled(on and gt_method.currentText() == "grid")

    cb_ground.stateChanged.connect(lambda _s: _sync_ground_truth())
    gt_method.currentTextChanged.connect(lambda _t: _sync_ground_truth())

    def _ground_truth_args() -> list:
        if not (cb_ground.isEnabled() and cb_ground.isChecked()):
            return []
        args = ["--ground-truth", obj_edit.text(), "--gt-method", gt_method.currentText()]
        if gt_method.currentText() == "grid":
            return args + ["--gt-spacing", str(gt_spacing.value())]
        return args + ["--gt-samples", str(gt_samples.value())]

    def _parameters() -> list:
        """The problem's parameters: from the objective when loaded, else from the map."""
        problem = state["problem"]
        if problem is not None:
            return [p["label"] for p in problem["parameters"]]
        return _parameter_keys(state["map"])

    def _sync_landscape() -> None:
        """Offer the single-objective landscape only where there is a plane to draw it on.

        Above two parameters the points would be a projection: two far apart in a
        parameter with no axis land on top of each other, so a fold in the surface reads
        as scatter in the data. The best-value trace stays either way - it is about the
        campaign rather than about the space, so it holds regardless of parameter count.
        """
        if _n_objectives() != "1":
            return
        n = len(_parameters())
        btn_pareto.setEnabled(1 <= n <= 2)
        if n == 0:
            btn_pareto.setToolTip("Load an objective, or rebuild the map, to discover "
                                  "the parameters.")
        elif n > 2:
            btn_pareto.setToolTip(
                f"{n} parameters: the objective cannot be drawn against them. Plot the "
                f"best value over the campaign instead.")
        else:
            btn_pareto.setToolTip("")

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
            # No objective loaded: fall back to whatever the last built map recorded,
            # the way the original tab discovered its result keys.
            everything = _result_keys(state["map"])
            objectives = everything
            senses = {}

        repopulate(x_combo, objectives)
        if _n_objectives() == "1":
            # The rows are a landscape rather than a trade-off: one objective, and the
            # one or two parameters it is drawn over. So y and z offer parameters, and
            # neither is an objective whose second entry should be preselected.
            parameters = _parameters()
            repopulate(y_combo, parameters)
            repopulate(z_combo, parameters, blank_first=True)
            # A problem of exactly two parameters is a plane, and the plane is the whole
            # picture there - so both axes are filled in, the way two objectives preselects
            # its second objective. More than two and there is no obvious pair to guess at.
            if len(parameters) == 2:
                z_combo.setCurrentText(parameters[1])
        else:
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
        _sync_landscape()

    def _load_objective(path: str) -> None:
        try:
            problem = problem_definition(load_objective(path))
        # SystemExit alongside Exception, and not by accident: load_objective raises it
        # for a path that is no module at all, which is right for the plot scripts - they
        # are processes, and exiting with the message is how they report it. Here it is
        # one field of a running window, and SystemExit is not an Exception, so catching
        # only that let an empty or misspelled path take the application down.
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - a bad path must not kill the tab
            state["problem"] = None
            obj_status.setText(f"Could not load: {exc}")
            unload_btn.setEnabled(False)
            _sync_ground_truth()
            return
        state["problem"] = problem
        _refresh_keys()
        _sync_ground_truth()
        unload_btn.setEnabled(True)
        senses = ", ".join(f"{o['label']} ({'min' if o['to_minimize'] else 'max'})"
                           for o in problem["objectives"])
        n_obj = len(problem["objectives"])
        obj_status.setText(f"{n_obj} objective{'' if n_obj == 1 else 's'}: {senses}. "
                           f"ref_point={problem['ref_point']}")

    def _unload_objective() -> None:
        """Drop the loaded problem definition and fall back to the steps' own keys.

        The path goes too: it is what --ground-truth is handed, so leaving it behind
        would let a cleared objective still be drawn under the observations.
        """
        state["problem"] = None
        obj_edit.clear()
        obj_status.setText(_NO_OBJECTIVE)
        unload_btn.setEnabled(False)
        _refresh_keys()
        _sync_ground_truth()

    def _browse_objective() -> None:
        chosen, _ = QFileDialog.getOpenFileName(page, "Choose an objective", obj_edit.text(),
                                                "Python (*.py)")
        if chosen:
            obj_edit.setText(chosen)
            _load_objective(chosen)

    browse.clicked.connect(_browse_objective)
    load_btn.clicked.connect(lambda: _load_objective(obj_edit.text()))
    unload_btn.clicked.connect(_unload_objective)
    btn_refresh.clicked.connect(_refresh_keys)

    # ---- Objective-count wiring -----------------------------------------------

    def _on_objective_count_change() -> None:
        count = _n_objectives()
        is_many = count == "4+"
        is_three = count == "3"
        is_single = count == "1"
        # 4+ has no scatter to draw beyond three axes, so the axis frame gives way to
        # the checklist and the Pareto button goes with it.
        axes_box.setVisible(not is_many)
        nd_box.setVisible(is_many)
        z_row.setEnabled(not is_many)
        # z is a real objective only with three; with two it colours the points and with
        # one it is the second parameter, so in neither does its sense mean anything.
        z_sense.setEnabled(is_three)
        # One objective reads the same three rows as a landscape - the one objective, and
        # the one or two parameters it is drawn over - so only the objective's sense is
        # left live.
        axes_box.setTitle("Objective and parameters" if is_single else "Objectives")
        x_lead.setText("objective:" if is_single else "x:")
        y_lead.setText("parameter:" if is_single else "y:")
        z_lead.setText("parameter 2:" if is_single else ("z:" if is_three else "z (colour):"))
        y_sense.setEnabled(not is_single)
        # One objective has no front and no volume: what the two buttons plot instead is
        # the landscape and the best value reached, so they say so.
        cb_ground.setToolTip(
            "Load an objective to draw the true landscape under the observations" if is_single
            else "Load an objective to draw the true front under the observations")
        btn_pareto.setText("Plot objective" if is_single else "Plot Pareto")
        btn_hv.setText("Plot best value" if is_single else "Plot hypervolume")
        btn_hvi.setText("Plot improvement" if is_single else "Plot HV improvement")
        btn_pareto.setEnabled(not is_many)
        # Grouping and point labels belong to the scatter, which 4+ objectives has none of.
        for widget in (cb_grouped, cb_numbers):
            widget.setEnabled(not is_many)
        for widget in ERRORBAR_TEXT:
            widget.setEnabled(cb_grouped.isChecked() and not is_many)
        # The key lists mean different things per objective count, so they are rebuilt
        # rather than left showing the previous one's.
        _refresh_keys()

    count_group.buttonToggled.connect(
        lambda _b, checked: checked and _on_objective_count_change())

    # ---- Launching -----------------------------------------------------------

    def _rebuild_map() -> bool:
        """Rebuild experiment_map.json + group_map.json from the current selection.

        Synchronous and cheap - it only re-reads the selected step records - so a plot
        launched right after always sees the selection as it is now. The scripts read the
        map through configs.settings.data_path, so that is repointed here too.
        """
        steps = step_list.checked_paths
        if not steps:
            status.setText("Select at least one step in the Steps window.")
            return False
        try:
            exp_map = build_map(steps)
            groups = build_groups(exp_map, decimals_spin.value())
        except Exception as exc:  # noqa: BLE001 - a build error must not kill the click
            status.setText(f"Map build failed: {exc}")
            map_status.setText(f"Map build failed: {exc}")
            return False

        state["map"], state["groups"] = exp_map, groups
        state["source"] = "selection"
        # A plot is a separate process reading configs.settings.data_path, so the map has
        # to exist somewhere on disk for it. That somewhere is a scratch directory for the
        # session, not the user's data tree - Save map is how a copy gets kept.
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        series = len({e["experiment_type"] for e in exp_map["experiments"]})
        map_status.setText(f"{len(exp_map['experiments'])} observations from "
                           f"{len(steps)} selected director"
                           f"{'y' if len(steps) == 1 else 'ies'}, {series} series, "
                           f"held in memory")
        return True

    def _write_map(out_dir) -> Path:
        """Write the map held in memory into `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment_map.json").write_text(
            json.dumps(state["map"], indent=2), encoding="utf-8")
        (out_dir / "group_map.json").write_text(
            json.dumps(state["groups"], indent=2), encoding="utf-8")
        return out_dir

    def _regroup() -> None:
        """Re-group the map held in memory at the current decimals.

        Applied to whatever is in memory, loaded map included: the saved grouping was made
        at whatever decimals were set then, and leaving it in place would mean the spinbox
        reads one thing while the plots group by another. Only the grouping is redone - the
        observations themselves do not depend on it.
        """
        if not state["map"]:
            return
        state["groups"] = build_groups(state["map"], decimals_spin.value())
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        repeated = [g["replicates"] for g in state["groups"] if g["replicates"] > 1]
        map_status.setText(
            f"{len(state['map']['experiments'])} observations, {len(state['groups'])} "
            f"groups at {decimals_spin.value()} decimals, {len(repeated)} with repeats"
            + (f" (sizes {sorted(repeated, reverse=True)})" if repeated else ""))

    decimals_spin.valueChanged.connect(lambda _v: _regroup())

    def _ensure_map() -> bool:
        """The map to work from: a loaded one as it is, else rebuilt from the selection.

        A loaded map is left alone so that plotting it does not quietly replace it with
        whatever happens to be ticked in the tree; Rebuild map now is how you go back to
        the selection.
        """
        if state["source"] == "loaded" and state["map"]:
            return True
        return _rebuild_map()

    def _load_map() -> None:
        """Read a saved map back in, and work from it until the next rebuild."""
        chosen = QFileDialog.getExistingDirectory(page, "Load a map from",
                                                  step_list.root or "")
        if not chosen:
            return
        source = Path(chosen)
        try:
            exp_map = json.loads(
                (source / "experiment_map.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            map_status.setText(f"Could not read experiment_map.json there: {exc}")
            return
        if not isinstance(exp_map, dict) or "experiments" not in exp_map:
            map_status.setText(f"{source / 'experiment_map.json'} is not an experiment map.")
            return
        try:
            groups = json.loads((source / "group_map.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            groups = None
        # A map saved without its groups, or saved before grouping wrote a group_id into
        # each entry, is still usable: the grouping is derivable from the map itself.
        if groups is None or any("group_id" not in e
                                 for e in exp_map.get("experiments", [])):
            groups = build_groups(exp_map, decimals_spin.value())

        state["map"], state["groups"] = exp_map, groups
        state["source"] = "loaded"
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        _refresh_keys()
        map_status.setText(f"Loaded {len(exp_map['experiments'])} observations from "
                           f"{source} — plots use this until you rebuild")

    def _save_map() -> None:
        """Ask where to keep a copy of the map, and write it there."""
        if not state["map"]:
            map_status.setText("Nothing to save — click “Rebuild map now” first.")
            return
        chosen = QFileDialog.getExistingDirectory(page, "Save the map to",
                                                  step_list.root or "")
        if not chosen:
            return
        try:
            _write_map(chosen)
        except OSError as exc:
            map_status.setText(f"Could not save: {exc}")
            return
        map_status.setText(f"Saved experiment_map.json and group_map.json to {chosen}")

    def _sense_args(*pairs) -> list:
        """--maximize for each axis whose toggle is on.

        The scripts minimise by default and take only --maximize, so a minimised axis
        contributes no flag at all.
        """
        args = []
        for key, toggle in pairs:
            if key and toggle.isChecked():
                args += ["--maximize", key]
        return args

    def _constraint_args() -> list:
        args = []
        for spec in con_collect():
            args += ["--constraint", spec]
        return args

    def _launch(script: str, *extra) -> None:
        """Make sure a map exists, then run one of the analysis modules against it."""
        if not _ensure_map():
            return
        module = f"{MODULES}.{script}"
        proc = launch_analysis(module, *extra)
        watch([proc],
              on_start=lambda: set_text_async(status, f"Running {script}..."),
              on_done=lambda: set_text_async(status, "Done."))

    def _grouped_args() -> list:
        if not cb_grouped.isChecked():
            return []
        mode = "sem" if rb_sem.isChecked() else "std" if rb_std.isChecked() else "minmax"
        return ["--grouped", "--errorbar", mode]

    def _plot_objective() -> None:
        """The single-objective landscape: the objective over the one or two parameters
        chosen."""
        obj = x_combo.currentText()
        if not obj:
            status.setText("Choose the objective.")
            return
        pairs = [(combo, entry) for combo, entry in ((y_combo, y_entry), (z_combo, z_entry))
                 if combo.currentText()]
        if not pairs:
            status.setText("Choose a parameter to draw the objective against.")
            return
        extra = _constraint_args() + ["--objective", obj,
                                      "--objective-label", x_entry.text()]
        # --parameter-label is read positionally against --parameter, so the two are
        # passed as pairs and stay in step.
        for combo, entry in pairs:
            extra += ["--parameter", combo.currentText(),
                      "--parameter-label", entry.text()]
        extra += _grouped_args()
        if cb_numbers.isChecked():
            extra.append("--show-numbers")
        extra += _sense_args((obj, x_sense)) + _ground_truth_args()
        _launch("plot_objective", *extra)

    def _plot_pareto() -> None:
        if _n_objectives() == "1":
            _plot_objective()
            return
        x, y, z = x_combo.currentText(), y_combo.currentText(), z_combo.currentText()
        if not x or not y:
            status.setText("Choose an x and a y objective.")
            return
        extra = _constraint_args()
        if _n_objectives() == "3":
            if not z:
                status.setText("Three objectives needs a z objective.")
                return
            extra += _grouped_args()
            if cb_numbers.isChecked():
                extra.append("--show-numbers")
            extra += ["--x", x, "--y", y, "--z", z,
                      "--xlabel", x_entry.text(), "--ylabel", y_entry.text(),
                      "--zlabel", z_entry.text()]
            extra += _sense_args((x, x_sense), (y, y_sense), (z, z_sense))
            _launch("plot_pareto_3d", *extra)
            return
        extra += _grouped_args()
        if cb_numbers.isChecked():
            extra.append("--show-numbers")
        extra += ["--x", x, "--y", y,
                  "--xlabel", x_entry.text(), "--ylabel", y_entry.text()]
        if z:
            extra += ["--z", z, "--zlabel", z_entry.text()]
        extra += _sense_args((x, x_sense), (y, y_sense)) + _ground_truth_args()
        _launch("plot_pareto_2d", *extra)

    def _plot_hypervolume(improvement: bool) -> None:
        extra = _constraint_args() + (["--improvement"] if improvement else [])
        if cb_grouped.isChecked():
            extra.append("--grouped")
        if _n_objectives() == "1":
            # One objective: the metric is the best value it reached, so the objective
            # goes in on its own and the parameter rows have no part in it.
            x = x_combo.currentText()
            if not x:
                status.setText("Choose the objective.")
                return
            extra += ["--objective", x] + _sense_args((x, x_sense))
        elif _n_objectives() == "4+":
            chosen = nd_collect()
            if len(chosen) < 2:
                status.setText("Tick at least two objectives for the hypervolume.")
                return
            for key, _is_max in chosen:
                extra += ["--objective", key]
            extra += [flag for key, is_max in chosen if is_max
                      for flag in ("--maximize", key)]
        else:
            x, y, z = x_combo.currentText(), y_combo.currentText(), z_combo.currentText()
            if not x or not y:
                status.setText("Choose an x and a y objective.")
                return
            pairs = [(x, x_sense), (y, y_sense)]
            extra += ["--x", x, "--y", y]
            # The two-objective colour axis is not an objective, so it joins only with
            # three objectives.
            if _n_objectives() == "3" and z:
                extra += ["--z", z]
                pairs.append((z, z_sense))
            extra += _sense_args(*pairs)
        _launch("plot_hypervolume", *extra)

    # The viewers rebuild first, so what they show is the current selection rather than
    # whatever was last written.
    btn_rebuild.clicked.connect(lambda: _rebuild_map() and _refresh_keys())
    btn_view_exp.clicked.connect(
        lambda: _ensure_map() and _view_json_dialog(page, state["map"], "Experiment map"))
    btn_view_grp.clicked.connect(
        lambda: _ensure_map() and _view_group_map_dialog(page, state["groups"],
                                                         state["map"]))
    btn_save_map.clicked.connect(_save_map)
    btn_load_map.clicked.connect(_load_map)

    btn_pareto.clicked.connect(_plot_pareto)
    btn_hv.clicked.connect(lambda: _plot_hypervolume(False))
    btn_hvi.clicked.connect(lambda: _plot_hypervolume(True))

    # Diagnostic rows: label, script, and whether it understands --grouped.
    for text, script, grouped in (
            ("Evolution", "plot_evolution", True),
            ("Correlation matrix", "plot_correlation_matrix", False),
            ("Results boxplot", "plot_results_boxplot", False),
            ("Results vs datetime", "plot_results_vs_datetime", False),
    ):
        label = QLabel(f"{text}:")
        label.setFixedWidth(160)
        button = QPushButton("Plot")
        button.clicked.connect(
            lambda _checked=False, s=script, g=grouped:
            _launch(s, *(["--grouped"] if g and cb_grouped.isChecked() else [])))
        diag_layout.addWidget(_row(label, button))

    # An objective usually sits with the tutorial that produced the data, so offer the
    # first one found under the selected root rather than leaving the field blank.
    if step_list.root:
        guess = next(Path(step_list.root).glob("**/objective.py"), None)
        if guess is not None:
            obj_edit.setText(str(guess))

    _on_objective_count_change()
    _sync_ground_truth()
    return page
