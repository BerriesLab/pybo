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
    QDoubleSpinBox, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from pybo_gui.configs import settings as configs_settings
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import build_map
from pybo_gui.modules.bayesian_campaign_analysis.build_group_map import build_groups
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective, problem_definition
from pybo_gui.gui.launchers import launch_analysis, watch
from pybo_gui.gui.message_log import post
from pybo_gui.gui.widgets import (
    bind_label_entry, make_constraints_widget, make_objective_checklist,
    make_parameters_widget, make_sense_toggle, make_trackers_widget, repopulate,
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
    # What counts as the same setting is the resolution in the Parameters box, and only
    # that: a second knob here would be a second answer to one question, free to disagree
    # with the objective's own. A parameter with no resolution is compared as measured.
    map_layout.addWidget(_row(btn_rebuild, btn_view_exp, btn_view_grp, btn_save_map,
                              btn_load_map))
    layout.addWidget(map_box)

    # ---- Objective -----------------------------------------------------------
    obj_box = QGroupBox("Objective")
    obj_layout = QVBoxLayout(obj_box)
    obj_edit = QLineEdit()
    obj_edit.setPlaceholderText("path to the run's objective.py")
    browse = QPushButton("Browse")
    load_btn = QPushButton("Load objective")
    unload_btn = QPushButton("Unload")
    unload_btn.setToolTip("Go back to the keys the selected steps carry, with no senses")
    unload_btn.setEnabled(False)
    obj_layout.addWidget(_row(obj_edit, browse, load_btn, unload_btn))
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

    # ---- Parameters ----------------------------------------------------------
    # Above the objectives because the parameters are what a setting *is*: the resolution
    # here decides which observations count as repeats of one, which the grouped error
    # bars and every group-aware plot then rest on.
    # Deferred on purpose: _regroup is defined further down, and the lambda only looks it
    # up when an edit is actually committed.
    par_box, par_collect, par_set_keys = make_parameters_widget(
        on_change=lambda: _regroup())
    layout.addWidget(par_box)

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

    trk_box, trk_set_keys = make_trackers_widget()
    layout.addWidget(trk_box)

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
    # A different axis from Grouped: that averages the repeats of one setting within a
    # step, this averages whole runs of an arm against each other. They compose, so both
    # can be on at once.
    cb_aggregate = QCheckBox("Average runs")
    band_combo = QComboBox()
    band_combo.addItems(["ci95", "sem", "std", "minmax"])
    BAND_TEXT = {
        "ci95": "95% CI - where the arm's mean lies. The one that says whether two arms differ.",
        "sem": "Std. error of the mean - narrower than the CI, and not a significance statement.",
        "std": "Std. dev. - where a single run lands. Does not shrink with more runs.",
        "minmax": "Min/max - the plain range across the runs.",
    }
    cb_numbers = QCheckBox("Show numbers")
    # 2-D only. Whether the non-dominated points are joined is not a judgement the plot
    # can make for the user: on a constrained problem the line claims trade-offs a
    # constraint may forbid, which is still worth drawing when the front is what the
    # figure is about.
    cb_front = QCheckBox("Front line")
    cb_front.setChecked(True)
    cb_front.setToolTip("Join the non-dominated points into a dashed front line, on a "
                        "constrained problem as well as an unconstrained one. There the "
                        "line is a guide: a trade-off between two front points is not "
                        "necessarily attainable.")
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
    # A fraction of each axis' own range, so 1.0 is the coarsest grid there is - two
    # points per axis, the bounds - and anything above it would mean less than that.
    # Small values are still uncapped in point count: _grid_X prints what it is about
    # to build rather than refusing it.
    gt_spacing.setRange(0.0001, 1.0)
    gt_spacing.setValue(0.05)
    gt_spacing.setSingleStep(0.01)
    gt_spacing.setPrefix("Δ = ")
    gt_spacing.setToolTip("Step along each axis as a fraction of its range: "
                          "0.05 is 21 points per parameter, whatever its units")
    cb_gt_noisy = QCheckBox("Noisy")
    cb_gt_noisy.setToolTip("Draw the ground truth the way a run would have observed it, "
                           "noise and all, instead of the noiseless value underneath")
    # The explanations live on the controls themselves rather than in a line underneath.
    # They describe what a control means, not what an action did, so the log is the wrong
    # place for them - and a tooltip is always available instead of only after a click.
    for widget, text in ERRORBAR_TEXT.items():
        widget.setEnabled(False)
        widget.setToolTip(text)
    cb_grouped.stateChanged.connect(
        lambda _s: [w.setEnabled(cb_grouped.isChecked()) for w in ERRORBAR_TEXT])
    for position, key in enumerate(BAND_TEXT):
        band_combo.setItemData(position, BAND_TEXT[key], Qt.ItemDataRole.ToolTipRole)

    def _on_band_change() -> None:
        on = cb_aggregate.isChecked()
        band_combo.setEnabled(on)
        # The combo shows one entry at a time, so its own tooltip has to follow it: the
        # per-item tooltips are only reachable once the list is open.
        band_combo.setToolTip(BAND_TEXT[band_combo.currentText()] if on else "")

    cb_aggregate.stateChanged.connect(lambda _s: _on_band_change())
    band_combo.currentTextChanged.connect(lambda _t: _on_band_change())
    _on_band_change()
    plot_layout.addWidget(_row(btn_pareto, btn_hv, btn_hvi, btn_refresh))
    plot_layout.addWidget(_row(cb_grouped, rb_sem, rb_std, rb_minmax, cb_numbers, cb_front))
    plot_layout.addWidget(_row(cb_aggregate, QLabel("Band:"), band_combo))
    plot_layout.addWidget(_row(cb_ground, gt_method, gt_samples, gt_spacing, cb_gt_noisy))
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
        cb_gt_noisy.setEnabled(on)

    cb_ground.stateChanged.connect(lambda _s: _sync_ground_truth())
    gt_method.currentTextChanged.connect(lambda _t: _sync_ground_truth())

    def _ground_truth_args() -> list:
        if not (cb_ground.isEnabled() and cb_ground.isChecked()):
            return []
        args = ["--ground-truth", obj_edit.text(), "--gt-method", gt_method.currentText()]
        args += ["--gt-spacing", str(gt_spacing.value())] if gt_method.currentText() == "grid" \
            else ["--gt-samples", str(gt_samples.value())]
        if cb_gt_noisy.isChecked():
            args.append("--gt-noisy")
        return args

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
        # Only the objective knows the bounds, units and resolutions; with none loaded the
        # rows still appear, bare, so a resolution can be typed in before one is picked.
        par_set_keys(_parameters(),
                     {p["label"]: p for p in problem["parameters"]}
                     if problem is not None else {})
        trk_set_keys(problem["trackers"] if problem is not None else [])
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
            post(f"Could not load: {exc}")
            unload_btn.setEnabled(False)
            _sync_ground_truth()
            return
        state["problem"] = problem
        _refresh_keys()
        # The Parameters box now shows the objective's own resolutions; a map already
        # built from before was grouped at whatever was there previously (typically
        # blank, i.e. compared as measured), so it has to be regrouped to match.
        _regroup()
        _sync_ground_truth()
        unload_btn.setEnabled(True)
        senses = ", ".join(f"{o['label']} ({'min' if o['to_minimize'] else 'max'})"
                           for o in problem["objectives"])
        n_obj = len(problem["objectives"])
        post(f"{n_obj} objective{'' if n_obj == 1 else 's'}: {senses}. "
                           f"ref_point={problem['ref_point']}")

    def _unload_objective() -> None:
        """Drop the loaded problem definition and fall back to the steps' own keys.

        The path goes too: it is what --ground-truth is handed, so leaving it behind
        would let a cleared objective still be drawn under the observations.
        """
        state["problem"] = None
        obj_edit.clear()
        post(_NO_OBJECTIVE)
        unload_btn.setEnabled(False)
        _refresh_keys()
        # Same reasoning as loading: whatever resolutions the boxes show now may differ
        # from what the map was last grouped at, so a map already built has to catch up.
        _regroup()
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
        # No steps is a request for the ground truth on its own, as long as there is a
        # ground truth to draw: build_map([]) is an empty map, and a plot over one draws
        # the backdrop and no series. Without one it is just an empty plot, so that still
        # asks for a selection rather than opening a blank figure.
        if not steps and not cb_ground.isChecked():
            post("Select at least one step in the Steps window.")
            return False
        try:
            exp_map = build_map(steps)
            groups = build_groups(exp_map, par_collect())
        except Exception as exc:  # noqa: BLE001 - a build error must not kill the click
            post(f"Map build failed: {exc}")
            return False

        state["map"], state["groups"] = exp_map, groups
        state["source"] = "selection"
        # A plot is a separate process reading configs.settings.data_path, so the map has
        # to exist somewhere on disk for it. That somewhere is a scratch directory for the
        # session, not the user's data tree - Save map is how a copy gets kept.
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        if not steps:
            post("No steps selected — the ground truth will be drawn "
                               "on its own")
            return True
        series = len({e["experiment_type"] for e in exp_map["experiments"]})
        post(f"{len(exp_map['experiments'])} observations from "
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
        """Re-group the map held in memory at the current resolutions.

        Applied to whatever is in memory, loaded map included: the saved grouping was made
        at whatever resolutions were set then, and leaving it in place would mean the
        Parameters box reads one thing while the plots group by another. Only the grouping
        is redone - the observations themselves do not depend on it.
        """
        if not state["map"]:
            return
        resolutions = par_collect()
        state["groups"] = build_groups(state["map"], resolutions)
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        # Just that the box took effect. The counts said little, and listing every group's
        # size ran to hundreds of numbers on a study-sized map, which is what stretched the
        # window: a status line has to fit on one.
        n = len(state["groups"])
        post(f"regrouped into {n} groups, "
                           f"{len(resolutions)} parameters by resolution")

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
            post(f"Could not read experiment_map.json there: {exc}")
            return
        if not isinstance(exp_map, dict) or "experiments" not in exp_map:
            post(f"{source / 'experiment_map.json'} is not an experiment map.")
            return
        try:
            groups = json.loads((source / "group_map.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            groups = None
        # A map saved without its groups, or saved before grouping wrote a group_id into
        # each entry, is still usable: the grouping is derivable from the map itself.
        if groups is None or any("group_id" not in e
                                 for e in exp_map.get("experiments", [])):
            groups = build_groups(exp_map, par_collect())

        state["map"], state["groups"] = exp_map, groups
        state["source"] = "loaded"
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        _refresh_keys()
        post(f"Loaded {len(exp_map['experiments'])} observations from "
                           f"{source} — plots use this until you rebuild")

    def _save_map() -> None:
        """Ask where to keep a copy of the map, and write it there."""
        if not state["map"]:
            post("Nothing to save — click “Rebuild map now” first.")
            return
        chosen = QFileDialog.getExistingDirectory(page, "Save the map to",
                                                  step_list.root or "")
        if not chosen:
            return
        try:
            _write_map(chosen)
        except OSError as exc:
            post(f"Could not save: {exc}")
            return
        post(f"Saved experiment_map.json and group_map.json to {chosen}")

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
              on_start=lambda: post(f"Running {script}..."),
              on_done=lambda: post("Done."),
              on_fail=lambda codes: post(
                  f"{script} exited with {codes[0]} — see its console. "
                          f"Exit 2 is a constraint that would not parse."),
              # The script's own console, which nothing else here shows - a plot that
              # still exits 0 (an arm skipped for lack of a common range, say) has no
              # other way to say so.
              on_output=lambda line: post(f"{script}: {line}"))

    def _aggregate_args() -> list:
        """The flags for collapsing an arm's runs into one curve, or nothing when off."""
        if not cb_aggregate.isChecked():
            return []
        return ["--aggregate-runs", "--band", band_combo.currentText()]

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
            post("Choose the objective.")
            return
        pairs = [(combo, entry) for combo, entry in ((y_combo, y_entry), (z_combo, z_entry))
                 if combo.currentText()]
        if not pairs:
            post("Choose a parameter to draw the objective against.")
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
            post("Choose an x and a y objective.")
            return
        extra = _constraint_args()
        if _n_objectives() == "3":
            if not z:
                post("Three objectives needs a z objective.")
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
        extra += _aggregate_args()
        # Said either way rather than left to the plot's own judgement: the checkbox is
        # the answer, constrained problem or not.
        extra += ["--front-line", "always" if cb_front.isChecked() else "never"]
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
                post("Choose the objective.")
                return
            extra += ["--objective", x] + _sense_args((x, x_sense))
        elif _n_objectives() == "4+":
            chosen = nd_collect()
            if len(chosen) < 2:
                post("Tick at least two objectives for the hypervolume.")
                return
            for key, _is_max in chosen:
                extra += ["--objective", key]
            extra += [flag for key, is_max in chosen if is_max
                      for flag in ("--maximize", key)]
        else:
            x, y, z = x_combo.currentText(), y_combo.currentText(), z_combo.currentText()
            if not x or not y:
                post("Choose an x and a y objective.")
                return
            pairs = [(x, x_sense), (y, y_sense)]
            extra += ["--x", x, "--y", y]
            # The two-objective colour axis is not an objective, so it joins only with
            # three objectives.
            if _n_objectives() == "3" and z:
                extra += ["--z", z]
                pairs.append((z, z_sense))
            extra += _sense_args(*pairs)
        # The per-step gain view rewrites what a point means, and the plot refuses the
        # two together, so the flags are only added to the cumulative one.
        if not improvement:
            extra += _aggregate_args()
        # Where the hypervolume is measured from. Sent whenever an objective is loaded,
        # regardless of the Ground truth checkbox - that one is about drawing the
        # sampled surface, a separate question from having a fixed reference point, and
        # a moving one would make a run's own hypervolume depend on which other runs
        # happen to be selected alongside it.
        if state["problem"] is not None:
            extra += ["--ground-truth", obj_edit.text()]
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

    # Diagnostic rows: label, script, and whether it understands --grouped and
    # --aggregate-runs. None of these average runs: at a given step, runs with different
    # seeds are evaluating unrelated points, so the mean of their raw values describes
    # nothing - only a cumulative quantity like the hypervolume means the same thing at
    # step k in every run.
    for text, script, grouped, aggregates in (
            ("Evolution", "plot_evolution", True, False),
            ("Correlation matrix", "plot_correlation_matrix", False, False),
            ("Results boxplot", "plot_results_boxplot", False, False),
            ("Results vs datetime", "plot_results_vs_datetime", False, False),
    ):
        label = QLabel(f"{text}:")
        label.setFixedWidth(160)
        button = QPushButton("Plot")
        button.clicked.connect(
            lambda _checked=False, s=script, g=grouped, a=aggregates:
            _launch(s, *(["--grouped"] if g and cb_grouped.isChecked() else []),
                    *(_aggregate_args() if a else [])))
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
