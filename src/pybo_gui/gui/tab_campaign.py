"""The campaign-analysis tabs: the constructor, and the plots.

Turns a step selection plus a problem definition into a pybo_gui.modules.bayesian_campaign_analysis command line.
The tabs draw nothing themselves: every button launches the matching analysis module, which
stays runnable from a terminal with the same flags.

Two pages, one `build`: the constructor assembles the campaign - the experiment map, the
objective behind it, the resolutions that decide what counts as one setting - and the plot
page turns it into figures. They share every bit of that, so they are built in one closure
and returned as a pair rather than each holding half of it.

Load objective is what fills the tab in - the step records name their values but carry no
problem definition, so the axis keys, the min/max senses and the hypervolume reference
point all come from the run's objective.py. Every sense stays editable afterwards: the
objective is the default, not the last word.
"""
import json
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListView,
    QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
    QDoubleSpinBox, QSpinBox, QSplitter, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from pybo_gui.configs import settings as configs_settings
from pybo_gui.configs import workspace
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import (
    build_map, map_stamp, stamp_digest)
from pybo_gui.modules.bayesian_campaign_analysis.build_group_map import build_groups
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import load_objective, problem_definition
from pybo_gui.gui.launchers import (
    launch_analysis, run_off_thread, stop_token, stopped_since, watch,
)
from pybo_gui.gui.message_log import post
from pybo_gui.gui.widgets import (
    bind_label_entry, make_constraints_widget, make_objective_checklist,
    make_objectives_widget, make_parameters_widget, make_sense_toggle,
    make_trackers_widget, repopulate,
)

MODULES = "pybo_gui.modules.bayesian_campaign_analysis"


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


def _edit_objective_dialog(parent: QWidget, path: str, on_saved) -> None:
    """A non-modal editor for the objective.py at `path`: edit its source and save it
    back in place. `on_saved` runs after every successful save, so the caller can reload
    the objective and pick up whatever changed - the same edit-then-Reload loop
    load_btn's own tooltip describes, minus the trip out to a separate text editor.

    Left open after a save rather than closing, since tuning an objective is usually
    several rounds of edit-save-check rather than one.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        post(f"Could not open {path}: {exc}")
        return

    dlg = QDialog(parent)
    dlg.resize(900, 700)
    v = QVBoxLayout(dlg)

    editor = QPlainTextEdit()
    editor.setPlainText(text)
    editor.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    # No line-wrap for source code: a wrapped long line reads like two, and moves under
    # you as you type on the ones before it.
    editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    v.addWidget(editor)

    status = QLabel()
    status.setStyleSheet("color: grey;")
    v.addWidget(status)

    save_btn = QPushButton("Save")
    close_btn = QPushButton("Close")
    v.addWidget(_row(save_btn, close_btn))

    # A dict rather than a bare bool so the nested closures below can write it - same
    # reason build()'s own `state` dicts exist.
    dirty = {"value": False}

    def _mark_dirty() -> None:
        dirty["value"] = True
        dlg.setWindowTitle(f"Edit objective — {path} *")

    def _mark_clean(message: str) -> None:
        dirty["value"] = False
        dlg.setWindowTitle(f"Edit objective — {path}")
        status.setStyleSheet("color: grey;")
        status.setText(message)

    editor.textChanged.connect(_mark_dirty)
    _mark_clean(f"Editing {path}")

    def _save() -> None:
        try:
            Path(path).write_text(editor.toPlainText(), encoding="utf-8")
        except OSError as exc:
            status.setStyleSheet("color: red;")
            status.setText(f"Could not save: {exc}")
            return
        _mark_clean(f"Saved {path}.")
        post(f"Objective saved: {path}")
        on_saved()

    save_btn.clicked.connect(_save)
    close_btn.clicked.connect(dlg.close)

    def _close_event(event) -> None:
        if not dirty["value"]:
            event.accept()
            return
        choice = QMessageBox.question(
            dlg, "Unsaved changes", "Discard unsaved changes to the objective?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if choice == QMessageBox.StandardButton.Discard:
            event.accept()
        else:
            event.ignore()

    dlg.closeEvent = _close_event
    dlg.show()


class _GroupItem(QTreeWidgetItem):
    """A group map row that sorts a column by number when it holds numbers.

    Qt compares the cell text, which reads wrong down a parameter column: 10 lands before
    9, and a group with no value there ("None") sorts wherever the alphabet puts it.
    Numbers are compared as numbers, and anything else falls back to the text - so the
    cells with no number gather at one end instead of interleaving.
    """

    def __lt__(self, other):
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        mine, theirs = self.text(column), other.text(column)
        try:
            return float(mine) < float(theirs)
        except ValueError:
            return mine < theirs


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
        item = _GroupItem([str(g.get(c, "")) for c in cols])
        # The group_id travels with the row rather than being read back out of a cell, so
        # the detail pane keeps working however the rows are ordered.
        item.setData(0, Qt.ItemDataRole.UserRole, g.get("group_id"))
        tree.addTopLevelItem(item)
    # After the rows are in: sorting while inserting re-sorts on every append.
    tree.header().setSortIndicatorShown(True)
    tree.setSortingEnabled(True)
    tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)
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


def build(step_list, settings) -> tuple[QWidget, QWidget]:
    """Construct the two pages, constructor first: what assembles a campaign, and what
    plots it.

    `step_list` is the selector window it reads the selection from; `settings` is the
    shared object the Settings tab writes, read at launch so a style change applies to
    the next plot without rebuilding anything.
    """
    # Two pages, one closure. The tab is split in two - assembling the campaign, and
    # plotting it - but the halves share one state, one map and one set of key lists:
    # loading an objective repopulates the axis combos, and a plot click rebuilds the map
    # from the current selection. Built together and handed back as a pair, so neither of
    # those has to reach across a module boundary to find the other.
    page = QWidget()
    layout = QVBoxLayout(page)
    plot_page = QWidget()
    plot_page_layout = QVBoxLayout(plot_page)
    # "source" records where the map came from: a selection rebuild, or a file the
    # user loaded. A loaded map must survive the next plot click rather than being
    # silently rebuilt from a selection it has nothing to do with.
    state: dict = {"problem": None, "map": None, "groups": None, "source": None}
    # One directory per session, so a rebuild never writes into the user's data tree.
    # Save map is the only thing that puts a map where they chose.
    #
    # Read once, here: the Workspace setting can change while this session runs, but the
    # directory it already writes to must not, or configs.settings.data_path and the map
    # on disk would drift apart. Unset means a temporary directory, as it always was.
    _scratch = workspace.new_instance_dir()

    # ---- Experiment map ------------------------------------------------------
    # First: it is what every other action on this page ends up calling, and the one
    # thing that carries a change made below - an objective, a resolution - through to
    # the plots. Further down it reads as an afterthought rather than the step nothing
    # takes effect without.
    map_box = QGroupBox("Experiment map")
    map_layout = QVBoxLayout(map_box)
    btn_rebuild = QPushButton("Rebuild map now")
    btn_regroup = QPushButton("Regroup")
    btn_view_exp = QPushButton("View experiment map")
    btn_view_grp = QPushButton("View group map")
    btn_save_map = QPushButton("Save map")
    btn_load_map = QPushButton("Load map")
    # What counts as the same setting is the resolution in the Parameters box, and only
    # that: a second knob here would be a second answer to one question, free to disagree
    # with the objective's own. A parameter with no resolution is compared as measured.
    map_layout.addWidget(_row(btn_rebuild, btn_regroup, btn_view_exp, btn_view_grp, btn_save_map,
                              btn_load_map))
    layout.addWidget(map_box)

    # ---- Objective -----------------------------------------------------------
    # First: nothing else on either page is usable until this is loaded.
    obj_box = QGroupBox("Objective")
    obj_layout = QVBoxLayout(obj_box)
    obj_edit = QLineEdit()
    obj_edit.setPlaceholderText("path to the run's objective.py")
    browse = QPushButton("Browse")
    # Browse loads what it picks, so this is not how an objective first arrives. It is
    # here for the two cases Browse does not cover: a path typed into the field by hand,
    # and re-reading a file that has been edited since it was loaded.
    load_btn = QPushButton("Reload objective")
    load_btn.setToolTip("Read the file again — after editing the objective, or for a "
                        "path typed in by hand")
    edit_btn = QPushButton("Edit objective")
    edit_btn.setToolTip("Open the file in a built-in editor and save changes back to "
                        "it - the objective reloads automatically on every save.")
    # No Unload: with the objective required, dropping it would only leave a tab that
    # can do nothing. Loading another one over it is how a campaign changes objective.
    obj_layout.addWidget(_row(obj_edit, browse, load_btn, edit_btn))
    layout.addWidget(obj_box)

    # What the objective just loaded above actually declares, so the campaign says what it
    # is optimising and which way, next to the file it read that from.
    objectives_box, objectives_set_keys = make_objectives_widget()
    layout.addWidget(objectives_box)

    # ---- Objective count -------------------------------------------------------

    def _n_objectives() -> str:
        """How many objectives the loaded problem declares, as the buckets the tab draws
        by: "1", "2", "3" or "4+", and "" with no objective loaded.

        Not a choice any more. The dimensionality is a property of the objective
        function, so offering it as a setting only let the tab be told something the
        problem contradicts - a two-objective campaign drawn as if it had three.
        """
        problem = state["problem"]
        if problem is None:
            return ""
        count = len(problem.get("objectives") or [])
        return {0: "", 1: "1", 2: "2", 3: "3"}.get(count, "4+")

    # ---- Parameters ----------------------------------------------------------
    # Above the objectives because the parameters are what a setting *is*: the resolution
    # here decides which observations count as repeats of one, which the grouped error
    # bars and every group-aware plot then rest on.
    # Editing a resolution does not regroup either: nothing changes what the plots see
    # except Rebuild map now, whether the resolutions came from an objective or were
    # typed here. Deferred on purpose - the lambda only looks the function up when an
    # edit is actually committed, by which time it is defined further down.
    par_box, par_collect, par_set_keys = make_parameters_widget(
        on_change=lambda: _note_resolutions_changed())
    layout.addWidget(par_box)

    # ---- Axes ----------------------------------------------------------------
    # ---- Grouping ------------------------------------------------------------
    # At the top because it cuts across frames: what counts as one point, and what a bar
    # or band around it means, is decided once and applies to whichever plot is drawn
    # next - not to the plot whose button happens to sit nearest.
    #
    # Not every plot takes them yet - the boxplot, the correlation matrix and the two
    # time views ignore both - so each note says which do. That is a gap to close in
    # those scripts, not a property of the setting, and the wording says "yet" so the
    # note does not have to be rewritten as they catch up.
    group_box = QGroupBox("Grouping")
    group_layout = QVBoxLayout(group_box)

    def _grouping_note(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("color: grey;")
        label.setWordWrap(True)
        return label

    # The axes and the two plots drawn against them, in one frame: what the rows pick is
    # what the Pareto front and the hypervolume are measured over, so separating the
    # choice from the buttons that use it only made the reader connect them.
    #
    # "Axes", not "Objectives": with one objective the rows are that objective over its
    # parameters. The constructor tab has the box that says what the objectives are.
    axes_box = QGroupBox("Pareto and hypervolume")
    axes_layout = QVBoxLayout(axes_box)
    # The rows apart from the box, because with four or more objectives there is no
    # scatter to draw and they give way to the checklist - while the hypervolume, which
    # takes its objectives from that checklist instead, still has buttons to offer.
    axes_rows = QWidget()
    axes_rows_layout = QVBoxLayout(axes_rows)
    axes_rows_layout.setContentsMargins(0, 0, 0, 0)
    x_combo, y_combo, z_combo = QComboBox(), QComboBox(), QComboBox()
    x_entry, y_entry, z_entry = QLineEdit(), QLineEdit(), QLineEdit()
    x_row, x_lead, x_sense = _axis_row("x:", x_combo, x_entry)
    y_row, y_lead, y_sense = _axis_row("y:", y_combo, y_entry)
    z_row, z_lead, z_sense = _axis_row("z (colour):", z_combo, z_entry)
    for row in (x_row, y_row, z_row):
        axes_rows_layout.addWidget(row)
    axes_layout.addWidget(axes_rows)
    # The ground truth goes straight under the axis rows: it is read *by* those axes - the
    # surface is sampled over them, and the reference point it carries is what the
    # hypervolume is measured from - so it belongs with them rather than below the buttons
    # that draw them. Its widgets are built further down, so a placeholder holds the
    # position; appending there instead would put the row after every option row.
    gt_holder = QWidget()
    gt_row_slot = QVBoxLayout(gt_holder)
    gt_row_slot.setContentsMargins(0, 0, 0, 0)
    axes_layout.addWidget(gt_holder)
    for combo, entry in ((x_combo, x_entry), (y_combo, y_entry), (z_combo, z_entry)):
        bind_label_entry(combo, entry)
    # Pins the diverging colormap's neutral midpoint to a value the user cares about
    # instead of leaving it wherever the data's own min/max happen to average to.
    # Lives on the z row since it only means anything where z drives the colour -
    # three objectives, or two with a colour key chosen.
    z_center_entry = QLineEdit()
    z_center_entry.setPlaceholderText("centre (optional)")
    z_center_entry.setValidator(QDoubleValidator())
    z_center_entry.setMaximumWidth(110)
    z_center_entry.setToolTip("Value the colour scale is centred on. Left blank, the "
                              "colour midpoint falls wherever the data's own range "
                              "averages to.")
    z_row.layout().insertWidget(z_row.layout().count() - 1, z_center_entry)
    plot_page_layout.addWidget(axes_box)

    nd_box, nd_collect, nd_set_keys = make_objective_checklist()
    plot_page_layout.addWidget(nd_box)

    # With the campaign, not with the plots: what is feasible is a property of the problem,
    # and the plots only read it. Typed in here for now - the objective already declares
    # its constraints, and deducing them from it is what should eventually fill this box.
    con_box, con_collect, con_set_keys = make_constraints_widget()
    layout.addWidget(con_box)

    trk_box, trk_set_keys = make_trackers_widget()
    layout.addWidget(trk_box)

    # ---- Plots ---------------------------------------------------------------
    plot_box = QGroupBox("Plots")
    plot_layout = QVBoxLayout(plot_box)
    btn_pareto = QPushButton("Plot Pareto")
    btn_hv = QPushButton("Plot HV")
    btn_hvi = QPushButton("Plot HV improvement")
    btn_gain = QPushButton("Score campaign")
    btn_gain_ninit = QPushButton("Plot gain vs n_initial")

    # One button per quantity rather than a mode the plot buttons read: each draws a
    # different figure, and a selector meant the same button produced a hypervolume or a
    # regret depending on a combo three rows away. The last two measure a campaign against
    # the problem's own optimum, so they need HV* to exist - see optimum_label.
    btn_rho = QPushButton("Plot norm. HV")
    btn_rho.setToolTip("HV(n)/HV*, reaching 1 at the optimum. Needs HV*, which "
                       "campaign_optimum computes from a terminal.")
    btn_regret = QPushButton("Plot regret")
    btn_regret.setToolTip("HV* − HV(n), reaching 0 at the optimum, on a log axis so the "
                          "rate a run converges at is readable rather than only where it "
                          "ended up. Needs HV*, which campaign_optimum computes from a "
                          "terminal.")

    # HV*, shown but not computed here. Estimating it is minutes of sampling and refinement
    # against knobs worth watching converge, which is a terminal's job rather than a button
    # that blocks a window - and it is a property of the problem, so it is measured once and
    # reused by every campaign run against that objective, not per selection. What the tab
    # does is read it back, so it is visible whether the two buttons above have anything to
    # divide by at all.
    optimum_label = QLabel("HV*: —")
    optimum_label.setToolTip("The best hypervolume the problem allows. Computed by "
                             "campaign_optimum and stored as optimum.json beside the "
                             "objective; this only reads it back.")

    # Convergence. n_c decides gamma, m_c and every n_tau, and --tol is in the metric's own
    # units, so a campaign whose hypervolume lives on a different scale needs a different
    # one - which is why this cannot sit at a hidden default.
    conv_patience = QSpinBox()
    conv_patience.setRange(2, 500)
    conv_patience.setValue(10)
    conv_patience.setPrefix("patience ")
    conv_patience.setToolTip("Iterations improving by less than the tolerance that mark "
                             "convergence. Too short and a plateau is read as an ending.")
    conv_tol_rel = QLineEdit("1e-4")
    conv_tol_rel.setValidator(QDoubleValidator(0.0, 1.0, 12))
    conv_tol_rel.setMaximumWidth(90)
    conv_tol_rel.setToolTip("Flatness threshold as a fraction of each run's own HV(n0), so "
                            "one setting reads the same on a campaign whose hypervolume "
                            "is 300 and on one where it is 3. Empty falls back to the "
                            "absolute --tol default.")
    tau_edit = QLineEdit("0.5, 0.9, 0.99")
    tau_edit.setMaximumWidth(120)
    tau_edit.setToolTip("Fractions of the achievable gain that n_tau reports the "
                        "evaluation count for. Comma-separated.")

    # Both ask the problem about the records rather than taking them at face value, and
    # both only mean anything with an objective loaded.
    cb_true_obj = QCheckBox("True objective")
    cb_true_obj.setToolTip("Score each observation by the noiseless objective at the "
                           "parameters it used, rather than the value recorded. Only "
                           "meaningful on a simulated campaign, where it puts HV(n) and "
                           "HV* on the same surface.")
    cb_input_feasible = QCheckBox("Input-feasible only")
    cb_input_feasible.setToolTip("Drop from the front every observation whose parameters "
                                 "break the problem's input constraints. HV* never sees "
                                 "such a point, so a handful can double a campaign's "
                                 "hypervolume and push it above the optimum.")
    # One checkbox per grouping key. Ticked keys are what tells records apart, so every
    # box ticked draws each observation on its own and unticking one pools over it. The two
    # switches this replaced were fixed points in that space - Grouped was "untick repeat",
    # Average runs was "untick run" - and between them they made most of it unreachable.
    KEY_TEXT = {
        "parameters": "Parameters — what makes two measurements the same setting. Untick "
                      "it and every setting in a group merges, which is rarely wanted.",
        "run": "Run — the folder a measurement came from. Untick it to pool runs: the "
               "band then shows how differently the optimizer behaves from seed to seed.",
        "strategy": "Strategy — bayesian, sobol, random. Untick it to pool strategies "
                    "together, which is usually only wanted with everything else pooled.",
        "n_initial": "Initial design size — untick it to compare strategies pooled over "
                     "n10/n15/n20 instead of nine separate series.",
        "provenance": "Provenance — real rig or simulated. Untick it and a rig's runs "
                      "average together with a study's, which is rarely meaningful.",
        "technology": "Technology — what produced the measurement, as opposed to what "
                      "chose it.",
    }
    # Display text for a key's checkbox, where it should read as something other than
    # the key itself with underscores swapped for spaces. Independent of _series._FIELD,
    # which spells these same keys for a different reason - looking a value up on an
    # experiment record - so a cosmetic rename here can't silently change what that reads.
    KEY_LABEL = {
        "parameters": "Parameters",
        "run": "Folder",
        "strategy": "Optimizer",
        "n_initial": "Initial Population Size",
        "provenance": "Experiment Type",
        "technology": "Technology",
    }
    # The starting arrangement, dragged from here rather than fixed: --group-by now sends
    # keys in whatever order the list holds them, since a series' label is built by walking
    # that same order (see _series.parse_keys) - so dragging "technology" ahead of
    # "strategy" is what makes a label read "gaas bayesian" instead of "bayesian gaas".
    # It never changes *which* records pool together, which is a plain equality test on the
    # ticked keys and comes out the same whatever order they're listed in.
    DEFAULT_KEY_ORDER = ("provenance", "strategy", "n_initial", "parameters", "technology", "run")
    # Ticked by default: what the campaign was (experiment type, optimizer, design size)
    # and what setting was measured (parameters). Technology and run start unticked -
    # pooling across them is the common case, not the exception.
    DEFAULT_TICKED = {"provenance", "strategy", "n_initial", "parameters"}
    key_items = {}
    key_order_list = QListWidget()
    key_order_list.setFlow(QListView.Flow.LeftToRight)
    key_order_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
    key_order_list.setFrameShape(QFrame.Shape.NoFrame)
    key_order_list.setSpacing(4)
    key_order_list.setFixedHeight(34)
    key_order_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    key_order_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    key_order_list.setToolTip("Drag to reorder. Ticked keys are what tells records apart; "
                              "their order here is only what a series label lists first.")
    for key in DEFAULT_KEY_ORDER:
        item = QListWidgetItem(KEY_LABEL.get(key, key.replace("_", " ")))
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                      | Qt.ItemFlag.ItemIsDragEnabled)
        item.setCheckState(Qt.CheckState.Checked if key in DEFAULT_TICKED
                           else Qt.CheckState.Unchecked)
        # .get, not [key]: DEFAULT_KEY_ORDER is the one list a key is added to, and a new
        # one arriving without a blurb here should cost it a tooltip, not take the
        # whole tab down on a KeyError.
        item.setToolTip(KEY_TEXT.get(key, ""))
        item.setData(Qt.ItemDataRole.UserRole, key)
        key_order_list.addItem(item)
        key_items[key] = item

    def _key_checked(key: str) -> bool:
        return key_items[key].checkState() == Qt.CheckState.Checked

    def _checked_keys_in_order() -> list:
        """The ticked keys, in the list's current (user-dragged) order."""
        return [key_order_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(key_order_list.count())
                if key_order_list.item(i).checkState() == Qt.CheckState.Checked]

    # What a merged point's bar shows - the spread of the repeats behind it, which only
    # exists where a setting was measured more than once in one run. Separate from the band
    # below, which is the spread across the curves of a series.
    err_combo = QComboBox()
    err_combo.addItems(["sem", "std", "minmax"])
    ERRORBAR_TEXT = {
        "sem": "Std. error — how well the point's position is known. Repeats shrink it by √n.",
        "std": "Std. dev. — how much a single measurement scatters. Repeats do not shrink it.",
        "minmax": "Min/max — the plain range of the group's measurements.",
    }
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
    cb_front = QCheckBox("Show Pareto front line")
    cb_front.setChecked(True)
    cb_front.setToolTip("Join the non-dominated points into a dashed front line, on a "
                        "constrained problem as well as an unconstrained one. There the "
                        "line is a guide: a trade-off between two front points is not "
                        "necessarily attainable.")
    # 2-D only. The second front is the union, design points included - a front over the
    # proposals alone would run through points the design had already beaten.
    cb_design_front = QCheckBox("Show initial Pareto front line")
    cb_gt_front = QCheckBox("Show ground-truth Pareto front line")
    cb_gt_front.setChecked(True)
    cb_gt_front.setToolTip("Draw the true front along with the ground truth's cloud. "
                           "A constrained problem has no single front, so there it makes "
                           "no difference.")
    cb_design_front.setToolTip("Also draw the initial design's own front, in grey under "
                               "the campaign's. The pair says what proposing added to "
                               "the dataset the run started from.")
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
    for position, key in enumerate(ERRORBAR_TEXT):
        err_combo.setItemData(position, ERRORBAR_TEXT[key], Qt.ItemDataRole.ToolTipRole)
    for position, key in enumerate(BAND_TEXT):
        band_combo.setItemData(position, BAND_TEXT[key], Qt.ItemDataRole.ToolTipRole)

    def _on_grouping_change() -> None:
        """Keep the two spread controls offered only when something is pooled into them.

        The error bar describes a merged point and the band a merged curve, so each is
        meaningless while its own kind of merging is off - and an enabled control that
        changes nothing is worse than an absent one.
        """
        # The error bar is always available: repeats of a setting always merge, so a
        # campaign that measured any has a bar to show. The band needs runs pooled.
        err_combo.setEnabled(True)
        band_combo.setEnabled(not _key_checked("run"))
        err_combo.setToolTip(ERRORBAR_TEXT[err_combo.currentText()]
                             if err_combo.isEnabled() else "")
        band_combo.setToolTip(BAND_TEXT[band_combo.currentText()]
                              if band_combo.isEnabled() else "")

    # itemChanged also fires on a plain drag reorder in PySide6 - harmless, since this
    # only reads check state, not position.
    key_order_list.itemChanged.connect(lambda _item: _on_grouping_change())
    err_combo.currentTextChanged.connect(lambda _t: _on_grouping_change())
    band_combo.currentTextChanged.connect(lambda _t: _on_grouping_change())
    _on_grouping_change()

    group_layout.addWidget(key_order_list)
    group_layout.addWidget(_row(QLabel("Error bar:"), err_combo,
                                QLabel("Band:"), band_combo))
    group_layout.addWidget(_grouping_note(
        "Records that agree on every ticked key are one group, drawn as its mean. Repeats "
        "of a setting within a run always merge — they are one measurement made twice, and "
        "the error bar is their spread. Every box ticked, the default, then draws one "
        "point per setting per run and one curve per run. Untick run and whole runs "
        "average into one curve or front per series, with the band showing how differently "
        "the optimizer behaves from seed to seed; untick n_initial as well and the design "
        "sizes pool, giving one series per strategy. Drag a key left or right to change "
        "the order a series' label lists it in - that never changes what pools together."))
    plot_page_layout.insertWidget(0, group_box)
    # Scoring is its own row: the first writes gain.json and the second reads it, so
    # they run in that order and neither belongs beside the drawing buttons.
    # Into the axes frame, not this one: they draw against the rows up there.
    # One row: the three hypervolume buttons draw the same trace against different y axes,
    # so they read as variants of each other rather than as separate tools.
    axes_layout.addWidget(_row(btn_pareto, btn_hv, btn_hvi, btn_rho, btn_regret))
    # And with them, the options only these plots read: the front lines are the Pareto
    # plot's own, and the point labels are read by Pareto and the objective landscape.
    # The hypervolume takes none of the three.
    axes_layout.addWidget(_row(cb_numbers, cb_front, cb_design_front))
    # Into the slot reserved under the axis rows, not appended here - see gt_holder.
    gt_row_slot.addWidget(_row(cb_ground, gt_method, gt_samples, gt_spacing, cb_gt_noisy,
                               cb_gt_front))
    plot_layout.addWidget(_row(btn_gain, btn_gain_ninit))
    # The optimum every absolute metric divides by, shown so it is visible whether there is
    # one at all - computing it is campaign_optimum's job, from a terminal. Read by the
    # score and by the HV plots alike, hence its own row above the settings those share.
    plot_layout.addWidget(_row(optimum_label))
    # What n_c is judged by, and the targets n_tau reports against. gamma, m_c and every
    # n_tau column move with these, so they belong beside the button that computes them.
    plot_layout.addWidget(_row(QLabel("Convergence:"), conv_patience,
                               QLabel("tol/HV(n0):"), conv_tol_rel,
                               QLabel("tau:"), tau_edit))
    # Read by the score and the HV plot both, so a table and a curve of the same campaign
    # are scoring the same thing.
    plot_layout.addWidget(_row(cb_true_obj, cb_input_feasible))
    plot_page_layout.addWidget(plot_box)

    # ---- Diagnostics ---------------------------------------------------------
    # The campaign-wide plots that take no axis selection: one row each, as in the
    # original tab's diagnostic frame.
    diag_box = QGroupBox("Diagnostic tools")
    diag_layout = QVBoxLayout(diag_box)
    plot_page_layout.addWidget(diag_box)
    layout.addStretch()
    plot_page_layout.addStretch()

    # ---- Key discovery -------------------------------------------------------

    def _sync_plot_buttons() -> None:
        """Only offer the plots once an objective says what the campaign is.

        Every one of them needs the dimensionality - which axes exist, whether there is
        a front to draw, what the hypervolume is measured over - and that is the
        objective's to declare. Without one the tab has the map's column names and no
        way to tell an objective from a parameter among them, so a plot drawn then would
        be drawn against a guess.
        """
        ready = state["problem"] is not None
        # The two problem-view options and HV* need the objective just as much: all three
        # ask the problem something the records cannot answer.
        buttons = (btn_pareto, btn_hv, btn_hvi, btn_rho, btn_regret, btn_gain,
                   btn_gain_ninit, cb_true_obj, cb_input_feasible)
        for button in buttons:
            button.setEnabled(ready)
        hint = "" if ready else "Load an objective first: it defines how many objectives "                                "the campaign has, and every plot needs that."
        for button in buttons:
            button.setToolTip(hint)
        if ready:
            _show_optimum()

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
        cb_gt_front.setEnabled(on)

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

        Called by the four things that can change what the keys should be: loading an
        objective, changing the objective count, loading a map from a file, and Rebuild
        map now. There is deliberately no button for it. With an objective loaded this
        reads nothing but `problem`, which only those paths alter, so a manual press
        re-derived identical data - and repopulate() resets the combos, so its one visible
        effect was to discard the axes and senses the user had chosen.
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
        objectives_set_keys(problem["objectives"] if problem is not None else [])
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
            settings.objective_path = ""
            post(f"Could not load: {exc}")
            _sync_ground_truth()
            return
        state["problem"] = problem
        settings.objective_path = path
        # The count comes from the problem now, so loading one is what changes it - the
        # radio buttons used to fire this and no longer exist.
        _on_objective_count_change()
        _refresh_keys()
        # The Parameters box now shows the objective's own resolutions, but a map already
        # built was grouped at whatever was there before - typically blank, i.e. compared
        # as measured. Regrouping here would change what the group map says under the
        # user without their asking; saying so and leaving the map alone puts that in
        # their hands, and Rebuild map now is the one action that changes it.
        _note_resolutions_changed()
        _sync_ground_truth()
        _sync_plot_buttons()
        senses = ", ".join(f"{o['label']} ({'min' if o['to_minimize'] else 'max'})"
                           for o in problem["objectives"])
        n_obj = len(problem["objectives"])
        post(f"{n_obj} objective{'' if n_obj == 1 else 's'}: {senses}. "
                           f"ref_point={problem['ref_point']}")

    def _browse_objective() -> None:
        chosen, _ = QFileDialog.getOpenFileName(page, "Choose an objective", obj_edit.text(),
                                                "Python (*.py)")
        if chosen:
            obj_edit.setText(chosen)
            _load_objective(chosen)

    def _edit_objective() -> None:
        path = obj_edit.text().strip()
        if not path:
            post("Type or browse to an objective.py first.")
            return
        _edit_objective_dialog(page, path, on_saved=lambda: _load_objective(path))

    browse.clicked.connect(_browse_objective)
    load_btn.clicked.connect(lambda: _load_objective(obj_edit.text()))
    edit_btn.clicked.connect(_edit_objective)

    # ---- Objective-count wiring -----------------------------------------------

    def _on_objective_count_change() -> None:
        count = _n_objectives()
        is_many = count == "4+"
        is_three = count == "3"
        is_single = count == "1"
        # 4+ has no scatter to draw beyond three axes, so the axis rows give way to the
        # checklist and the Pareto button goes with them - but the hypervolume reads its
        # objectives from that checklist and still has buttons here, so the frame stays.
        axes_rows.setVisible(not is_many)
        btn_pareto.setVisible(not is_many)
        nd_box.setVisible(is_many)
        z_row.setEnabled(not is_many)
        # z is a real objective only with three; with two it colours the points and with
        # one it is the second parameter, so in neither does its sense mean anything.
        z_sense.setEnabled(is_three)
        # z only drives a colour scale with three objectives, or two with a colour key
        # chosen - never with one, where it is a plain parameter axis instead.
        z_center_entry.setEnabled(is_three or count == "2")
        # One objective reads the same three rows as a landscape - the one objective, and
        # the one or two parameters it is drawn over - so only the objective's sense is
        # left live.
        axes_box.setTitle("Objective landscape and best value" if is_single
                          else "Pareto and hypervolume")
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
        btn_hv.setText("Plot best value" if is_single else "Plot HV")
        btn_hvi.setText("Plot improvement" if is_single else "Plot HV improvement")
        btn_pareto.setEnabled(not is_many)
        # Point labels belong to the scatter, which 4+ objectives has none of. The
        # grouping keys stay live either way: the hypervolume still pools by them.
        cb_numbers.setEnabled(not is_many)
        # The key lists mean different things per objective count, so they are rebuilt
        # rather than left showing the previous one's.
        _refresh_keys()

    # ---- Launching -----------------------------------------------------------

    def _rebuild_map(on_done=None, force: bool = False) -> None:
        """Rebuild experiment_map.json + group_map.json from the current selection.

        Asynchronous: the reading happens on a worker thread and `on_done(ok)` is called
        here afterwards, so callers continue from there rather than from a return value.
        Reading a campaign's step records is seconds to minutes - 80 s for a 40-run study,
        nearly all of it file opens - and on the GUI thread that is a window Windows greys
        out as "not responding".

        A second rebuild while one is in flight is refused rather than queued: the two
        would race to write the same scratch map, and the loser's would be what the plots
        then read. The scripts read the map through configs.settings.data_path, which is
        repointed once the build lands.
        """
        finish = on_done or (lambda ok: None)
        if state.get("building"):
            post("Still building the previous map — wait for it to finish.")
            finish(False)
            return
        # Everything the build needs is read from the widgets here, on the GUI thread,
        # and closed over: the worker must not touch a Qt object.
        steps = step_list.checked_paths
        # A reference not otherwise selected still has to be in the map, or its own
        # points would never reach the plot for _reference to draw - so its
        # directories are folded into the roots build_map reads, independently of
        # whether the user also ticked Include on them.
        references = step_list.reference_paths
        roots = sorted(set(steps) | set(references))
        # No roots is a request for the ground truth on its own, as long as there is a
        # ground truth to draw: build_map([]) is an empty map, and a plot over one draws
        # the backdrop and no series. Without one it is just an empty plot, so that still
        # asks for a selection rather than opening a blank figure.
        if not roots and not cb_ground.isChecked():
            post("Select at least one step in the Steps window.")
            finish(False)
            return
        resolutions = par_collect()
        state["building"] = True
        post(f"Building the campaign map from {len(roots)} director"
             f"{'y' if len(roots) == 1 else 'ies'}...")

        def _work():
            """The slow half: reading every selected step record. No Qt in here.

            Unless it can be skipped. The stamp says what the map depends on - the
            records' own mtimes and sizes, the selection, the resolutions, the schema the
            code writes - and a cached map taken under an identical stamp is the same map.
            Two large reads instead of thousands of small ones.
            """
            stamp = map_stamp(roots, references=references, resolutions=resolutions)
            cache = workspace.cache_dir()
            cached = None if cache is None else cache / stamp_digest(stamp)
            if cached is not None and not force and (cached / "stamp.json").exists():
                try:
                    if json.loads((cached / "stamp.json").read_text(encoding="utf-8")) == stamp:
                        return (json.loads((cached / "experiment_map.json").read_text(encoding="utf-8")),
                                json.loads((cached / "group_map.json").read_text(encoding="utf-8")),
                                True, stamp)
                except (OSError, ValueError):
                    # An unreadable or half-written cache entry is not worth failing over:
                    # rebuilding is always correct, only slower.
                    pass
            exp_map = build_map(roots, reference_roots=references)
            groups = build_groups(exp_map, resolutions)
            if cached is not None:
                cached.mkdir(parents=True, exist_ok=True)
                (cached / "experiment_map.json").write_text(json.dumps(exp_map, indent=2),
                                                            encoding="utf-8")
                (cached / "group_map.json").write_text(json.dumps(groups, indent=2),
                                                       encoding="utf-8")
                # Written last, so a stamp on disk always has a complete map beside it.
                (cached / "stamp.json").write_text(json.dumps(stamp, indent=2),
                                                   encoding="utf-8")
            return exp_map, groups, False, stamp

        def _apply(result) -> None:
            """Back on the GUI thread with whatever the worker produced."""
            state["building"] = False
            if isinstance(result, BaseException):
                post(f"Map build failed: {result}")
                finish(False)
                return
            exp_map, groups, reused, stamp = result
            state["map"], state["groups"] = exp_map, groups
            state["source"] = "selection"
            # What this map was built from. Regroup reuses it rather than taking a fresh
            # one, so a regrouped map is cached as coming from the records it really came
            # from - see _regroup.
            state["stamp"] = stamp
            # A plot is a separate process reading configs.settings.data_path, so the map
            # has to exist somewhere on disk for it. That somewhere is a scratch directory
            # for the session, not the user's data tree - Save map is how a copy gets kept.
            _write_map(_scratch)
            configs_settings.set_data_path(_scratch)
            if not roots:
                post("No steps selected — the ground truth will be drawn on its own")
                finish(True)
                return
            series = len({e["experiment_type"] for e in exp_map["experiments"]})
            # Said out loud on purpose: a reused map that is somehow wrong has to be
            # visible, not silent. Rebuild map now is the way to force one either way.
            how = "reused, unchanged since it was built" if reused else "held in memory"
            message = (f"{len(exp_map['experiments'])} observations from "
                       f"{len(roots)} selected director"
                       f"{'y' if len(roots) == 1 else 'ies'}, {series} series, {how}")
            if references:
                message += f", {len(references)} reference director" \
                           f"{'y' if len(references) == 1 else 'ies'}"
            post(message)
            finish(True)

        run_off_thread(_work, _apply)

    def _write_map(out_dir) -> Path:
        """Write the map held in memory into `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment_map.json").write_text(
            json.dumps(state["map"], indent=2), encoding="utf-8")
        (out_dir / "group_map.json").write_text(
            json.dumps(state["groups"], indent=2), encoding="utf-8")
        return out_dir

    def _cache_regrouped(resolutions: dict) -> None:
        """Store the regrouped map under the records it was built from.

        Nothing to store when the map came from a file rather than a build, since then no
        fingerprint of the records stands behind it.
        """
        cache = workspace.cache_dir()
        if cache is None or not state.get("stamp"):
            return
        stamp = {**state["stamp"],
                 "resolutions": {str(k): v for k, v in sorted(resolutions.items())}}
        entry = cache / stamp_digest(stamp)
        try:
            entry.mkdir(parents=True, exist_ok=True)
            (entry / "experiment_map.json").write_text(json.dumps(state["map"], indent=2),
                                                       encoding="utf-8")
            (entry / "group_map.json").write_text(json.dumps(state["groups"], indent=2),
                                                  encoding="utf-8")
            (entry / "stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 - a cache that cannot be written is not fatal
            post(f"Could not cache the regrouped map: {exc}")

    def _regroup() -> None:
        """Re-group the map held in memory at the current resolutions.

        The cheap half of a rebuild: which observations count as one setting depends on
        the resolutions, the observations themselves do not - so this needs no step
        record read again. Seconds against minutes, which is why it is a button of its
        own rather than something Rebuild map now is the only way to reach.

        Applied to whatever is in memory, loaded map included: a saved grouping was made
        at whatever resolutions were set then, and leaving it would mean the Parameters
        box reads one thing while the plots group by another.

        Cached under the fingerprint the map was *built* with, not a fresh one. Pressing
        Regroup asserts that the map in memory still stands - but only now. Taking a new
        fingerprint would extend that assertion into the future: a record rewritten since
        the build would be recorded as though the map already reflected it, and the next
        rebuild would reuse a map that does not. Keeping the original fingerprint says
        what is true - these records, in that state, grouped this way - so a rebuild after
        a record changes still misses and rebuilds.
        """
        if not state["map"]:
            post("Nothing to regroup — build a map first.")
            return
        resolutions = par_collect()
        state["groups"] = build_groups(state["map"], resolutions)
        _write_map(_scratch)
        configs_settings.set_data_path(_scratch)
        _cache_regrouped(resolutions)
        post(f"Regrouped into {len(state['groups'])} groups, "
             f"{len(resolutions)} parameters by resolution")

    def _note_resolutions_changed() -> None:
        """Say that the Parameters box and the map now disagree, when they do.

        Nothing regroups on its own - not loading an objective, not typing a resolution.
        One action changes what the plots see, and it is Rebuild map now; anything else
        would move the ground under a group map the user is in the middle of reading.

        Only with a map already built: with none there is nothing to be out of step with,
        and the next build will use whatever the boxes hold by then.
        """
        if not state["map"]:
            return
        post("Resolutions changed — the map is still grouped at the previous ones. "
             "Press Regroup to group by these, or Rebuild map now to re-read the "
             "records as well.")

    def _with_shown_map(on_ready) -> None:
        """Call `on_ready(True)` with whatever map is already held, building one only if
        there is none.

        For the viewers, which exist to show the current state rather than to produce a
        fresh one. Reading a large campaign's records is the expensive thing the GUI
        does, and doing it to redraw a JSON the tab is already holding is work for
        nothing - Rebuild map now is what asks for a new one.
        """
        if state["map"]:
            on_ready(True)
            return
        _rebuild_map(on_ready)

    def _with_map(on_ready) -> None:
        """Call `on_ready(ok)` with a map to work from: a loaded one as it is, else one
        rebuilt from the selection.

        A loaded map is left alone so that plotting it does not quietly replace it with
        whatever happens to be ticked in the tree; Rebuild map now is how you go back to
        the selection. That case answers immediately; a rebuild answers when its worker
        finishes, which is why this hands the caller a callback instead of a bool.
        """
        if state["source"] == "loaded" and state["map"]:
            on_ready(True)
            return
        _rebuild_map(on_ready)

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
        # Loaded from a file, so there is no record fingerprint behind it and a regroup
        # of it has nothing honest to cache under.
        state["stamp"] = None
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

    def _launch(script: str, *extra, verbose: bool = False) -> None:
        """Make sure a map exists, then run one of the analysis modules against it.

        The map may take a worker thread and a minute to build, so the launch is what
        happens once it lands rather than the next line here. That wait is also the one
        window in which Stop plots has nothing to terminate, hence the token: pressed
        while the map builds, it has to reach the plot that has not started yet.

        `verbose` is for the scripts whose output *is* the result - see _run_script.
        """
        post(f"Preparing {script}...")
        token = stop_token()

        def _ready(ok) -> None:
            if not ok:
                return
            if stopped_since(token):
                post(f"{script} was dropped — Stop plots was pressed while its map built.")
                return
            _run_script(script, *extra, verbose=verbose)

        _with_map(_ready)

    def _run_script(script: str, *extra, verbose: bool = False, after=None) -> None:
        module = f"{MODULES}.{script}"
        proc = launch_analysis(module, *extra)
        # For a plot, the log says what is running and how it ended and nothing more: the
        # result is the window that opens, and a script that prints a table would bury
        # every other message under it. Output is collected rather than posted, so a run
        # that succeeds stays quiet and one that fails can still say why - an exit code
        # alone leaves nothing to act on, and the reason is part of how it ended.
        #
        # A scoring script has no window. Its table is the whole result, and collected-
        # and-discarded meant pressing Score campaign produced no numbers anywhere the
        # GUI could show them. Those pass verbose=True and are echoed line by line.
        output = []

        def _line(line: str) -> None:
            output.append(line)
            if verbose:
                post(f"    {line}")

        def _failed(codes) -> None:
            post(f"{script} exited with {codes[0]}.")
            # Already echoed above, so repeating the tail would print it twice.
            if not verbose:
                for line in output[-4:]:
                    post(f"    {line}")
            if not output:
                post("    (it printed nothing - see its console.)")
            if codes[0] == 2:
                post("    Exit 2 is a constraint that would not parse, or a metric asked "
                     "for without the optimum it needs.")

        def _done() -> None:
            post(f"{script} finished.")
            if after is not None:
                after()

        watch([proc],
              on_start=lambda: post(f"Running {script}..."),
              on_done=_done,
              on_fail=_failed,
              on_output=_line)

    def _group_by_args(bands: bool = True, errorbar: bool = True) -> list:
        """The ticked keys, and the two spread controls they make meaningful.

        Every plot takes --group-by. --band is only understood by the ones that average
        whole curves, hence `bands`; --errorbar only by the ones that draw a merged
        point, hence `errorbar` - plot_hypervolume and plot_evolution draw neither
        merged points nor (for evolution) averaged curves, so both flags would be
        arguments their own parser has never heard of.
        """
        args = [flag for key in _checked_keys_in_order() for flag in ("--group-by", key)]
        if errorbar:
            args += ["--errorbar", err_combo.currentText()]
        if bands and not _key_checked("run"):
            args += ["--band", band_combo.currentText()]
        return args

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
        extra += _group_by_args(bands=False)
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
            extra += _group_by_args(bands=False)
            if cb_numbers.isChecked():
                extra.append("--show-numbers")
            extra += ["--x", x, "--y", y, "--z", z,
                      "--xlabel", x_entry.text(), "--ylabel", y_entry.text(),
                      "--zlabel", z_entry.text()]
            if z_center_entry.text():
                extra += ["--zcenter", z_center_entry.text()]
            extra += _sense_args((x, x_sense), (y, y_sense), (z, z_sense))
            _launch("plot_pareto_3d", *extra)
            return
        extra += _group_by_args()
        if cb_numbers.isChecked():
            extra.append("--show-numbers")
        extra += ["--x", x, "--y", y,
                  "--xlabel", x_entry.text(), "--ylabel", y_entry.text()]
        if z:
            extra += ["--z", z, "--zlabel", z_entry.text()]
            if z_center_entry.text():
                extra += ["--zcenter", z_center_entry.text()]
        extra += _sense_args((x, x_sense), (y, y_sense)) + _ground_truth_args()
        # Said either way rather than left to the plot's own judgement: the checkbox is
        # the answer, constrained problem or not.
        extra += ["--front-line", "always" if cb_front.isChecked() else "never"]
        if cb_design_front.isChecked():
            extra += ["--front-scope", "initial-vs-all"]
        # Only here: plot_pareto_3d draws no ground-truth front and would not take it.
        if cb_ground.isChecked():
            extra += ["--gt-front", "always" if cb_gt_front.isChecked() else "never"]
        _launch("plot_pareto_2d", *extra)

    def _metric_objective_args() -> list | None:
        """How the campaign's metric is defined, as flags: the objectives it is measured
        over and which of them are maximized.

        Shared by every script scoring a campaign rather than drawing its points -
        plot_hypervolume and campaign_gain - so the two always score the same thing.
        None means the selection is incomplete and the caller should not launch; the
        reason has already been posted.
        """
        extra = []
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
        return extra

    def _problem_view_args() -> list:
        """The two flags that ask the problem about the records rather than taking them at
        face value. Shared by the score and the hypervolume plot, so a table and a curve of
        the same campaign are never scoring different things."""
        args = []
        if cb_true_obj.isChecked():
            args.append("--true-objective")
        if cb_input_feasible.isChecked():
            args.append("--input-feasible")
        return args

    def _convergence_args() -> list:
        """--patience, --tol-rel and the --tau targets, as the scoring script takes them.

        A blank tolerance is left out rather than sent as zero: campaign_gain then keeps
        its own absolute --tol, where zero would mean no improvement is ever flat and no
        run ever converges. A tau list that will not parse is reported and the defaults
        stand, since a typo there should not silently rescore the campaign against targets
        nobody chose.
        """
        args = ["--patience", str(conv_patience.value())]
        tol = conv_tol_rel.text().strip()
        if tol:
            args += ["--tol-rel", tol]
        raw = tau_edit.text().strip()
        if raw:
            try:
                taus = [float(part) for part in raw.replace(";", ",").split(",")
                        if part.strip()]
            except ValueError:
                taus = []
            if not taus or any(not 0 < t <= 1 for t in taus):
                post(f"Could not read the tau targets from {raw!r} — they must be "
                     f"comma-separated fractions in (0, 1]. Using the defaults.")
            else:
                args += [flag for t in taus for flag in ("--tau", str(t))]
        return args

    def _show_optimum() -> None:
        """Report whether an HV* exists for the loaded objective, and what it is.

        Read off optimum.json every time rather than remembered: that file is what the
        plot scripts themselves consult, so showing anything else could say an optimum is
        available when the scripts would not find one.

        Looked for beside the objective, which is where campaign_optimum puts it and the
        first place the scripts look. HV* describes the problem, not a selection of runs
        against it, so one estimate serves every campaign on that objective.
        """
        objective_path = obj_edit.text().strip()
        if not objective_path:
            optimum_label.setText("HV*: —")
            return
        path = os.path.join(os.path.dirname(os.path.abspath(objective_path)),
                            "optimum.json")
        try:
            with open(path, encoding="utf-8") as file:
                cached = json.load(file)
            value = cached["hv_star"]
            keys = ", ".join(cached.get("context", {}).get("objectives", []))
        except (OSError, ValueError, KeyError):
            optimum_label.setText("HV*: — (run campaign_optimum)")
            optimum_label.setToolTip(
                "No optimum.json beside this objective, so Plot norm. HV and Plot regret "
                "have nothing to measure against. From a terminal:\n\n"
                "  python -m pybo_gui.modules.bayesian_campaign_analysis"
                ".campaign_optimum \\\n"
                f"      --ground-truth {objective_path} --x <obj> --y <obj>\n\n"
                "Keep --refine above 0: sampling alone leaves HV* below what a campaign "
                "reaches, and its regret negative.")
            return
        optimum_label.setText(f"HV* = {value:.6g}  ({keys})")
        optimum_label.setToolTip(f"Read from {path}. Computed by campaign_optimum; this "
                                 f"only reads it back.")

    def _plot_hypervolume(improvement: bool, metric: str = "hv") -> None:
        """The hypervolume trace, on one of three y axes.

        `metric` is the flag plot_hypervolume takes: "hv", "normalized" or "regret". The
        last two need an optimum; the script refuses with exit 2 and says so in the log
        rather than drawing against a guess, so nothing is checked here - the check that
        matters is the one the script makes against the optimum.json it will actually read.
        """
        objectives = _metric_objective_args()
        if objectives is None:
            return
        extra = _constraint_args() + (["--improvement"] if improvement else []) + objectives
        extra += _group_by_args(errorbar=False)
        # Grouped exactly like Plot HV, by whatever the boxes say - no override here.
        # Ticking off `run` while --improvement is on averages per-step gains across
        # runs, which the script refuses (exit 2) rather than draw a misleading curve;
        # _failed already surfaces that refusal's own message, which says to tick
        # `run` or drop --improvement, so the two buttons never silently disagree on
        # what "as grouped" means.
        # Where the hypervolume is measured from. Sent whenever an objective is loaded,
        # regardless of the Ground truth checkbox - that one is about drawing the
        # sampled surface, a separate question from having a fixed reference point, and
        # a moving one would make a run's own hypervolume depend on which other runs
        # happen to be selected alongside it.
        if state["problem"] is not None:
            extra += ["--ground-truth", obj_edit.text()]
        extra += ["--metric", metric]
        # Where Compute HV* left its estimate. The plot reads the map from the scratch
        # directory but the optimum from the selection's cache entry, so it has to be told
        # the second - the two are deliberately not the same place.
        extra += ["--score-dir", _gain_dir()]
        extra += _problem_view_args()
        _launch("plot_hypervolume", *extra)

    def _gain_dir() -> str:
        """Where a campaign-level score and HV* are written: this selection's cache entry.

        The workspace cache, under the same digest the map itself is cached at, because
        both files answer a question asked *of a selection* - which arms, averaged how,
        against which optimum - and the selection is exactly what that digest fingerprints.
        Score the same set twice and the second run replaces the first; score a different
        set and it lands somewhere else, instead of overwriting a report about other runs
        that happened to share a directory.

        Not the campaign root: writing a report into the tree the records live in leaves a
        gain.json among the data that nothing there produced. Not the scratch directory
        either - that is rebuilt from the selection every time and does not outlive the
        session, so a score written there is lost when the GUI closes.

        The per-run scores are a separate matter and do not come here. campaign_gain writes
        those beside each run, where they belong: they describe that run, plot_gain_vs_ninitial
        reads them back across selections the cache digest would keep apart, and
        clear_cache deletes cache contents as things rebuilt on demand, which a score is
        not.

        Falls back to the campaign root, then the scratch, when there is no workspace or no
        map has been built - both are cases with nowhere better to put it.
        """
        cache = workspace.cache_dir()
        if cache is not None and state.get("stamp"):
            entry = cache / stamp_digest(state["stamp"])
            try:
                entry.mkdir(parents=True, exist_ok=True)
                return str(entry)
            except OSError:  # noqa: BLE001 - fall through to somewhere writable
                pass
        return step_list.root or str(_scratch)

    def _score_campaign() -> None:
        """Reduce every run to gamma / n_tau / n_c, scoring the same metric the
        hypervolume plot draws. Prints its table into the log and writes gain.json
        beside the runs, which is what the sensitivity plot below reads."""
        objectives = _metric_objective_args()
        if objectives is None:
            return
        extra = _constraint_args() + objectives + ["--out-dir", _gain_dir()]
        # The same fixed reference the hypervolume plot uses, and for the same reason:
        # gamma is a ratio of hypervolumes, so a reference that moves with the selection
        # would change a run's score depending on what was plotted beside it.
        if state["problem"] is not None:
            extra += ["--ground-truth", obj_edit.text()]
        extra += _convergence_args() + _problem_view_args()
        # verbose: the table this prints is the whole result. There is no window to look
        # at, so collected-and-discarded output means pressing the button produces nothing.
        _launch("campaign_gain", *extra, verbose=True)

    def _plot_gain_vs_ninitial() -> None:
        """Gain and cost against the initial design's size.

        Takes no run list: the plot reads the per-run scores for whatever the map holds,
        and the map has just been rebuilt from the ticked selection by _launch. So the
        selection drives it, and scoring a campaign once leaves every run of it plottable
        in any later combination.
        """
        _launch("plot_gain_vs_ninitial")

    # The viewers rebuild first, so what they show is the current selection rather than
    # whatever was last written.
    # Rebuild map now always rebuilds: it is the way out when a cached map is
    # somehow wrong, so it must not itself consult the cache.
    btn_regroup.clicked.connect(lambda: _regroup())
    btn_rebuild.clicked.connect(
        lambda: _rebuild_map(lambda ok: ok and _refresh_keys(), force=True))
    # The viewers show what is held rather than rebuilding to show it; only an empty tab
    # has to build first, and then the dialog opens when the map lands rather than when
    # the click returns.
    btn_view_exp.clicked.connect(lambda: _with_shown_map(
        lambda ok: ok and _view_json_dialog(page, state["map"], "Experiment map")))
    btn_view_grp.clicked.connect(lambda: _with_shown_map(
        lambda ok: ok and _view_group_map_dialog(page, state["groups"], state["map"])))
    btn_save_map.clicked.connect(_save_map)
    btn_load_map.clicked.connect(_load_map)

    btn_pareto.clicked.connect(_plot_pareto)
    btn_hv.clicked.connect(lambda: _plot_hypervolume(False))
    btn_hvi.clicked.connect(lambda: _plot_hypervolume(True))
    btn_rho.clicked.connect(lambda: _plot_hypervolume(False, "normalized"))
    btn_regret.clicked.connect(lambda: _plot_hypervolume(False, "regret"))
    btn_gain.clicked.connect(lambda: _score_campaign())
    btn_gain_ninit.clicked.connect(lambda: _plot_gain_vs_ninitial())

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
            _launch(s, *(_group_by_args(bands=a, errorbar=False) if g else [])))
        diag_layout.addWidget(_row(label, button))

    # An objective usually sits with the tutorial that produced the data, so offer the
    # first one found under the selected root rather than leaving the field blank.
    if step_list.root:
        guess = next(Path(step_list.root).glob("**/objective.py"), None)
        if guess is not None:
            obj_edit.setText(str(guess))

    _on_objective_count_change()
    _sync_ground_truth()
    _sync_plot_buttons()
    return page, plot_page
