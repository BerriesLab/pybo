"""The ground-truth tab: fit a polynomial surrogate on a selected set of experiments.

Selection itself is the Steps window's job, same as the campaign tab uses it - not a
second picker here. This tab turns that selection into an experiment map (the same
build_experiment_map.build_map the campaign plots read, cached separately so the two
never overwrite each other) and then into a
pybo.ground_truth.build_polynomial_gt command line, the same way tab_campaign turns a
selection into an analysis command - the script stays runnable from a terminal with the
same flags, this tab only saves typing them.

The objective is read from settings.objective_path rather than loaded again here: the
constructor tab already has a Browse/Load row for it, and per-parameter resolutions and
the file to paste the result into are the same objective either way.
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QPlainTextEdit, QPushButton, QSpinBox, QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from pybo_gui.configs import ground_truth as gt_store
from pybo_gui.configs import workspace
from pybo_gui.gui.launchers import launch_analysis, run_off_thread, watch
from pybo_gui.gui.message_log import post
from pybo_gui.gui.widgets import make_objective_checklist
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import (
    build_map, map_stamp, stamp_digest)
from pybo_gui.modules.bayesian_campaign_analysis.objective_loader import (
    load_objective, problem_definition)


def _build_gt_map(roots: list) -> tuple:
    """(experiment map, path to its JSON on disk), read from gt_map_cache when a
    workspace has an unchanged one, built fresh otherwise.

    Mirrors tab_campaign._rebuild_map's own cache-or-build shape, against
    workspace.gt_map_cache_dir() instead of the campaign's cache_dir() - "same method,
    files that don't overwrite" is the whole reason there are two.
    """
    stamp = map_stamp(roots)
    cache = workspace.gt_map_cache_dir()
    if cache is not None:
        entry = cache / stamp_digest(stamp)
        if (entry / "stamp.json").exists():
            try:
                if json.loads((entry / "stamp.json").read_text(encoding="utf-8")) == stamp:
                    exp_map = json.loads((entry / "experiment_map.json").read_text(encoding="utf-8"))
                    return exp_map, entry / "experiment_map.json"
            except (OSError, ValueError):
                # An unreadable or half-written cache entry is not worth failing over:
                # rebuilding is always correct, only slower.
                pass
        exp_map = build_map(roots)
        entry.mkdir(parents=True, exist_ok=True)
        map_path = entry / "experiment_map.json"
        map_path.write_text(json.dumps(exp_map, indent=2), encoding="utf-8")
        # Written last, so a stamp on disk always has a complete map beside it.
        (entry / "stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        return exp_map, map_path
    # No workspace, nowhere durable to cache: written to a fixed scratch path instead
    # of a throwaway temp file, so a caller (or a curious user) can still find it.
    exp_map = build_map(roots)
    map_path = Path(tempfile.gettempdir()) / "ground_truth_experiment_map.json"
    map_path.write_text(json.dumps(exp_map, indent=2), encoding="utf-8")
    return exp_map, map_path


def _populate_preview(tree: QTreeWidget, records: list) -> None:
    tree.setSortingEnabled(False)
    tree.clear()
    for record in records:
        start = record.get("start_time")
        when = datetime.fromtimestamp(start).isoformat() if start else "—"
        params = record.get("parameters") or {}
        params_text = ", ".join(f"{k}={v}" for k, v in sorted(params.items())) or "(none)"
        item = QTreeWidgetItem([when, params_text, Path(record["path"]).name])
        item.setToolTip(2, record["path"])
        tree.addTopLevelItem(item)
    # Populate before sorting is enabled, or Qt re-sorts on every insertion instead of
    # once - see _view_group_map_dialog's own note on the same gotcha. Sorted by
    # Datetime ascending to start, same column build_map itself ordered these by, so
    # rows that ran close together - worth checking as replicates of one setting - sit
    # next to each other; click Parameters instead to group by that.
    tree.header().setSortIndicatorShown(True)
    tree.setSortingEnabled(True)
    tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)


def _show_result(parent: QWidget, text: str,
                  title: str = "Ground truth — paste into objective.py") -> None:
    """Mirrors tab_campaign._view_json_dialog's shape: a non-modal QDialog owned by
    `parent`, so Qt's own parent-child ownership keeps it alive."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(700, 600)
    v = QVBoxLayout(dlg)
    txt = QPlainTextEdit()
    txt.setReadOnly(True)
    txt.setPlainText(text or "Nothing produced — see the log.")
    v.addWidget(txt)
    copy = QPushButton("Copy to clipboard")
    copy.clicked.connect(lambda: (QApplication.clipboard().setText(txt.toPlainText()),
                                  post("Copied to clipboard.")))
    v.addWidget(copy)
    dlg.show()


def build(step_list, settings) -> QWidget:
    saved = gt_store.get_state()
    # Which map the last "Build experiment map" produced, read by "Build ground truth" -
    # a plain dict rather than a bare variable so the nested closures below can write it.
    # obj_checklist_path is the objective the HV* checklist was last populated from, so a
    # tab switch that finds nothing changed does not re-import and re-run the objective
    # module (torch and all) just to redraw the same rows.
    state = {"map_path": None, "obj_checklist_path": None}

    class _Tab(QWidget):
        """Refreshes the objective label and HV* checklist whenever this tab becomes
        visible, so a reload in the constructor tab shows up here without a cross-tab
        signal."""

        def showEvent(self, event):
            _refresh_objective_label()
            _refresh_hv_checklist()
            super().showEvent(event)

    page = _Tab()
    layout = QVBoxLayout(page)

    obj_box = QGroupBox("Objective")
    obj_layout = QVBoxLayout(obj_box)
    objective_label = QLabel()
    objective_label.setWordWrap(True)

    def _refresh_objective_label() -> None:
        path = settings.objective_path
        objective_label.setText(f"Objective: {path}" if path else
                                "No objective loaded — load one in the "
                                "“Bayesian campaign constructor” tab.")

    obj_layout.addWidget(objective_label)
    layout.addWidget(obj_box)
    _refresh_objective_label()

    map_box = QGroupBox("Experiment map")
    map_layout = QVBoxLayout(map_box)
    map_note = QLabel("Tick the folders to fit on in the Steps window, then "
                      "build the map here — same selection mechanism as the "
                      "campaign tab, a separate map so the two never overwrite "
                      "each other.")
    map_note.setWordWrap(True)
    map_layout.addWidget(map_note)
    build_map_btn = QPushButton("Build experiment map")
    map_layout.addWidget(build_map_btn)

    tree = QTreeWidget()
    tree.setColumnCount(3)
    tree.setHeaderLabels(["Datetime", "Parameters", "Folder"])
    tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    map_layout.addWidget(tree)
    layout.addWidget(map_box, stretch=1)

    def _on_map_built(result) -> None:
        if isinstance(result, BaseException):
            post(f"Could not build the map: {result}")
            return
        exp_map, map_path = result
        state["map_path"] = map_path
        _populate_preview(tree, exp_map["experiments"])
        post(f"Ground-truth map built: {len(exp_map['experiments'])} observation(s).")

    def _on_build_map() -> None:
        roots = step_list.checked_paths
        if not roots:
            post("Tick at least one folder in the Steps window first.")
            return
        post(f"Building the ground-truth experiment map from {len(roots)} folder"
             f"{'' if len(roots) == 1 else 's'}...")
        run_off_thread(lambda: _build_gt_map(roots), _on_map_built)

    build_map_btn.clicked.connect(_on_build_map)

    settings_box = QGroupBox("Fit settings")
    settings_row = QWidget()
    settings_row_layout = QHBoxLayout(settings_row)
    settings_row_layout.setContentsMargins(0, 0, 0, 0)
    degree_spin = QSpinBox()
    degree_spin.setRange(1, 10)
    degree_spin.setValue(saved["degree"])
    degree_spin.setPrefix("degree ")
    positive_cb = QCheckBox("Positive (log-fit objectives)")
    positive_cb.setChecked(saved["positive"])
    source_combo = QComboBox()
    source_combo.addItems(["all", "initial", "proposed"])
    source_combo.setCurrentText(saved["source"])
    for w in (degree_spin, positive_cb, QLabel("Source:"), source_combo):
        settings_row_layout.addWidget(w)
    settings_row_layout.addStretch()
    QVBoxLayout(settings_box).addWidget(settings_row)
    layout.addWidget(settings_box)

    build_btn = QPushButton("Build ground truth")
    layout.addWidget(build_btn)

    def _run_build(map_path: Path, objective_path: str) -> None:
        paste_path = ((workspace.get_workspace() or Path(tempfile.gettempdir()))
                     / "ground_truth_paste.txt")
        args = ["--map", str(map_path), "--objective", objective_path,
                "--degree", str(degree_spin.value()), "--source", source_combo.currentText(),
                "--paste-out", str(paste_path)]
        if positive_cb.isChecked():
            args.append("--positive")
        proc = launch_analysis("pybo.ground_truth.build_polynomial_gt", *args)
        # Collected, not echoed: the coefficient table and R2 the script prints are
        # already in the result dialog's paste-out, in the readable shape meant to be
        # pasted - repeating them line by line in the log would just be noise. Kept
        # around only so a failure has something to show.
        output = []

        def _done() -> None:
            text = paste_path.read_text(encoding="utf-8") if paste_path.exists() else ""
            _show_result(page, text)
            post("Ground truth built.")

        def _failed(codes) -> None:
            post(f"build_polynomial_gt exited with {codes[0]}.")
            for line in output[-4:]:
                post(f"    {line}")
            if not output:
                post("    (it printed nothing - see its console.)")

        watch([proc],
              on_start=lambda: post("Building ground truth..."),
              on_done=_done,
              on_fail=_failed,
              on_output=output.append)

    def _on_build() -> None:
        objective_path = settings.objective_path
        if not objective_path:
            post("Load an objective in the constructor tab first.")
            return
        if state["map_path"] is None:
            post("Build the experiment map first.")
            return
        gt_store.set_state(degree=degree_spin.value(), positive=positive_cb.isChecked(),
                           source=source_combo.currentText())
        _run_build(state["map_path"], objective_path)

    build_btn.clicked.connect(_on_build)

    hv_box = QGroupBox("Maximum HV (HV*)")
    hv_layout = QVBoxLayout(hv_box)
    hv_note = QLabel("The best hypervolume the loaded objective allows, by dense "
                     "Sobol sampling and local refinement - the same HV* the campaign "
                     "tab's “Plot norm. HV” and “Plot regret” need. Ticks and senses "
                     "below default to the objective's own declared ones; unticking an "
                     "objective drops it from the volume rather than changing its sense. "
                     "Shown here only - nothing is written to disk; run campaign_optimum "
                     "from a terminal instead if you want optimum.json saved beside the "
                     "objective for the campaign tab to read back.")
    hv_note.setWordWrap(True)
    hv_layout.addWidget(hv_note)

    nd_box, nd_collect, nd_set_keys = make_objective_checklist()
    hv_layout.addWidget(nd_box)

    hv_settings_row = QWidget()
    hv_settings_layout = QHBoxLayout(hv_settings_row)
    hv_settings_layout.setContentsMargins(0, 0, 0, 0)
    samples_spin = QSpinBox()
    samples_spin.setRange(1, 10_000_000)
    samples_spin.setSingleStep(4096)
    samples_spin.setValue(65536)
    samples_spin.setPrefix("samples ")
    samples_spin.setToolTip("Quasi-random samples of the parameter box. Raise it until "
                            "the convergence table's trailing gain is a fraction of a "
                            "percent.")
    batch_spin = QSpinBox()
    batch_spin.setRange(1, 1_000_000)
    batch_spin.setSingleStep(1024)
    batch_spin.setValue(4096)
    batch_spin.setPrefix("batch ")
    batch_spin.setToolTip("Samples per batch - sets how finely the convergence table "
                          "is reported.")
    refine_spin = QSpinBox()
    refine_spin.setRange(0, 50)
    refine_spin.setValue(6)
    refine_spin.setPrefix("refine ")
    refine_spin.setToolTip("Rounds of local refinement after sampling (0 = off). "
                           "Sampling alone tends to leave HV* below what a campaign "
                           "reaches; refinement pushes the estimate onto the front.")
    for w in (samples_spin, batch_spin, refine_spin):
        hv_settings_layout.addWidget(w)
    hv_settings_layout.addStretch()
    hv_layout.addWidget(hv_settings_row)

    hv_btn = QPushButton("Compute HV*")
    hv_layout.addWidget(hv_btn)
    layout.addWidget(hv_box)

    def _refresh_hv_checklist() -> None:
        path = settings.objective_path
        if not path:
            nd_set_keys([])
            state["obj_checklist_path"] = None
            return
        if path == state["obj_checklist_path"]:
            return
        try:
            problem = problem_definition(load_objective(path))
        except (Exception, SystemExit):  # noqa: BLE001 - a broken objective must not
            # kill the tab; unlike tab_campaign's explicit Load button, this runs on
            # every tab switch, so it stays quiet rather than posting on each one.
            nd_set_keys([])
            state["obj_checklist_path"] = None
            return
        labels = [o["label"] for o in problem["objectives"]]
        senses = {o["label"]: o["to_minimize"] for o in problem["objectives"]}
        nd_set_keys(labels, senses)
        state["obj_checklist_path"] = path

    def _run_compute_hv(objective_path: str, chosen: list) -> None:
        args = ["--ground-truth", objective_path]
        for key, _is_max in chosen:
            args += ["--objective", key]
        args += [flag for key, is_max in chosen if is_max
                 for flag in ("--maximize", key)]
        args += ["--samples", str(samples_spin.value()), "--batch", str(batch_spin.value()),
                 "--refine", str(refine_spin.value()), "--no-save"]
        proc = launch_analysis(
            "pybo_gui.modules.bayesian_campaign_analysis.campaign_optimum", *args)
        # Streamed to the log as it prints, unlike _run_build's output: the convergence
        # and refinement tables are the whole point of watching this one run, per
        # campaign_optimum's own docstring. Also collected, so the result window can
        # show the full report rather than just the final number.
        output = []

        def _capture(line: str) -> None:
            output.append(line)
            post(line)

        def _done() -> None:
            text = "\n".join(output) if output else "Nothing produced — see the log."
            _show_result(page, text, title="Maximum HV (HV*)")
            post("HV* computed.")

        def _failed(codes) -> None:
            post(f"campaign_optimum exited with {codes[0]}.")
            if not output:
                post("    (it printed nothing - see its console.)")

        watch([proc],
              on_start=lambda: post("Computing HV* - this can take a while; watch the "
                                    "log for the convergence table..."),
              on_done=_done,
              on_fail=_failed,
              on_output=_capture)

    def _on_compute_hv() -> None:
        objective_path = settings.objective_path
        if not objective_path:
            post("Load an objective in the constructor tab first.")
            return
        chosen = nd_collect()
        if len(chosen) < 2:
            post("Tick at least two objectives - a single one has no hypervolume to "
                 "maximise.")
            return
        _run_compute_hv(objective_path, chosen)

    hv_btn.clicked.connect(_on_compute_hv)

    return page
