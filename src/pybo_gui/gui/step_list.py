"""The selection window - and, since it is what every map is built from, the map itself.

Shows the directory tree under a chosen root and lets any node be ticked. A ticked node
means "everything under this", so one click takes a whole study, and ticking nodes in
different places takes steps from different studies - the tree decides nothing on your
behalf, it just shows what is there.

This window also owns building experiment_map.json + group_map.json from that selection
(rebuild_map/regroup/save_map/load_map below), and the map held in memory
(current_map/current_groups/map_source/map_fingerprint) - the natural place for it, since
every one of those buttons acts on checked_paths/reference_paths, the tree's own state.
Two things it still cannot know on its own, because they belong to boxes built elsewhere
in the tab (the Parameters box, the Objective box): what resolution each parameter is
compared at, and whether a ground truth is loaded. `configure()` is how the rest of the
tab hands those over, once it exists - this window is built before it, so they cannot be
constructor arguments.

Nothing here knows what a step record's *fields* mean, still - the map builder's
find_steps()/build_map() do the reading, this window only says which directories to read
and holds onto whatever they hand back.

Other parts of the GUI read `checked_paths` and subscribe to `on_selection`.
"""
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

from pybo_gui.configs import settings as configs_settings
from pybo_gui.configs import workspace
from pybo_gui.gui.launchers import run_off_thread
from pybo_gui.gui.message_log import post
from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import (
    build_map, find_steps, map_stamp, stamp_digest,
)
from pybo_gui.modules.bayesian_campaign_analysis.build_group_map import build_groups

_PATH_ROLE = Qt.ItemDataRole.UserRole
_FILLED_ROLE = Qt.ItemDataRole.UserRole + 1


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


class _DragSelectTree(QTreeWidget):
    """A tree whose checkboxes can be swept with a left-drag.

    Pressing anywhere on a row toggles it - the box is a small target and the whole row
    means that row - and the drag then inverts every further row the cursor passes.
    Inverting rather than stamping the first row's state is what lets one gesture both
    add and remove: a sweep across a mixed list ticks what was clear and clears what was
    ticked, instead of flattening the range to whatever the first row happened to be.

    Two columns are checkable - Include (0) and Reference (1) - each swept
    independently: a press picks up whichever column it landed in and the sweep stays
    on that column, so one gesture never touches both.

    The expand arrows keep their own press: it opens a node, it does not choose one.

    Qt itself ticks a box on the *release*, not the press, which is no use to a gesture
    that has to know the state from the first row onwards. So a press on a row takes the
    gesture over: this class sets every row it touches, and the release is not forwarded,
    or the delegate would toggle the pressed row a second time and cancel it out.

    `on_drag` is called with (True, column) when a sweep starts and (False, column) when
    it ends, so the window can hold off recounting the selection until the gesture is
    over - the count walks the filesystem, which is not something to do once per row.
    """

    _CHECKABLE_COLUMNS = (0, 1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sweeping = False
        self._applied = set()
        self._column = 0
        self.on_drag = None

    @staticmethod
    def _invert(item, column) -> None:
        item.setCheckState(column, Qt.CheckState.Unchecked
                           if item.checkState(column) == Qt.CheckState.Checked
                           else Qt.CheckState.Checked)

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        item = self.itemAt(pos)
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if item is None or item.data(0, _PATH_ROLE) is None:
            return
        # visualItemRect starts at the row proper; anything left of it is the branch
        # indicator, whose press belongs to expanding the node.
        if pos.x() < self.visualItemRect(item).left():
            return
        column = self.columnAt(pos.x())
        if column not in self._CHECKABLE_COLUMNS:
            return
        self._sweeping = True
        self._applied = {id(item)}
        self._column = column
        self._invert(item, column)

    def mouseMoveEvent(self, event):
        if not self._sweeping or not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        item = self.itemAt(event.position().toPoint())
        if item is None or item.data(0, _PATH_ROLE) is None or id(item) in self._applied:
            return
        if len(self._applied) == 1 and self.on_drag is not None:
            self.on_drag(True, self._column)  # the press alone was a click; now a sweep
        self._applied.add(id(item))
        self._invert(item, self._column)

    def mouseReleaseEvent(self, event):
        if not self._sweeping:
            super().mouseReleaseEvent(event)
            return
        swept = len(self._applied) > 1
        self._sweeping, self._applied = False, set()
        event.accept()  # withhold the release: see the note on Qt's toggle above
        if swept and self.on_drag is not None:
            self.on_drag(False, self._column)


class StepListWindow(QWidget):
    """A non-closable side window for choosing what a plot covers."""

    def __init__(self, parent: QWidget = None, *, initial_root: str = "",
                 on_selection=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Selection")
        self.resize(460, 640)
        # Don't hold the application open on our own once the main window closes.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._force_close = False
        self._on_selection = on_selection
        # Recounting walks the filesystem, so it waits until a sweep is finished.
        self._sweeping = False

        # ---- Map state -------------------------------------------------------
        # "source" records where the map came from: a selection rebuild, or a file the
        # user loaded. A loaded map must survive the next plot click rather than being
        # silently rebuilt from a selection it has nothing to do with.
        self._map = {"map": None, "groups": None, "source": None, "stamp": None,
                    "building": False}
        # One directory per session, so a rebuild never writes into the user's data tree.
        # save_map is the only thing that puts a map where the user chose.
        self._scratch = workspace.new_instance_dir()
        # What this window cannot know on its own - see configure() and the module
        # docstring. Safe no-op defaults, so a click before the rest of the tab exists
        # (it never happens - build() wires these before the event loop starts - but
        # nothing here should rely on that being true) fails soft rather than crashing.
        self._resolutions = dict
        self._has_ground_truth = lambda: False
        self._on_map_changed = lambda: None

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self._root_edit = QLineEdit(initial_root)
        self._root_edit.setPlaceholderText("Directory to browse")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.scan)
        row.addWidget(self._root_edit)
        row.addWidget(browse)
        row.addWidget(rescan)
        layout.addLayout(row)

        self._tree = _DragSelectTree()
        self._tree.on_drag = self._on_drag
        self._tree.setHeaderLabels(["Directory", "Reference"])
        self._tree.header().setStretchLastSection(False)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.setColumnWidth(1, 80)
        # Drag across rows to sweep their checkboxes; the press still toggles one.
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.NoSelection)
        self._tree.itemChanged.connect(self._on_item_changed)
        # Children are built the first time a node is opened, so pointing at a deep tree
        # costs one directory listing rather than a full walk.
        self._tree.itemExpanded.connect(self._fill)
        layout.addWidget(self._tree)

        buttons = QHBoxLayout()
        clear = QPushButton("Clear selection")
        clear.clicked.connect(self._clear)
        buttons.addWidget(clear)
        buttons.addStretch()
        layout.addLayout(buttons)

        self._status = QLabel("")
        self._status.setStyleSheet("color: grey;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # ---- Experiment map ---------------------------------------------------
        # Below the tree its buttons act on (checked_paths/reference_paths), rather than
        # in a tab across the main window from it.
        map_box = QGroupBox("Experiment map")
        map_layout = QVBoxLayout(map_box)
        btn_rebuild = QPushButton("Rebuild map now")
        btn_regroup = QPushButton("Regroup")
        btn_view_exp = QPushButton("View experiment map")
        btn_view_grp = QPushButton("View group map")
        btn_save_map = QPushButton("Save map")
        btn_load_map = QPushButton("Load map")
        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(0, 0, 0, 0)
        for widget in (btn_rebuild, btn_regroup, btn_view_exp, btn_view_grp,
                      btn_save_map, btn_load_map):
            buttons_row.addWidget(widget)
        buttons_row.addStretch()
        map_layout.addLayout(buttons_row)
        layout.addWidget(map_box)

        btn_rebuild.clicked.connect(
            lambda: self.rebuild_map(on_done=lambda ok: ok and self._on_map_changed(),
                                     force=True))
        btn_regroup.clicked.connect(self._regroup_clicked)
        btn_view_exp.clicked.connect(lambda: self.with_shown_map(
            lambda ok: ok and self._view_json_dialog(self.current_map, "Experiment map")))
        btn_view_grp.clicked.connect(lambda: self.with_shown_map(
            lambda ok: ok and self._view_group_map_dialog()))
        btn_save_map.clicked.connect(self.save_map)
        btn_load_map.clicked.connect(self._load_map_clicked)

    # ---------- experiment map ----------

    def configure(self, *, resolutions, has_ground_truth, on_map_changed=None) -> None:
        """Hand this window what it cannot know on its own, once the rest of the tab
        that owns those things exists - see the module docstring.

        `resolutions` and `has_ground_truth` are callables, read fresh whenever a map
        build needs them, not values taken once here: a resolution typed after this
        runs must still reach the next rebuild. `on_map_changed` is called with no
        arguments after Rebuild map now or Load map succeed - not after every rebuild
        (see with_map, used before a plot launches, which does not notify) - so the tab
        can refresh whatever reads the map's keys without doing so on every plot click.
        """
        self._resolutions = resolutions
        self._has_ground_truth = has_ground_truth
        self._on_map_changed = on_map_changed or (lambda: None)

    @property
    def current_map(self) -> dict | None:
        return self._map["map"]

    @property
    def current_groups(self) -> list | None:
        return self._map["groups"]

    @property
    def map_source(self) -> str | None:
        return self._map["source"]

    @property
    def map_fingerprint(self) -> dict | None:
        """What the current map was built from - map_stamp's own record of the roots,
        the selection and the resolutions - or None for a loaded map, which has no
        fingerprint of records behind it (see load_map)."""
        return self._map["stamp"]

    @property
    def scratch_dir(self) -> Path:
        """Where the map in memory is written for the analysis scripts to read - a
        per-session directory, not the user's data tree. save_map is the only thing
        that puts a copy where they chose."""
        return self._scratch

    def rebuild_map(self, on_done=None, force: bool = False) -> None:
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
        if self._map.get("building"):
            post("Still building the previous map — wait for it to finish.")
            finish(False)
            return
        # Everything the build needs is read from the widgets here, on the GUI thread,
        # and closed over: the worker must not touch a Qt object.
        steps = self.checked_paths
        # A reference not otherwise selected still has to be in the map, or its own
        # points would never reach the plot for _reference to draw - so its
        # directories are folded into the roots build_map reads, independently of
        # whether the user also ticked Include on them.
        references = self.reference_paths
        roots = sorted(set(steps) | set(references))
        # No roots is a request for the ground truth on its own, as long as there is a
        # ground truth to draw: build_map([]) is an empty map, and a plot over one draws
        # the backdrop and no series. Without one it is just an empty plot, so that still
        # asks for a selection rather than opening a blank figure.
        if not roots and not self._has_ground_truth():
            post("Select at least one step in the Steps window.")
            finish(False)
            return
        resolutions = self._resolutions()
        self._map["building"] = True
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
            self._map["building"] = False
            if isinstance(result, BaseException):
                post(f"Map build failed: {result}")
                finish(False)
                return
            exp_map, groups, reused, stamp = result
            self._map["map"], self._map["groups"] = exp_map, groups
            self._map["source"] = "selection"
            # What this map was built from. regroup reuses it rather than taking a fresh
            # one, so a regrouped map is cached as coming from the records it really came
            # from - see _cache_regrouped.
            self._map["stamp"] = stamp
            # A plot is a separate process reading configs.settings.data_path, so the map
            # has to exist somewhere on disk for it. That somewhere is a scratch directory
            # for the session, not the user's data tree - save_map is how a copy gets kept.
            self._write_map(self._scratch)
            configs_settings.set_data_path(self._scratch)
            if not roots:
                post("No steps selected — the ground truth will be drawn on its own")
                finish(True)
                return
            series = len({e["experiment_type"] for e in exp_map["experiments"]})
            # Said out loud on purpose: a reused map that is somehow wrong has to be
            # visible, not silent. Rebuild map now is the way to force one either way.
            how = "reused, unchanged since it was built" if reused else "freshly built from the records"
            message = (f"{len(exp_map['experiments'])} observations from "
                       f"{len(roots)} selected director"
                       f"{'y' if len(roots) == 1 else 'ies'}, {series} series, {how}")
            if references:
                message += f", {len(references)} reference director" \
                           f"{'y' if len(references) == 1 else 'ies'}"
            post(message)
            finish(True)

        run_off_thread(_work, _apply)

    def _write_map(self, out_dir) -> Path:
        """Write the map held in memory into `out_dir`."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "experiment_map.json").write_text(
            json.dumps(self._map["map"], indent=2), encoding="utf-8")
        (out_dir / "group_map.json").write_text(
            json.dumps(self._map["groups"], indent=2), encoding="utf-8")
        return out_dir

    def _cache_regrouped(self, resolutions: dict) -> None:
        """Store the regrouped map under the records it was built from.

        Nothing to store when the map came from a file rather than a build, since then no
        fingerprint of the records stands behind it.
        """
        cache = workspace.cache_dir()
        if cache is None or not self._map.get("stamp"):
            return
        stamp = {**self._map["stamp"],
                 "resolutions": {str(k): v for k, v in sorted(resolutions.items())}}
        entry = cache / stamp_digest(stamp)
        try:
            entry.mkdir(parents=True, exist_ok=True)
            (entry / "experiment_map.json").write_text(
                json.dumps(self._map["map"], indent=2), encoding="utf-8")
            (entry / "group_map.json").write_text(
                json.dumps(self._map["groups"], indent=2), encoding="utf-8")
            (entry / "stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 - a cache that cannot be written is not fatal
            post(f"Could not cache the regrouped map: {exc}")

    def regroup(self) -> None:
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
        if not self._map["map"]:
            post("Nothing to regroup — build a map first.")
            return
        resolutions = self._resolutions()
        self._map["groups"] = build_groups(self._map["map"], resolutions)
        self._write_map(self._scratch)
        configs_settings.set_data_path(self._scratch)
        self._cache_regrouped(resolutions)
        post(f"Regrouped into {len(self._map['groups'])} groups, "
             f"{len(resolutions)} parameters by resolution")

    def _regroup_clicked(self) -> None:
        self.regroup()
        self._on_map_changed()

    def with_shown_map(self, on_ready) -> None:
        """Call `on_ready(True)` with whatever map is already held, building one only if
        there is none.

        For the viewers, which exist to show the current state rather than to produce a
        fresh one. Reading a large campaign's records is the expensive thing the GUI
        does, and doing it to redraw a JSON the tab is already holding is work for
        nothing - Rebuild map now is what asks for a new one.
        """
        if self._map["map"]:
            on_ready(True)
            return
        self.rebuild_map(on_ready)

    def with_map(self, on_ready) -> None:
        """Call `on_ready(ok)` with a map to work from: a loaded one as it is, else one
        rebuilt from the selection.

        A loaded map is left alone so that plotting it does not quietly replace it with
        whatever happens to be ticked in the tree; Rebuild map now is how you go back to
        the selection. That case answers immediately; a rebuild answers when its worker
        finishes, which is why this hands the caller a callback instead of a bool.
        """
        if self._map["source"] == "loaded" and self._map["map"]:
            on_ready(True)
            return
        self.rebuild_map(on_ready)

    def load_map(self) -> None:
        """Read a saved map back in, and work from it until the next rebuild."""
        chosen = QFileDialog.getExistingDirectory(self, "Load a map from", self.root or "")
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
            groups = build_groups(exp_map, self._resolutions())

        self._map["map"], self._map["groups"] = exp_map, groups
        self._map["source"] = "loaded"
        # Loaded from a file, so there is no record fingerprint behind it and a regroup
        # of it has nothing honest to cache under.
        self._map["stamp"] = None
        self._write_map(self._scratch)
        configs_settings.set_data_path(self._scratch)
        post(f"Loaded {len(exp_map['experiments'])} observations from "
                           f"{source} — plots use this until you rebuild")

    def _load_map_clicked(self) -> None:
        self.load_map()
        self._on_map_changed()

    def save_map(self) -> None:
        """Ask where to keep a copy of the map, and write it there."""
        if not self._map["map"]:
            post("Nothing to save — click “Rebuild map now” first.")
            return
        chosen = QFileDialog.getExistingDirectory(self, "Save the map to", self.root or "")
        if not chosen:
            return
        try:
            self._write_map(chosen)
        except OSError as exc:
            post(f"Could not save: {exc}")
            return
        post(f"Saved experiment_map.json and group_map.json to {chosen}")

    def _view_json_dialog(self, payload, title: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(700, 600)
        v = QVBoxLayout(dlg)
        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(json.dumps(payload, indent=2) if payload else
                         "Nothing built yet — click “Rebuild map now” first.")
        v.addWidget(txt)
        dlg.show()

    def _view_group_map_dialog(self) -> None:
        groups, exp_map = self.current_groups, self.current_map
        dlg = QDialog(self)
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
            # The group_id travels with the row rather than being read back out of a cell,
            # so the detail pane keeps working however the rows are ordered.
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

    # ---------- selection ----------

    @property
    def root(self) -> str:
        return self._root_edit.text().strip()

    @property
    def checked_paths(self) -> list:
        """The ticked (Include) directories, with any nested under another ticked one
        dropped.

        Without that pruning a study and one of its own steps could both be ticked, and
        the builder would read that step twice - the same observation twice over in the
        map.
        """
        return self._ticked(0)

    @property
    def reference_paths(self) -> list:
        """The directories ticked Reference, pruned the same way as checked_paths.

        Independent of Include: a reference marks the benchmark a plot is read
        against regardless of which candidate runs happen to be included alongside
        it, so it survives Clear selection and is unioned into the map's roots by
        whoever builds it (tab_campaign._rebuild_map), not folded into
        checked_paths here.
        """
        return self._ticked(1)

    def _ticked(self, column: int) -> list:
        ticked = [Path(item.data(0, _PATH_ROLE)) for item in self._walk()
                  if item.data(0, _PATH_ROLE) is not None
                  and item.checkState(column) == Qt.CheckState.Checked]
        return [str(path) for path in ticked
                if not any(other != path and other in path.parents for other in ticked)]

    def _walk(self, parent: QTreeWidgetItem = None):
        """Every item currently in the tree, depth first."""
        if parent is None:
            for i in range(self._tree.topLevelItemCount()):
                yield from self._walk(self._tree.topLevelItem(i))
            return
        yield parent
        for i in range(parent.childCount()):
            yield from self._walk(parent.child(i))

    def _clear(self) -> None:
        # Include only: Reference marks the benchmark a plot is read against, which
        # stays meaningful across re-picking a fresh set of candidate runs to compare
        # it with, so a plain "start over" on the selection leaves it alone.
        self._tree.blockSignals(True)
        for item in self._walk():
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._tree.blockSignals(False)
        self._update_status()

    def _cascade(self, item: QTreeWidgetItem, column: int, state) -> None:
        """Give every already-listed descendant the same state, in `column`.

        Nodes not opened yet have no children to set; _fill passes the state down when
        it lists them, so an unopened subtree ends up matching too. Include and
        Reference cascade independently - ticking one says nothing about the other.
        """
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, _PATH_ROLE) is None:  # the "..." placeholder
                continue
            child.setCheckState(column, state)
            self._cascade(child, column, state)

    def _update_ancestors(self, item: QTreeWidgetItem, column: int) -> None:
        """Bring each ancestor's `column` into line with its children's.

        Without this a ticked parent would keep covering a child the user has just
        unticked - the parent is what gets sent, so the untick would do nothing. A
        parent whose children disagree goes partially checked, which is not a selection
        of its own: its ticked children are then sent individually.
        """
        parent = item.parent()
        while parent is not None:
            states = [parent.child(i).checkState(column)
                      for i in range(parent.childCount())
                      if parent.child(i).data(0, _PATH_ROLE) is not None]
            if states and all(s == Qt.CheckState.Checked for s in states):
                state = Qt.CheckState.Checked
            elif not states or all(s == Qt.CheckState.Unchecked for s in states):
                state = Qt.CheckState.Unchecked
            else:
                state = Qt.CheckState.PartiallyChecked
            if parent.checkState(column) == state:
                break
            parent.setCheckState(column, state)
            parent = parent.parent()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column not in _DragSelectTree._CHECKABLE_COLUMNS:
            return
        # Ticking a directory means everything under it, so the tree says so: the
        # children follow, and the ancestors are re-derived from what is now ticked.
        self._tree.blockSignals(True)
        if item.checkState(column) != Qt.CheckState.PartiallyChecked:
            self._cascade(item, column, item.checkState(column))
        self._update_ancestors(item, column)
        self._tree.blockSignals(False)
        self._update_status()
        if column == 0 and self._on_selection is not None and not self._sweeping:
            self._on_selection(self.checked_paths)

    def _on_drag(self, sweeping: bool, column: int) -> None:
        self._sweeping = sweeping
        if sweeping:
            return
        self._update_status()
        if column == 0 and self._on_selection is not None:
            self._on_selection(self.checked_paths)

    def _update_status(self) -> None:
        if self._sweeping:
            return
        selected = self.checked_paths
        references = self.reference_paths
        if not selected and not references:
            self._status.setText("Nothing selected — tick any directory to include "
                                 "everything under it.")
            return
        # What the selection actually expands to, since a directory says nothing about
        # how many records are beneath it.
        n_steps = len(find_steps(selected)) if selected else 0
        text = (f"{len(selected)} director{'y' if len(selected) == 1 else 'ies'} selected "
                f"— {n_steps} step record{'' if n_steps == 1 else 's'}")
        if references:
            text += (f"; {len(references)} director"
                     f"{'y' if len(references) == 1 else 'ies'} marked reference")
        self._status.setText(text)

    # ---------- scanning ----------

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a directory", self.root)
        if chosen:
            self._root_edit.setText(chosen)
            self.scan()

    def _add(self, parent, path: Path) -> QTreeWidgetItem:
        item = QTreeWidgetItem([path.name or str(path)])
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(0, Qt.CheckState.Unchecked)
        item.setCheckState(1, Qt.CheckState.Unchecked)
        item.setData(0, _PATH_ROLE, str(path))
        item.setData(0, _FILLED_ROLE, False)
        if parent is None:
            self._tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        # A placeholder child is what gives the node an expand arrow before we have
        # listed it; _fill replaces it on first open.
        if self._has_subdirs(path):
            item.addChild(QTreeWidgetItem(["..."]))
        return item

    @staticmethod
    def _has_subdirs(path: Path) -> bool:
        try:
            return any(child.is_dir() for child in path.iterdir())
        except OSError:
            return False

    def _fill(self, item: QTreeWidgetItem) -> None:
        """Replace a node's placeholder with its real subdirectories, once."""
        if item.data(0, _FILLED_ROLE):
            return
        item.setData(0, _FILLED_ROLE, True)
        path = Path(item.data(0, _PATH_ROLE))
        self._tree.blockSignals(True)
        item.takeChildren()
        try:
            children = sorted(c for c in path.iterdir() if c.is_dir())
        except OSError:
            children = []
        for child in children:
            self._add(item, child)
        # A node ticked before it was opened must pass that down, or its new children
        # would look unselected while the selection still covers them. Both columns,
        # independently - either may have been ticked without the other.
        for column in _DragSelectTree._CHECKABLE_COLUMNS:
            if item.checkState(column) == Qt.CheckState.Checked:
                for i in range(item.childCount()):
                    item.child(i).setCheckState(column, Qt.CheckState.Checked)
        self._tree.blockSignals(False)

    def scan(self) -> None:
        """Rebuild the tree from the current root."""
        self._tree.blockSignals(True)
        self._tree.clear()
        root = Path(self.root) if self.root else None
        if root is not None and root.is_dir():
            top = self._add(None, root)
            self._tree.blockSignals(False)
            top.setExpanded(True)  # fills the first level
        else:
            self._tree.blockSignals(False)
            self._status.setText("Not a directory." if self.root else
                                 "Choose a directory to browse.")
            return
        self._update_status()

    # ---------- lifecycle ----------

    def force_close(self) -> None:
        self._force_close = True
        self.close()

    def closeEvent(self, event):
        # Non-closable while the application is up: the tab depends on the selection.
        if self._force_close:
            event.accept()
        else:
            event.ignore()