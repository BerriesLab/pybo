"""The selection window.

Shows the directory tree under a chosen root and lets any node be ticked. A ticked node
means "everything under this", so one click takes a whole study, and ticking nodes in
different places takes steps from different studies - the tree decides nothing on your
behalf, it just shows what is there.

Nothing here knows what a step record looks like: the map builder's find_steps() globs
experiment.json under whatever directories it is given, so this window's only job is to
say which directories those are.

Other parts of the GUI read `checked_paths` and subscribe to `on_selection`.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import find_steps

_PATH_ROLE = Qt.ItemDataRole.UserRole
_FILLED_ROLE = Qt.ItemDataRole.UserRole + 1


class _DragSelectTree(QTreeWidget):
    """A tree whose checkboxes can be swept with a left-drag.

    Qt handles the press, so clicking a box still toggles it and the expand arrows still
    work; the drag then applies whatever state that first row ended up in to every row
    the cursor passes. Sweeping down a list of runs is one gesture rather than one click
    each, and sweeping from a ticked row clears a range just as easily.

    `on_drag` is called with True when a sweep starts and False when it ends, so the
    window can hold off recounting the selection until the gesture is over - the count
    walks the filesystem, which is not something to do once per row.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._applied = set()
        self.on_drag = None

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        item = self.itemAt(event.position().toPoint())
        if item is None or item.data(0, _PATH_ROLE) is None:
            return
        self._state = item.checkState(0)
        self._applied = {id(item)}

    def mouseMoveEvent(self, event):
        if self._state is None or not (event.buttons() & Qt.MouseButton.LeftButton):
            super().mouseMoveEvent(event)
            return
        item = self.itemAt(event.position().toPoint())
        if item is None or item.data(0, _PATH_ROLE) is None or id(item) in self._applied:
            return
        if len(self._applied) == 1 and self.on_drag is not None:
            self.on_drag(True)  # the press alone was a click; this is now a sweep
        self._applied.add(id(item))
        item.setCheckState(0, self._state)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        swept = self._state is not None and len(self._applied) > 1
        self._state, self._applied = None, set()
        if swept and self.on_drag is not None:
            self.on_drag(False)


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
        self._tree.setHeaderLabels(["Directory"])
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

    # ---------- selection ----------

    @property
    def root(self) -> str:
        return self._root_edit.text().strip()

    @property
    def checked_paths(self) -> list:
        """The ticked directories, with any nested under another ticked one dropped.

        Without that pruning a study and one of its own steps could both be ticked, and
        the builder would read that step twice - the same observation twice over in the
        map.
        """
        ticked = []
        for item in self._walk():
            path = item.data(0, _PATH_ROLE)
            if path and item.checkState(0) == Qt.CheckState.Checked:
                ticked.append(Path(path))
        pruned = []
        for path in ticked:
            if not any(other != path and other in path.parents for other in ticked):
                pruned.append(str(path))
        return pruned

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
        self._tree.blockSignals(True)
        for item in self._walk():
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._tree.blockSignals(False)
        self._update_status()

    def _cascade(self, item: QTreeWidgetItem, state) -> None:
        """Give every already-listed descendant the same state.

        Nodes not opened yet have no children to set; _fill passes the state down when
        it lists them, so an unopened subtree ends up matching too.
        """
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, _PATH_ROLE) is None:  # the "..." placeholder
                continue
            child.setCheckState(0, state)
            self._cascade(child, state)

    def _update_ancestors(self, item: QTreeWidgetItem) -> None:
        """Bring each ancestor into line with its children.

        Without this a ticked parent would keep covering a child the user has just
        unticked - the parent is what gets sent, so the untick would do nothing. A
        parent whose children disagree goes partially checked, which is not a selection
        of its own: its ticked children are then sent individually.
        """
        parent = item.parent()
        while parent is not None:
            states = [parent.child(i).checkState(0) for i in range(parent.childCount())
                      if parent.child(i).data(0, _PATH_ROLE) is not None]
            if states and all(s == Qt.CheckState.Checked for s in states):
                state = Qt.CheckState.Checked
            elif not states or all(s == Qt.CheckState.Unchecked for s in states):
                state = Qt.CheckState.Unchecked
            else:
                state = Qt.CheckState.PartiallyChecked
            if parent.checkState(0) == state:
                break
            parent.setCheckState(0, state)
            parent = parent.parent()

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        # Ticking a directory means everything under it, so the tree says so: the
        # children follow, and the ancestors are re-derived from what is now ticked.
        self._tree.blockSignals(True)
        if item.checkState(0) != Qt.CheckState.PartiallyChecked:
            self._cascade(item, item.checkState(0))
        self._update_ancestors(item)
        self._tree.blockSignals(False)
        self._update_status()
        if self._on_selection is not None and not self._sweeping:
            self._on_selection(self.checked_paths)

    def _on_drag(self, sweeping: bool) -> None:
        self._sweeping = sweeping
        if not sweeping:
            self._update_status()
            if self._on_selection is not None:
                self._on_selection(self.checked_paths)

    def _update_status(self) -> None:
        if self._sweeping:
            return
        selected = self.checked_paths
        if not selected:
            self._status.setText("Nothing selected — tick any directory to include "
                                 "everything under it.")
            return
        # What the selection actually expands to, since a directory says nothing about
        # how many records are beneath it.
        n_steps = len(find_steps(selected))
        self._status.setText(
            f"{len(selected)} director{'y' if len(selected) == 1 else 'ies'} selected "
            f"— {n_steps} step record{'' if n_steps == 1 else 's'}")

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
        # would look unselected while the selection still covers them.
        if item.checkState(0) == Qt.CheckState.Checked:
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, Qt.CheckState.Checked)
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