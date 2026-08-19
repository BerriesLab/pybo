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
    QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from pybo_gui.modules.bayesian_campaign_analysis.build_experiment_map import find_steps

_PATH_ROLE = Qt.ItemDataRole.UserRole
_FILLED_ROLE = Qt.ItemDataRole.UserRole + 1


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