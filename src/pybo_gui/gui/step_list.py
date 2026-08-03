"""The step-selector window.

Lists every step of every run under a chosen root, as a run -> steps tree with a
checkbox on each. Steps are the selectable unit because a campaign plot is built from
observations, and a step is the batch of observations one optimization iteration
produced - so ticking a subset asks "what did the campaign look like using only these".

Other parts of the GUI read `checked_paths` and subscribe to `on_selection`.
"""
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

_STEP_ROLE = Qt.ItemDataRole.UserRole


class StepListWindow(QWidget):
    """A non-closable side window listing the steps available under a root."""

    def __init__(self, parent: QWidget = None, *, initial_root: str = "",
                 on_selection=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Steps")
        self.resize(420, 640)
        # Don't hold the application open on our own once the main window closes.
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self._force_close = False
        self._on_selection = on_selection

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        self._root_edit = QLineEdit(initial_root)
        self._root_edit.setPlaceholderText("Study, run or data directory")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.scan)
        row.addWidget(self._root_edit)
        row.addWidget(browse)
        row.addWidget(rescan)
        layout.addLayout(row)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Run / step", "Observations"])
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        buttons = QHBoxLayout()
        for text, value in (("Select all", True), ("Select none", False)):
            btn = QPushButton(text)
            btn.clicked.connect(lambda _=False, v=value: self._set_all(v))
            buttons.addWidget(btn)
        buttons.addStretch()
        layout.addLayout(buttons)

        self._status = QLabel("")
        self._status.setStyleSheet("color: grey;")
        layout.addWidget(self._status)

    # ---------- selection ----------

    @property
    def root(self) -> str:
        return self._root_edit.text().strip()

    @property
    def checked_paths(self) -> list:
        """Directories of every ticked step, in tree order."""
        paths = []
        for i in range(self._tree.topLevelItemCount()):
            run = self._tree.topLevelItem(i)
            for j in range(run.childCount()):
                step = run.child(j)
                if step.checkState(0) == Qt.CheckState.Checked:
                    paths.append(step.data(0, _STEP_ROLE))
        return paths

    def _set_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._tree.topLevelItemCount()):
            self._tree.topLevelItem(i).setCheckState(0, state)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 0:
            return
        # A run's checkbox drives its steps; a step only reports upward.
        if item.data(0, _STEP_ROLE) is None:
            state = item.checkState(0)
            if state != Qt.CheckState.PartiallyChecked:
                self._tree.blockSignals(True)
                for j in range(item.childCount()):
                    item.child(j).setCheckState(0, state)
                self._tree.blockSignals(False)
        self._update_status()
        if self._on_selection is not None:
            self._on_selection(self.checked_paths)

    def _update_status(self) -> None:
        n = len(self.checked_paths)
        self._status.setText(f"{n} step{'' if n == 1 else 's'} selected")

    # ---------- scanning ----------

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a data directory", self.root)
        if chosen:
            self._root_edit.setText(chosen)
            self.scan()

    def scan(self) -> None:
        """Rebuild the tree from the current root.

        Steps are found by their experiment.json rather than by folder name, so a
        directory that holds no record simply does not appear.
        """
        self._tree.blockSignals(True)
        self._tree.clear()
        root = Path(self.root) if self.root else None
        records = sorted(root.glob("**/experiment.json")) if root and root.is_dir() else []

        by_run: dict = {}
        for record in records:
            by_run.setdefault(record.parent.parent, []).append(record.parent)

        for run_dir, steps in by_run.items():
            run_item = QTreeWidgetItem([run_dir.name, str(len(steps))])
            run_item.setFlags(run_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            run_item.setCheckState(0, Qt.CheckState.Unchecked)
            for step_dir in steps:
                step_item = QTreeWidgetItem([step_dir.name, ""])
                step_item.setFlags(step_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                step_item.setCheckState(0, Qt.CheckState.Unchecked)
                step_item.setData(0, _STEP_ROLE, str(step_dir))
                run_item.addChild(step_item)
            self._tree.addTopLevelItem(run_item)

        self._tree.expandAll()
        self._tree.resizeColumnToContents(0)
        self._tree.blockSignals(False)

        if not records:
            self._status.setText("No experiment.json found under that directory.")
        else:
            self._status.setText(f"{len(records)} steps in {len(by_run)} run(s) — none selected")

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
