"""One floating window holding everything the GUI has to say.

Messages used to be printed into a grey label under the button that produced them, which
put five of them in different places and made each one disappear as soon as the next
action overwrote it. They all go here instead: one window, in order, timestamped, so an
error from ten clicks ago is still readable.

The history is kept whether or not the window was ever opened, so opening it after
something went wrong still shows what happened.
"""
from datetime import datetime

from PySide6.QtCore import QCoreApplication, Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

_history: list = []
_window = None


class LogWindow(QWidget):
    """The log itself. Built on first open, then reused - closing it only hides it."""

    def __init__(self, parent=None):
        # Qt.Window rather than a plain child: it gets its own frame and can be moved off
        # the main window instead of being clipped inside it.
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("pyBO — log")
        self.resize(640, 320)

        layout = QVBoxLayout(self)
        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        layout.addWidget(self._view)

        clear = QPushButton("Clear")
        clear.clicked.connect(self.clear)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(clear)
        layout.addLayout(row)

        self._view.setPlainText("\n".join(_history))
        self._to_end()

    def _to_end(self) -> None:
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def append(self, line: str) -> None:
        self._view.appendPlainText(line)
        self._to_end()

    def clear(self) -> None:
        _history.clear()
        self._view.clear()


def post(text: str) -> None:
    """Record a message, and show it if the window happens to be open.

    Safe from any thread: the append is marshalled onto the main one the same way
    launchers.watch marshals its callbacks.
    """
    line = f"{datetime.now():%H:%M:%S}  {text}"
    _history.append(line)
    app = QCoreApplication.instance()
    if app is None or _window is None:
        return
    QTimer.singleShot(0, app, lambda: _window.append(line))


def show_log(parent=None) -> None:
    """Open the log, building it the first time and raising it afterwards."""
    global _window
    if _window is None:
        _window = LogWindow(parent)
    _window.show()
    _window.raise_()
    _window.activateWindow()


def history() -> list:
    """Every message so far, for tests."""
    return list(_history)
