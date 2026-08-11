"""Subprocess helpers for the GUI.

Every plot is produced by launching a pybo_gui.modules.bayesian_campaign_analysis module, not by drawing in the
GUI process: the analysis scripts stay usable from a terminal, and a plot that dies
takes its own window with it rather than the application.
"""
import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer

# Every plot still running, so stop_all can reach it. Identical options are launched
# again rather than deduplicated: the data behind a plot changes as a campaign grows,
# so a second window of the "same" plot is a comparison, not a duplicate.
_procs: list = []


def launch_analysis(module: str, *args) -> subprocess.Popen:
    """Run `python -m <module> <args...>` in its own window.

    No cwd is forced: pybo_gui.modules.bayesian_campaign_analysis is importable wherever pybo is installed, and
    inheriting the GUI's directory is what makes a relative path the user typed resolve
    the way they meant it.
    """
    _procs[:] = [p for p in _procs if p.poll() is None]
    proc = subprocess.Popen(
        [sys.executable, "-m", module, *[str(a) for a in args]])
    _procs.append(proc)
    return proc


def stop_all() -> None:
    """Terminate every plot still running."""
    for proc in list(_procs):
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass


def watch(procs, on_start=None, on_done=None, on_fail=None) -> None:
    """Wait on `procs` off the main thread, marshalling the callbacks back onto it.

    Binding the timer to QCoreApplication.instance() is what makes this safe: that
    object always lives on the main thread, so the callback runs there.
    """
    active = [p for p in procs if p is not None]
    if not active:
        return
    app = QCoreApplication.instance()
    if on_start is not None:
        QTimer.singleShot(0, app, on_start)

    def _wait():
        for proc in active:
            proc.wait()
        # A plot that exits non-zero printed its reason to its own console, which the GUI
        # does not show - so without this it reads as "Done." and looks like it never ran.
        failed = [p.returncode for p in active if p.returncode]
        if failed and on_fail is not None:
            QTimer.singleShot(0, app, lambda: on_fail(failed))
        elif on_done is not None:
            QTimer.singleShot(0, app, on_done)

    threading.Thread(target=_wait, daemon=True).start()
