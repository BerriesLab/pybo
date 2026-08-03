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

# Keyed by (module, *args) so clicking Plot twice with the same options raises the
# window that is already open instead of stacking duplicates.
_procs: dict = {}


def launch_analysis(module: str, *args) -> subprocess.Popen:
    """Run `python -m <module> <args...>`, deduplicated.

    No cwd is forced: pybo_gui.modules.bayesian_campaign_analysis is importable wherever pybo is installed, and
    inheriting the GUI's directory is what makes a relative path the user typed resolve
    the way they meant it.
    """
    key = (module, *args)
    running = _procs.get(key)
    if running is not None and running.poll() is None:
        return running
    _procs[key] = subprocess.Popen(
        [sys.executable, "-m", module, *[str(a) for a in args]])
    return _procs[key]


def stop_all() -> None:
    """Terminate every plot still running."""
    for proc in list(_procs.values()):
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass


def watch(procs, on_start=None, on_done=None) -> None:
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
        if on_done is not None:
            QTimer.singleShot(0, app, on_done)

    threading.Thread(target=_wait, daemon=True).start()
