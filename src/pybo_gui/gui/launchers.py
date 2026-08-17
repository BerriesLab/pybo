"""Subprocess helpers for the GUI.

Every plot is produced by launching a pybo_gui.modules.bayesian_campaign_analysis module, not by drawing in the
GUI process: the analysis scripts stay usable from a terminal, and a plot that dies
takes its own window with it rather than the application.
"""
import os
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

    stdout and stderr are piped and merged, for watch()'s on_output to read - a script
    that keeps running (its plot window still open) rather than exiting non-zero has no
    other way to say anything went partly wrong. PYTHONUNBUFFERED matches studies/_common.py's
    own trial subprocesses: without it, a print sits in the child's stdio buffer until
    it exits, arriving in the log no earlier than "Done." would have anyway.
    """
    _procs[:] = [p for p in _procs if p.poll() is None]
    env = os.environ | {"PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [sys.executable, "-m", module, *[str(a) for a in args]],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
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


def watch(procs, on_start=None, on_done=None, on_fail=None, on_output=None) -> None:
    """Wait on `procs` off the main thread, marshalling the callbacks back onto it.

    Binding the timer to QCoreApplication.instance() is what makes this safe: that
    object always lives on the main thread, so the callback runs there.

    `on_output`, when given, is called with each non-blank line a process prints -
    reading proc.stdout in place of a bare proc.wait() is what makes this possible, and
    it also has to happen either way: stdout is a pipe now (see launch_analysis), and an
    unread pipe fills and deadlocks the child once its output outgrows the OS buffer.
    A script that prints a partial-failure warning but still exits 0 (an arm's front
    that could not be aggregated, say) has no other route to the GUI; on_fail already
    covers the case where it exits non-zero instead.
    """
    active = [p for p in procs if p is not None]
    if not active:
        return
    app = QCoreApplication.instance()
    if on_start is not None:
        QTimer.singleShot(0, app, on_start)

    def _wait():
        for proc in active:
            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    line = raw_line.rstrip("\n")
                    if line and on_output is not None:
                        QTimer.singleShot(0, app, lambda l=line: on_output(l))
            proc.wait()
        # A plot that exits non-zero printed its reason to its own console, which the GUI
        # does not show - so without this it reads as "Done." and looks like it never ran.
        failed = [p.returncode for p in active if p.returncode]
        if failed and on_fail is not None:
            QTimer.singleShot(0, app, lambda: on_fail(failed))
        elif on_done is not None:
            QTimer.singleShot(0, app, on_done)

    threading.Thread(target=_wait, daemon=True).start()
