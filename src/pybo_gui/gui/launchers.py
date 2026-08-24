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

# The ones stop_all has terminated and watch has yet to report on. terminate() gives the
# child a non-zero exit code, which on its own is indistinguishable from a crash, so
# without this a plot the user closed on purpose is announced as one that failed.
_stopped: set = set()

# How many stops there have been. A plot that is still waiting on something slow - its
# campaign map, say - is not a process yet, so stop_all has nothing to terminate and it
# would open once the wait ended, after the user asked for the opposite. Whoever queued
# it takes a token first and asks stopped_since() before going ahead.
_stop_count = 0


def stop_token() -> int:
    """The current stop count, to hand back to stopped_since() later."""
    return _stop_count


def stopped_since(token: int) -> bool:
    """Whether stop_all has run since `token` was taken."""
    return _stop_count != token


def run_off_thread(work, on_done) -> None:
    """Run `work()` on a worker thread and hand what it returns to `on_done` here.

    For the slow parts that are not subprocesses - reading a campaign's step records is
    seconds to minutes, and on the GUI thread that is a window Windows greys out as "not
    responding". The callback is marshalled back the same way watch() does it, through a
    timer bound to the application object, which lives on the main thread.

    `work` must touch no Qt object: widgets belong to the main thread, so read whatever it
    needs from them before calling and close over the values. An exception raised inside
    is passed to `on_done` as its argument rather than killing the thread silently - the
    caller decides how to report it.
    """
    app = QCoreApplication.instance()

    def _run():
        try:
            result = work()
        except BaseException as error:  # noqa: BLE001 - handed on, not swallowed
            result = error
        QTimer.singleShot(0, app, lambda: on_done(result))

    threading.Thread(target=_run, daemon=True).start()


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


def stop_all() -> int:
    """Terminate every plot still running, and say how many that was.

    The count is what the caller reports: the button itself gives no sign it did
    anything, so a stop that found nothing running looks exactly like one that closed
    five windows. A plot still waiting on its campaign map is not a process yet, so it
    is not counted here; the stop is recorded for it too, and it drops itself when the
    map lands - see stopped_since.
    """
    global _stop_count
    _stop_count += 1
    stopped = 0
    for proc in list(_procs):
        # Already terminated but not yet dead, on a second press a moment after the first:
        # it is on its way out, and counting it again would report it stopped twice.
        if proc.poll() is None and proc not in _stopped:
            try:
                proc.terminate()
            except OSError:
                continue
            _stopped.add(proc)
            stopped += 1
    return stopped


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
        # A stop is not an outcome to report: whoever called stop_all has already said so,
        # and the non-zero code terminate() leaves behind is not the script's own.
        if any(p in _stopped for p in active):
            _stopped.difference_update(active)
            return
        # A plot that exits non-zero printed its reason to its own console, which the GUI
        # does not show - so without this it reads as "Done." and looks like it never ran.
        failed = [p.returncode for p in active if p.returncode]
        if failed and on_fail is not None:
            QTimer.singleShot(0, app, lambda: on_fail(failed))
        elif on_done is not None:
            QTimer.singleShot(0, app, on_done)

    threading.Thread(target=_wait, daemon=True).start()
