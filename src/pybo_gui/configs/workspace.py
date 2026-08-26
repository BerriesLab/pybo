"""Where a GUI session keeps the files it produces.

Every rebuild writes experiment_map.json and group_map.json somewhere on disk, because a
plot is a separate process that reads them through configs.settings.data_path. Left to
itself that somewhere is a temporary directory, which is gone the moment the session ends
- so the map is rebuilt from scratch next time, and reading a large campaign's step
records is minutes, not seconds.

Pointing this at a real folder keeps those files: they can be reused instead of rebuilt,
inspected, fed to a script run straight from a terminal
(PYBO_CAMPAIGN_DIR=<instance dir> python -m pybo_gui.modules...), and a session that
crashes leaves its map behind rather than taking it away.

One directory per session, not per folder. A temporary directory gives every process its
own by construction; a shared folder would have two open windows overwriting each other's
map, and whichever plot launched last would read the wrong one. The timestamp makes an
instance directory readable, the pid makes it unique.

Unset is the default, and means exactly the old behaviour - so nothing changes for anyone
who never opens the setting.
"""
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

_PKG_DIR = Path(__file__).parent

# --- Application seam: the single place this points at the host app, matching
# figure_settings.store's own seam. ------------------------------------------------
APP_DIR = _PKG_DIR / "gui_app"
STATE_PATH = APP_DIR / "state.json"

_DEFAULT_STATE = {"workspace": None}


def _read_state() -> dict:
    """The stored state, falling back to the default on anything unreadable.

    A state file that cannot be parsed is not worth an exception on startup: the setting
    is a convenience, and losing it costs a rebuild, not any data.
    """
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_DEFAULT_STATE)
    return {**_DEFAULT_STATE, **state} if isinstance(state, dict) else dict(_DEFAULT_STATE)


def _write_state(state: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_workspace() -> Path | None:
    """The folder session directories are made in, or None for a temporary one.

    A configured folder that has since been deleted or renamed reads as unset rather than
    as an error: the session still needs somewhere to write, and a temporary directory is
    the answer that always works.
    """
    stored = _read_state()["workspace"]
    if not stored:
        return None
    path = Path(stored)
    return path if path.is_dir() else None


def set_workspace(path) -> None:
    """Point at `path`, or pass a falsy value to go back to temporary directories."""
    _write_state({**_read_state(), "workspace": str(Path(path).resolve()) if path else None})


def cache_dir() -> Path | None:
    """Where built maps are kept for reuse, or None when there is no workspace.

    In the workspace, not in the session directory: a session gets a fresh directory
    every time, so a map cached inside one would never be read again. Sharing it is the
    whole point - the rebuild it saves is minutes on a large campaign.

    Without a workspace there is nowhere durable to put it, so nothing is cached and every
    session builds from scratch, exactly as before this existed.
    """
    workspace = get_workspace()
    if workspace is None:
        return None
    cache = workspace / "map_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def gt_map_cache_dir() -> Path | None:
    """Where a map built for the ground-truth tab is kept, or None with no workspace.

    A sibling of cache_dir(), not the same directory: the ground-truth tab and the
    campaign plots build maps from selections that are free to differ, and sharing one
    cache would mean one's rebuild could evict or be mistaken for the other's -
    keeping them apart is what "same method, files that don't overwrite" means here.
    """
    workspace = get_workspace()
    if workspace is None:
        return None
    cache = workspace / "gt_map_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def new_instance_dir() -> Path:
    """A fresh directory for this session to write its maps into.

    Called once per session: the setting can change while a session runs, but the
    directory it already writes to must not, or configs.settings.data_path and the map on
    disk would drift apart.
    """
    workspace = get_workspace()
    if workspace is None:
        return Path(tempfile.mkdtemp(prefix="pybo_campaign_"))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = workspace / f"{stamp}_{os.getpid()}"
    # The timestamp is only second-resolution and the pid is the same within a process, so
    # two calls in the same second would otherwise land in one directory and overwrite each
    # other's map. A suffix keeps them apart, the way cli.unique_dir does for run output.
    candidate, index = base, 0
    while candidate.exists():
        index += 1
        candidate = base.with_name(f"{base.name}_{index:03d}")
    # Named, not created. A session that never builds a map has nothing to put here, and
    # creating it up front left an empty dated directory behind every time the GUI was
    # opened and closed again. Whoever writes the first file makes it (see _write_map).
    return candidate

def _dir_size(path: Path) -> int:
    """Bytes held under `path`, skipping anything that vanishes while being counted."""
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def usage() -> dict | None:
    """What the workspace is holding, in bytes, or None when there is no workspace.

    Split into the two things that grow for different reasons: the cache, one entry per
    distinct selection ever built, which is safe to delete; and the session directories,
    one per run of the GUI, which are what a saved map or a crashed session left behind.
    """
    root = get_workspace()
    if root is None:
        return None
    caches = [root / "map_cache", root / "gt_map_cache"]
    cached = sum(_dir_size(c) for c in caches if c.is_dir())
    total = _dir_size(root)
    entries = sum(len(list(c.iterdir())) for c in caches if c.is_dir())
    return {"total": total, "cache": cached, "sessions": total - cached, "entries": entries}


def clear_cache() -> int:
    """Delete the cached maps, returning the bytes freed.

    Only the caches, and only their contents: a cached map is rebuilt from the records on
    demand, so losing it costs time and nothing else. The session directories are left
    alone - one of them holds the map the running GUI is pointing its plots at, and
    another may be what a crashed session left to be recovered.
    """
    root = get_workspace()
    if root is None:
        return 0
    freed = 0
    for cache in (root / "map_cache", root / "gt_map_cache"):
        if not cache.is_dir():
            continue
        freed += _dir_size(cache)
        for entry in cache.iterdir():
            shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
    return freed
