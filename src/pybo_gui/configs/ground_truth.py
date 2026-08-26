"""Where the ground-truth tab remembers its last fit settings.

Selection itself lives in the Steps window, which does not persist its own ticks across
a restart - so there is nothing to remember here beyond degree/positive/source. Same
idiom as configs/workspace.py regardless: a small JSON blob, merged against a default on
read so an older file never breaks a newer GUI. Kept in its own module and its own file
rather than folded into workspace.py's state.json - the workspace is a path shared by
every tab, this is one tab's own settings, and the two have no reason to change together.
"""
import json
from pathlib import Path

_PKG_DIR = Path(__file__).parent
APP_DIR = _PKG_DIR / "gui_app"
STATE_PATH = APP_DIR / "ground_truth_state.json"

_DEFAULT_STATE = {"degree": 2, "positive": False, "source": "all"}


def _read_state() -> dict:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_DEFAULT_STATE)
    return {**_DEFAULT_STATE, **state} if isinstance(state, dict) else dict(_DEFAULT_STATE)


def get_state() -> dict:
    """The fit settings the last "Build ground truth" was launched with."""
    return _read_state()


def set_state(**fields) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({**_read_state(), **fields}, indent=2),
                          encoding="utf-8")
