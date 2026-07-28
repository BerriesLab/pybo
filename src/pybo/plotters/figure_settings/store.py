"""Figure-settings store — a matplotlib-free file manager.

Everything figure styling needs lives in this package:

Shipped with the package (generic):
* ``defaults.yaml``          — generic base settings (dpi, font, grid, legend, scatter…).
* ``publisher_styles/*.yaml``— built-in journal styles (partial overrides + rcparams).

Provided by the host application (see APP_DIR below):
* ``defaults.yaml``          — the app's domain settings, merged over the package base.
* ``user_styles/*.yaml``     — user-created styles (partial overrides).
* ``state.json``             — the active selections: ``{"publisher": ..., "user": ...}``.

A *style* is a YAML mapping that is a partial of the defaults tree (any subset of keys,
plus an optional ``rcparams`` section and ``description``). The assembler
(``pybo.plotters.figure_settings.config``) resolves ``defaults -> publisher style -> user style`` in
memory. This module only reads/writes the files, so it stays free of matplotlib and is
safe to import from the GUI.
"""
import json
import yaml
from pathlib import Path

_PKG_DIR = Path(__file__).parent

# --- Application seam: the single place the package points at the host app. -------
# On extraction as a standalone package, replace this with configuration provided by
# the host application (a path passed in, an env var, or a config file). Everything
# below it is either shipped with the package (generic) or lives in the host app.
APP_DIR = _PKG_DIR.parent / "figure_settings_app"

# Shipped with the package (generic):
PACKAGE_DEFAULTS_PATH = _PKG_DIR / "defaults.yaml"
PUBLISHER_DIR = _PKG_DIR / "publisher_styles"
# Provided by the host application:
APP_DEFAULTS_PATH = APP_DIR / "defaults.yaml"
USER_DIR = APP_DIR / "user_styles"
STATE_PATH = APP_DIR / "state.json"

_DEFAULT_STATE = {"publisher": "ieee_single", "user": None}

USER_STYLE_TEMPLATE = (
    "# User style — a partial override of the defaults (see defaults.yaml).\n"
    "# List only the keys you want to change; they override the active publisher style.\n"
    "scatter:\n"
    "  marker_size: 45\n"
)


def _dir(kind: str) -> Path:
    return PUBLISHER_DIR if kind == "publisher" else USER_DIR


def _style_path(kind: str, name: str) -> Path:
    return _dir(kind) / f"{name}.yaml"


# --- Defaults (package generic base + host-app domain settings) --------------
def load_package_defaults() -> dict:
    data = yaml.safe_load(PACKAGE_DEFAULTS_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_app_defaults() -> dict:
    try:
        data = yaml.safe_load(APP_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def app_defaults_text() -> str:
    return APP_DEFAULTS_PATH.read_text(encoding="utf-8")


def write_app_defaults(text: str) -> None:
    APP_DEFAULTS_PATH.write_text(text, encoding="utf-8")


# --- Listing -----------------------------------------------------------------
def list_publisher_styles() -> list:
    return sorted(p.stem for p in PUBLISHER_DIR.glob("*.yaml"))


def list_user_styles() -> list:
    return sorted(p.stem for p in USER_DIR.glob("*.yaml")) if USER_DIR.exists() else []


# --- Active selections (state.json) ------------------------------------------
def _read_state() -> dict:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(_DEFAULT_STATE)
    if not isinstance(state, dict):
        return dict(_DEFAULT_STATE)
    return {"publisher": state.get("publisher"), "user": state.get("user")}


def _write_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_active() -> dict:
    """Active selections, dropping any that no longer point at an existing file."""
    state = _read_state()
    if state["publisher"] and state["publisher"] not in list_publisher_styles():
        state["publisher"] = None
    if state["user"] and state["user"] not in list_user_styles():
        state["user"] = None
    return state


def set_active_publisher(name) -> None:
    state = _read_state()
    state["publisher"] = name or None
    _write_state(state)


def set_active_user(name) -> None:
    state = _read_state()
    state["user"] = name or None
    _write_state(state)


# --- Style files -------------------------------------------------------------
def load_style(kind: str, name: str) -> dict:
    data = yaml.safe_load(_style_path(kind, name).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_active_styles() -> list:
    """Active style dicts in merge order: publisher then user (skipping unset)."""
    state = get_active()
    styles = []
    if state["publisher"]:
        styles.append(load_style("publisher", state["publisher"]))
    if state["user"]:
        styles.append(load_style("user", state["user"]))
    return styles


def style_text(kind: str, name: str) -> str:
    return _style_path(kind, name).read_text(encoding="utf-8")


def write_style(kind: str, name: str, text: str) -> None:
    path = _style_path(kind, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate(text: str):
    """(ok, error): the text is valid YAML and parses to a mapping."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return False, str(exc)
    if not isinstance(data, dict):
        return False, "Top-level YAML must be a mapping (key: value)."
    return True, ""


def save_user_style(name: str, text: str) -> None:
    write_style("user", name, text)


def delete_user_style(name: str) -> None:
    _style_path("user", name).unlink(missing_ok=True)
