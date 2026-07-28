"""Figure-settings store — a matplotlib-free file manager.

Shipped with the package (generic):
* ``defaults.yaml``          — generic base settings (dpi, font, grid, legend, scatter…).
* ``publisher_styles/*.yaml``— built-in journal styles (partial overrides + rcparams).

Provided by the host application (see APP_DIR below):
* ``defaults.yaml``          — the app's domain settings, merged over the package base.

A *style* is a YAML mapping that is a partial of the defaults tree (any subset of keys,
plus an optional ``rcparams`` section and ``description``). The assembler
(``pybo.plotters.figure_settings.config``) resolves ``defaults -> publisher style`` in
memory. This module only reads/writes the files, so it stays free of matplotlib and is
safe to import from a GUI.

Which style is active is not stored here: pyBO's default is a committed constant
(``config.DEFAULT_PUBLISHER``, overridable per run with ``--style``) rather than
per-machine runtime state, so there is no state file and no user-style layer.
"""
import yaml
from pathlib import Path

_PKG_DIR = Path(__file__).parent

# --- Application seam: the single place the package points at the host app. -------
# On extraction as a standalone package, replace this with configuration provided by
# the host application (a path passed in, an env var, or a config file). Everything
# below it is either shipped with the package (generic) or lives in the host app.
APP_DIR = _PKG_DIR.parent

# Shipped with the package (generic):
PACKAGE_DEFAULTS_PATH = _PKG_DIR / "defaults.yaml"
PUBLISHER_DIR = _PKG_DIR / "publisher_styles"
# Provided by the host application:
APP_DEFAULTS_PATH = APP_DIR / "defaults.yaml"


def _style_path(name: str) -> Path:
    return PUBLISHER_DIR / f"{name}.yaml"


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


# --- Style files -------------------------------------------------------------
def load_style(name: str) -> dict:
    data = yaml.safe_load(_style_path(name).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def style_text(name: str) -> str:
    return _style_path(name).read_text(encoding="utf-8")


def write_style(name: str, text: str) -> None:
    path = _style_path(name)
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
