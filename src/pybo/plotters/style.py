"""The resolved figure configuration every plotter reads.

``styles/defaults.py`` is the base; each sibling ``styles/<name>.py`` exposes a
``SETTINGS`` mapping that partially overrides it, selected with ``--style`` (see
pybo/utils/cli.py). Resolving is one deep merge, after which this module derives each
plot's figsize, thins line widths for narrow columns, and pushes the style's
``rcparams`` into matplotlib.

Resolution happens exactly once per process, and deliberately not on import: a CLI run
calls ``resolve()`` with the chosen ``--style`` once argparse has run, and anything else
falls back to ``ensure_resolved()`` when the first plotter is built. ``resolve()`` mutates
``fig_cfg`` in place, so modules that did ``from pybo.plotters.style import fig_cfg`` see
the values either way. Nothing may read ``fig_cfg`` at import time — a module-level read
(a constructor default argument, say) would capture an unresolved or stale value.
"""
import copy
import importlib
from pathlib import Path

STYLE_PKG = "pybo.plotters.styles"
STYLE_DIR = Path(__file__).parent / "styles"

# The style used when nothing asks for one. A project decision, so it lives in version
# control rather than in a runtime state file; pybo.utils.cli reads it as the default for
# --style, which keeps a plain `import pybo.plotters...` and a CLI run on the same style.
DEFAULT_STYLE = "ieee_double"

# Guards a settings tree that somehow drops column_width_in.
_DEFAULT_WIDTH = 10.0

fig_cfg: dict = {}
_column_width: float = _DEFAULT_WIDTH


def list_styles() -> list:
    """Names accepted by resolve() / --style: every module in styles/ but the base."""
    return sorted(p.stem for p in STYLE_DIR.glob("*.py")
                  if p.stem not in ("defaults", "__init__"))


def _load(name: str) -> dict:
    """The SETTINGS mapping of a style module. Imports are cached by Python, so a
    re-resolve costs nothing after the first."""
    return getattr(importlib.import_module(f"{STYLE_PKG}.{name}"), "SETTINGS", {})


def _deep_merge(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def scale_figsize(w: float, h: float) -> list:
    """[width, height] in inches for a plot of aspect ratio w:h, sized to the resolved
    column width. For plots that compute their shape at runtime instead of reading
    fig_cfg["aspect"]."""
    return [_column_width, round(_column_width * h / w, 3)]


def resolve(style: str | None = None, fmt: str | None = None) -> dict:
    """Rebuild ``fig_cfg`` in place and re-apply rcParams.

    With no argument ``DEFAULT_STYLE`` is used. ``fmt`` overrides the file type the
    resolved style asks for (--format). The choice affects this process only — nothing
    is written to disk.
    """
    global _column_width

    # deepcopy both: the modules are imported once and cached, so merging into them
    # would leak one resolve's overrides into the next.
    cfg = copy.deepcopy(_load("defaults"))
    _deep_merge(cfg, copy.deepcopy(_load(style or DEFAULT_STYLE)))
    if fmt:
        cfg["format"] = fmt

    # rcParams travel with the settings but are applied to matplotlib, not read as
    # fig_cfg. `description` is style metadata, not a plot setting.
    rcparams = cfg.pop("rcparams", {})
    cfg.pop("description", None)

    # --- Size: derive figsize from the resolved column width ---------------------
    # Each plot declares only an aspect ratio; column_width_in supplies the one physical
    # width (10 in by default; publisher styles set the journal column).
    _column_width = cfg.pop("column_width_in", None) or _DEFAULT_WIDTH
    cfg["figsize"] = {
        name: [_column_width, round(_column_width / aspect, 3)]
        for name, aspect in cfg["aspect"].items()
    }

    # --- Style: thin point-based strokes to suit the physical figure size --------
    # Line widths are absolute points and do NOT scale with the figure, so on a narrow
    # column the screen-tuned defaults look heavy. Marker sizes are left alone: scaling
    # an area by a linear factor would be wrong, and a style can override `s` directly.
    lw_scale = cfg.pop("linewidth_scale", None)
    for section in cfg.pop("scaled_sections", []) if lw_scale and lw_scale != 1 else []:
        for entry in cfg.get(section, {}).values():
            if isinstance(entry, dict) and "linewidth" in entry:
                entry["linewidth"] = round(entry["linewidth"] * lw_scale, 3)

    # Mutate in place: callers hold a reference to this exact dict.
    fig_cfg.clear()
    fig_cfg.update(cfg)

    if rcparams:
        import matplotlib
        matplotlib.rcParams.update(rcparams)

    return fig_cfg


def ensure_resolved() -> dict:
    """Resolve with ``DEFAULT_STYLE`` unless something already has.

    Resolution is not done on import: a CLI run calls ``resolve()`` with the chosen
    ``--style`` once argparse has run, and this is the fallback for everything else
    (a notebook, a test, library use), so exactly one resolve happens either way.
    """
    if not fig_cfg:
        resolve()
    return fig_cfg
