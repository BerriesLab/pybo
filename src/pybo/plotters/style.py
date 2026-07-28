"""The resolved figure configuration every plotter reads.

``styles/defaults.yaml`` is the base; each sibling ``styles/<name>.yaml`` is a partial
override of it, selected with ``--style`` (see pybo/utils/cli.py). Resolving is one deep
merge, after which this module derives each plot's figsize, thins line widths for narrow
columns, and pushes the style's ``rcparams`` into matplotlib.

``fig_cfg`` is resolved on import using ``DEFAULT_STYLE``. Call ``resolve()`` to rebuild it
for a different style: it mutates the dict in place, so modules that already did
``from pybo.plotters.style import fig_cfg`` see the new values. That is what lets
``--style`` take effect after argparse has run, long after import.
"""
import copy
from pathlib import Path

import yaml

STYLE_DIR = Path(__file__).parent / "styles"
DEFAULTS_PATH = STYLE_DIR / "defaults.yaml"

# The style used when nothing asks for one. A project decision, so it lives in version
# control rather than in a runtime state file; pybo.utils.cli reads it as the default for
# --style, which keeps a plain `import pybo.plotters...` and a CLI run on the same style.
DEFAULT_STYLE = "ieee_double"

# Guards a settings tree that somehow drops column_width_in.
_DEFAULT_WIDTH = 10.0

fig_cfg: dict = {}
_column_width: float = _DEFAULT_WIDTH


def list_styles() -> list:
    """Names accepted by resolve() / --style, i.e. every YAML here but the base."""
    return sorted(p.stem for p in STYLE_DIR.glob("*.yaml") if p != DEFAULTS_PATH)


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


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


def resolve(style: str | None = None) -> dict:
    """Rebuild ``fig_cfg`` in place and re-apply rcParams.

    With no argument ``DEFAULT_STYLE`` is used. The choice affects this process only —
    nothing is written to disk.
    """
    global _column_width

    cfg = copy.deepcopy(_load(DEFAULTS_PATH))
    _deep_merge(cfg, _load(STYLE_DIR / f"{style or DEFAULT_STYLE}.yaml"))

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


resolve()