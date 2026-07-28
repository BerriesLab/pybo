"""Assembled figure configuration read by every plotter.

Resolves the settings in memory as ``defaults -> active publisher style -> active user
style`` (pybo/plotters/figure_settings/store.py), derives per-plot figsize and scales line
widths, and pushes the resolved ``rcparams`` into matplotlib. Plotters read ``fig_cfg``
(and ``scale_figsize``) at draw time.

``fig_cfg`` is resolved once on import from the selections in ``state.json``. Call
``resolve()`` to rebuild it for a different style — it mutates the dict in place, so
modules that already did ``from ... import fig_cfg`` see the new values. That is what lets
a ``--style`` flag take effect after argparse has run, long after import.
"""
import copy

from pybo.plotters.figure_settings.store import (
    load_active_styles, load_app_defaults, load_package_defaults, load_style,
)

# Guards a settings tree that somehow drops column_width_in.
_DEFAULT_WIDTH = 10.0

fig_cfg: dict = {}
_column_width: float = _DEFAULT_WIDTH


def _deep_merge(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def scale_figsize(w: float, h: float) -> list:
    """[width, height] in inches for a plot of aspect ratio w:h, sized to the resolved
    column width. For plots that compute their shape at runtime (grid layouts,
    correlation matrices) instead of reading fig_cfg["aspect"]."""
    return [_column_width, round(_column_width * h / w, 3)]


def resolve(publisher: str | None = None, user: str | None = None) -> dict:
    """Rebuild ``fig_cfg`` in place and re-apply rcParams.

    With no arguments the active selections in ``state.json`` are used. Passing a name
    overrides that selection for this process only — nothing is written back to disk.
    """
    global _column_width

    resolved = copy.deepcopy(load_package_defaults())
    _deep_merge(resolved, load_app_defaults())
    if publisher is None and user is None:
        styles = load_active_styles()
    else:
        styles = []
        if publisher:
            styles.append(load_style("publisher", publisher))
        if user:
            styles.append(load_style("user", user))
    for style in styles:
        _deep_merge(resolved, style)

    # rcParams travel with the settings tree but are applied to matplotlib, not read
    # as fig_cfg. `description` is style metadata (shown in the GUI), not a plot setting.
    rcparams = resolved.pop("rcparams", {})
    resolved.pop("description", None)

    # --- Size: derive figsize from the resolved column width ---------------------
    # Each plot declares only an aspect ratio; column_width_in supplies the one physical
    # width (defaults 10 in; publisher styles set the journal column).
    _column_width = resolved.pop("column_width_in", None) or _DEFAULT_WIDTH
    resolved["figsize"] = {
        name: [_column_width, round(_column_width / aspect, 3)]
        for name, aspect in resolved["aspect"].items()
    }

    # --- Style: scale point-based elements to the physical figure size -----------
    # Line widths are absolute points and do NOT scale with the figure, so on a small
    # column the screen-tuned defaults look heavy. linewidth_scale (< 1) thins every
    # line proportionally. Which sections to walk is declared by the settings tree
    # itself (scaled_sections), so the host application names its own; marker sizes are
    # absolute and are left alone.
    lw_scale = resolved.pop("linewidth_scale", None)
    sections = resolved.pop("scaled_sections", [])
    if lw_scale and lw_scale != 1:
        for section in sections:
            for style in resolved.get(section, {}).values():
                if isinstance(style, dict) and "linewidth" in style:
                    style["linewidth"] = round(style["linewidth"] * lw_scale, 3)
        grid = resolved.get("grid", {})
        if "linewidth" in grid:
            grid["linewidth"] = round(grid["linewidth"] * lw_scale, 3)
        scatter = resolved.get("scatter", {})
        if "edge_width" in scatter:
            scatter["edge_width"] = round(scatter["edge_width"] * lw_scale, 3)

    # Mutate in place: callers hold a reference to this exact dict.
    fig_cfg.clear()
    fig_cfg.update(resolved)

    # --- rcParams: font family, spines, savefig, ... (values plotters do not pass) ---
    if rcparams:
        import matplotlib
        matplotlib.rcParams.update(rcparams)

    return fig_cfg


resolve()
