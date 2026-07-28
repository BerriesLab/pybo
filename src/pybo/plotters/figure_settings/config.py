"""Assembled figure configuration read by every plotter.

Resolves the settings in memory as ``defaults -> active publisher style -> active user
style`` (pybo/plotters/figure_settings/store.py), derives per-plot figsize and scales line
widths, and pushes the resolved ``rcparams`` into matplotlib. Plotters import this module
at startup and read ``fig_cfg`` (and ``scale_figsize``).
"""
import copy

from pybo.plotters.figure_settings.store import (
    load_active_styles, load_app_defaults, load_package_defaults,
)


def _deep_merge(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


# --- Resolve: package defaults + app defaults + publisher style + user style --
fig_cfg = copy.deepcopy(load_package_defaults())
_deep_merge(fig_cfg, load_app_defaults())
for _style in load_active_styles():
    _deep_merge(fig_cfg, _style)

# rcParams travel with the settings tree but are applied to matplotlib, not read as fig_cfg.
_rcparams = fig_cfg.pop("rcparams", {})
# `description` is style metadata (shown in the GUI), not a plot setting.
fig_cfg.pop("description", None)

# --- Size: derive figsize from the resolved column width ---------------------
# Each plot declares only an aspect ratio; column_width_in supplies the one physical
# width (defaults 10 in; publisher styles set the journal column). _DEFAULT_WIDTH only
# guards a settings tree that somehow drops the key.
_DEFAULT_WIDTH = 10.0
_column_width = fig_cfg.pop("column_width_in", None) or _DEFAULT_WIDTH


def scale_figsize(w: float, h: float) -> list:
    """[width, height] in inches for a plot of aspect ratio w:h, sized to the resolved
    column width. For plots that compute their shape at runtime (grid layouts,
    correlation matrices) instead of reading fig_cfg["aspect"]."""
    return [_column_width, round(_column_width * h / w, 3)]


fig_cfg["figsize"] = {
    _name: [_column_width, round(_column_width / _aspect, 3)]
    for _name, _aspect in fig_cfg["aspect"].items()
}

# --- Style: scale point-based elements to the physical figure size -----------
# Line widths and marker sizes are absolute points and do NOT scale with the figure, so
# on a small column the screen-tuned defaults look heavy. linewidth_scale (< 1) thins
# every line proportionally (traces, grid, reference lines, marker edges); marker *area*
# is set via scatter.marker_size.
_lw_scale = fig_cfg.pop("linewidth_scale", None)
if _lw_scale and _lw_scale != 1:
    for _section in ("signal", "refline", "histogram"):
        for _style in fig_cfg.get(_section, {}).values():
            if "linewidth" in _style:
                _style["linewidth"] = round(_style["linewidth"] * _lw_scale, 3)
    _grid = fig_cfg.get("grid", {})
    if "linewidth" in _grid:
        _grid["linewidth"] = round(_grid["linewidth"] * _lw_scale, 3)
    _scatter = fig_cfg.get("scatter", {})
    if "edge_width" in _scatter:
        _scatter["edge_width"] = round(_scatter["edge_width"] * _lw_scale, 3)

# --- rcParams: font family, spines, savefig, ... (values plotters do not pass) ---
if _rcparams:
    import matplotlib
    matplotlib.rcParams.update(_rcparams)
