"""Legend placement shared by the plots that can carry more series than a corner
comfortably holds - the two Pareto plots and the single-objective landscape.

The legend always stays inside the axes (loc="best" by default, or a fixed corner
where the caller wants it to stay put next to a colorbar) rather than being pushed
out into a margin below the plot: a crowded legend may sit over some data, which
reads fine since the data is still visible underneath it, but it must never spill
past the axes or the figure the way a legend drawn at a fixed font regardless of
entry count can. So instead of moving it, this shrinks it - font size and column
count chosen from the entry count and label length alone, aiming for a legend that
fits within a generous fraction of the figure's own size.

That sizing is an estimate, not a measurement: this module never renders a
candidate legend and checks how big it came out, because that size is only true
once a backend has actually drawn it, and that happens at a time (and DPI) this
module does not control - an interactive backend can rescale its renderer once a
window is genuinely on screen (a Retina display's backing-scale factor, applied
only on show by some backends, is exactly this), which quietly invalidates a size
measured beforehand. An estimate from font metrics has no such moment to be caught
before: it is known before the legend is even drawn, so it is right on the very
first frame, identically on every backend.
"""

# Legend geometry as multiples of the legend's own font size, in points - matplotlib's
# own defaults (legend.borderpad, labelspacing, handlelength, ...) rounded up a bit,
# since a slightly generous estimate costs a little size and a slightly tight one
# costs an overlap with the axes edge instead of just with the data.
_ROW_HEIGHT_EM = 2.0
_VPAD_EM = 1.4
_CHAR_WIDTH_EM = 0.62
_HANDLE_EM = 3.4
_COL_TRAILING_EM = 0.6
_COL_SPACING_EM = 2.0
_HPAD_EM = 0.8

# What "fits inside the plot" is targeted against: a fraction of the *figure's* own
# size rather than the axes' - simpler (no render needed to know it) and safely
# smaller than the axes already are, since axis labels and ticks take some of the
# figure's edges the legend never has to share space with anyway.
_TARGET_WIDTH_FRAC = 0.85
_TARGET_HEIGHT_FRAC = 0.55

_MIN_FONTSIZE = 5.0
_MAX_NCOL = 4


def _fit_legend_font(handles, fig_w_in: float, fig_h_in: float, base_fontsize: float):
    """The largest (fontsize, ncol) - up to `base_fontsize` - that keeps an estimated
    legend of `handles` within a generous fraction of the figure's size.

    Tries each column count and picks whichever leaves the legend most legible
    (largest resulting font) rather than fixing the column count by entry count
    alone: a handful of long labels needs different columns than many short ones.
    """
    n = len(handles)
    max_chars = max((len(h.get_label()) for h in handles), default=1)
    target_w_pt = _TARGET_WIDTH_FRAC * fig_w_in * 72.0
    target_h_pt = _TARGET_HEIGHT_FRAC * fig_h_in * 72.0

    best_fs, best_ncol = _MIN_FONTSIZE, 1
    for ncol in range(1, min(_MAX_NCOL, n) + 1):
        rows = -(-n // ncol)  # ceil
        h_coef = rows * _ROW_HEIGHT_EM + _VPAD_EM
        w_coef = (ncol * (_HANDLE_EM + _COL_TRAILING_EM) + ncol * max_chars * _CHAR_WIDTH_EM
                 + (ncol - 1) * _COL_SPACING_EM + _HPAD_EM)
        fs = min(base_fontsize, target_h_pt / h_coef, target_w_pt / w_coef)
        if fs > best_fs:
            best_fs, best_ncol = fs, ncol

    return max(best_fs, _MIN_FONTSIZE), best_ncol


def place_legend(fig, ax, handles, leg_cfg, fontsize, loc="best"):
    """Place `handles` as one legend on `ax`, sized to fit inside the plot.

    Always inside the axes - see the module docstring for why a legend that would
    otherwise be too big is shrunk rather than moved outside them.
    """
    fig_w, fig_h = fig.get_size_inches()
    fs, ncol = _fit_legend_font(handles, fig_w, fig_h, fontsize)
    return ax.legend(handles=handles, fontsize=fs, loc=loc, ncol=ncol,
                     frameon=leg_cfg["frameon"], framealpha=leg_cfg["framealpha"])
