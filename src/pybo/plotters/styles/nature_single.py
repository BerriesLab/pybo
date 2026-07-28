"""Nature (single column) — a partial override of defaults.SETTINGS."""

SETTINGS = {
    'dpi': 600,
    'column_width_in': 3.504,
    'linewidth_scale': 0.65,
    'marker_scale': 0.65,
    "rcparams": {
        'mathtext.fontset': 'dejavusans',
        'axes.linewidth': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        # Type sizes this journal specifies. Applied globally through
        # rcParams, so no plotter passes a fontsize.
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'legend.fontsize': 6,
        'legend.frameon': True,
        'legend.framealpha': 1.0,
        'legend.loc': 'upper right',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    },
}
