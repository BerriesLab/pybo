"""IEEE (double column) — a partial override of defaults.SETTINGS."""

SETTINGS = {
    'dpi': 600,
    'column_width_in': 7.16,
    'linewidth_scale': 0.65,
    "rcparams": {
        'mathtext.fontset': 'dejavusans',
        'axes.linewidth': 0.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'savefig.dpi': 600,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    },
}
