"""ACS (single column) — a partial override of defaults.SETTINGS."""

SETTINGS = {
    'dpi': 600,
    'column_width_in': 3.33,
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
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    },
}
