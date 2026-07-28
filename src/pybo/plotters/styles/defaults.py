"""pyBO figure settings — the base every figure starts from.

A *style* in this folder is a partial override of this module, chosen with ``--style``
(default: ieee_double). Resolution is a two-step deep merge:
    defaults.SETTINGS -> <style>.SETTINGS
See pybo/plotters/style.py.

Every entry under the semantic sections is the exact keyword mapping matplotlib
receives; the plotters spread it unchanged (``**fig_cfg["gp"]["mean"]``) and hold no
style values of their own.
"""
from torch._C._return_types import linalg_eig

# --- Semantic palette --------------------------------------------------------
# Pick one by name. Every block below refers to the roles, never to a literal colour,
# so switching the line below repaints every figure.
#
# Each palette fills the same six roles. Two rules to keep if you add or edit one:
#   * ground_truth must stay a neutral gray, lighter than gp_mean. The GP posterior is
#     drawn on top of the dense ground-truth trace, so a dark ground truth swallows it.
#   * feasible, infeasible and pareto appear together in one scatter. They are told
#     apart by marker as well as hue (o / X / *), which is what carries the distinction
#     when the colors are this soft — keep those markers distinct.
PALETTE = "dusty"

PALETTES = {
    # Paul Tol's muted qualitative scheme — soft, designed to work together.
    "tol_muted": dict(
        feasible="#44AA99",  # muted teal
        infeasible="#CC6677",  # muted rose
        pareto="#DDCC77",  # sand
        gp_mean="#4477AA",  # muted blue
        ground_truth="#8A8F98",  # neutral grey
        future="#AA4499",  # muted purple
    ),
    # Nord — cool and very low chroma; the quietest of the four.
    "nordic": dict(
        feasible="#8FBCBB",  # pale teal
        infeasible="#BF616A",  # dusty red
        pareto="#EBCB8B",  # soft gold
        gp_mean="#5E81AC",  # slate blue
        ground_truth="#8E95A3",  # neutral gray
        future="#B48EAD",  # muted mauve
    ),
    # Earthy and warm — sage, terracotta, ochre.
    "dusty": dict(
        feasible="#7FA98F",  # sage
        infeasible="#C98383",  # terracotta
        pareto="#D9A55C",  # ochre
        gp_mean="#5B7FA6",  # dusty blue
        ground_truth="#918C84",  # warm gray
        future="#9B84B8",  # dusty lilac
    ),
    # The original flat-ui colors: high chroma, loud next to the three above. Kept
    # because it is the most colourblind-separable set measured (worst pair dE 9.1
    # under deuteranopia, against roughly 6-7 for the soft palettes).
    "vivid": dict(
        feasible="#2ecc71",  # emerald green
        infeasible="#e74c3c",  # alizarin red
        pareto="#f39c12",  # orange
        gp_mean="#2980b9",  # belize blue
        ground_truth="#7F8C8D",  # neutral gray
        future="#7B3FF2",  # violet
    ),
}

_p = PALETTES[PALETTE]
FEASIBLE = _p["feasible"]  # feasible observations
INFEASIBLE = _p["infeasible"]  # infeasible observations
PARETO = _p["pareto"]  # Pareto front / best value
GP_MEAN = _p["gp_mean"]  # GP posterior mean and bands
GROUND_TRUTH = _p["ground_truth"]  # dense true-function background
FUTURE = _p["future"]  # proposed next X, forward arrows
EDGE = "black"  # marker outline

# --- Marker sizes (points^2) -------------------------------------------------
# Absolute, and deliberately not scaled by linewidth_scale: scaling an area by a linear
# factor would be wrong. A style that needs smaller markers overrides these entries.
MARKERSIZE_OBS = 8  # a measured observation
MARKERSIZE_GT = 4
S_OBS = MARKERSIZE_OBS ** 2  # a measured observation
S_GT = MARKERSIZE_GT ** 2  # one point of the dense true-function background
PARETO_FACTOR = 3  # Pareto and best-value stars, relative to the above
S_PARETO = S_OBS * PARETO_FACTOR
S_GT_PARETO = S_GT * PARETO_FACTOR
LINEWIDTH_METRICS = 1
LINEWIDTH_EVOL = LINEWIDTH_METRICS

SETTINGS = {
    # --- Output ---
    "dpi": 600,
    # File type every figure is written as; save_figure() applies it to the stem each
    # plotter chooses, so the extension is set here and nowhere else. A style may
    # override it (a journal wanting vector line art sets "pdf"), and --format wins over
    # both. Note dpi only affects rasterized content once the format is vector.
    "format": "png",
    # Padding (in font-size units) passed to tight_layout. matplotlib's default is 1.08;
    # a smaller value trims the white border while keeping the exact figsize.
    "layout_pad": 0.4,
    # Physical width of every figure, in inches (height follows each plot's aspect
    # ratio). Publisher styles override this with the journal's column width.
    "column_width_in": 10.0,
    # Multiplies the line widths of the sections named in scaled_sections. 1.0 = no
    # scaling; publisher styles thin lines for narrow columns (e.g. 0.65).
    "linewidth_scale": 1.0,
    # Same idea for markers. Linear: style.py squares it for scatter's `s`, which is an
    # area in points^2, and applies it directly to `markersize`, which is a diameter.
    "marker_scale": 1.0,
    # Every section style.py walks when applying the two scales above. A new semantic
    # section must be listed here or its strokes and markers will not follow the column.
    "scaled_sections": ["observation", "ground_truth", "gp", "next_X", "arrow",
                        "acqf", "metrics", "evolution", "refline"],

    # --- matplotlib rcParams (font family, spines, tick direction, savefig, ...) ---
    # Empty base = matplotlib defaults. Publisher styles override these; they are pushed
    # into matplotlib.rcParams by style.py rather than passed by the plotters.
    "rcparams": {},

    # --- Figure aspect ratios (width / height) ---
    # Shape only, never size: the physical width comes from column_width_in, and the
    # height is derived as width / aspect. fig_cfg["figsize"][name] (the [w, h] in inches
    # every plotter reads) is built from these in style.py. One key per plotter.
    "aspect": {
        "experiment_1d": 5 / 4,  # Experiment1DPlotter
        "experiment_2d": 5 / 4,  # Experiment2DPlotter
        "pareto_front_2d": 5 / 4,  # ParetoFront2DPlotter
        "acqf_1d": 5 / 4,  # Acqf1DPlotter
        "acqf_2d": 5 / 4,  # Acqf2DPlotter
        "evolution": 5 / 4,  # EvolutionPlotter (one per parameter/result)
        "best_value": 5 / 4,  # BestValuePlotter
        "hypervolume": 5 / 4,  # HypervolumePlotter
        "hypervolume_improvement": 5 / 4,  # HypervolumeImprovementPlotter
        "elapsed_time": 5 / 4,  # ElapsedTimePlotter
    },

    # --- Colormaps ---
    # Every surface and z-colored scatter in pyBO encodes magnitude (a posterior mean,
    # an acquisition value, a parameter), so they all take "sequential", which must be a
    # single hue light->dark. Before these settings existed four plotters hardcoded
    # 'coolwarm' and Acqf2DPlotter used 'viridis'; viridis is the correct one for
    # magnitude (perceptually uniform and colourblind-safe), so it is now the default for
    # all of them. Set this to coolwarm to restore the old look.
    # "diverging" is here for data with a meaningful midpoint and is not currently used.
    "cmap": {
        "sequential": "viridis",
        "diverging": "coolwarm",
    },

    # --- Observations (the measured points) ---
    "observation": {
        "feasible": dict(
            marker="o",
            facecolor=FEASIBLE,
            edgecolor=EDGE,
            s=S_OBS,
            alpha=0.8,
            label="Feasible Obs"
        ),
        "infeasible": dict(
            marker="X",
            facecolor=INFEASIBLE,
            edgecolor=EDGE,
            s=S_OBS,
            alpha=0.8,
            label="Infeasible Obs"
        ),
        "pareto": dict(
            marker="*",
            facecolor=PARETO,
            edgecolor=EDGE,
            s=S_PARETO,
            alpha=0.8,
            label="Pareto Obs"
        ),
        "best_value": dict(
            marker="*",
            facecolor=PARETO,
            edgecolor=EDGE,
            s=S_PARETO,
            alpha=0.8,
            label="Best Obs"
        ),
    },

    # --- Ground truth (the dense true-function background: thousands of tiny points) ---
    "ground_truth": {
        "feasible": dict(
            marker="o",
            facecolor=GROUND_TRUTH,
            edgecolor=None,
            s=S_GT,
            alpha=0.25,
            label="Feasible GT"
        ),
        "infeasible": dict(
            marker="x",
            facecolor=INFEASIBLE,
            edgecolor=None,
            s=S_GT,
            alpha=0.15,
            label="Infeasible GT"
        ),
        "pareto": dict(
            marker="*",
            facecolor=PARETO,
            edgecolor=EDGE,
            linewidths=0.6,
            s=S_GT_PARETO,
            alpha=0.7,
            label="Pareto GT"
        ),
        "best_value": dict(
            marker="*",
            facecolor=PARETO,
            edgecolor=EDGE,
            linewidths=0.6,
            s=S_GT_PARETO,
            alpha=0.7,
            label="Best GT Value"
        ),
    },

    # --- GP posterior (1D plots); bands fade as they widen ---
    "gp": {
        "mean": dict(
            color=GP_MEAN,
            label=r"GP $\mu$"
        ),
        "band_1sigma": dict(
            color=GP_MEAN,
            alpha=0.10,
            label=r"GP $\pm 1 \sigma$"
        ),
        "band_2sigma": dict(
            color=GP_MEAN,
            alpha=0.05,
            label=r"GP $\pm 2 \sigma$"
        ),
        "band_3sigma": dict(
            color=GP_MEAN,
            alpha=0.02,
            label=r"GP $\pm 3 \sigma$"
        ),
    },

    # --- The proposed next X ---
    "next_X": {
        "line_1d": dict(
            color=FUTURE,
            linestyle=":",
            linewidth=2,
            label=r"New $X$"
        ),
        "marker_2d": dict(
            marker="*",
            facecolor=FUTURE,
            edgecolors=EDGE,
            s=S_OBS,
            alpha=0.8,
            label="Next X"
        ),
    },

    # --- Trajectory arrows between successive proposals ---
    # "past" is drawn over the colored surface, so it stays white for contrast.
    "arrow": {
        "future": dict(
            arrowstyle="->",
            color=FUTURE,
            lw=1.5,
            alpha=0.8,
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=0.1",
            ls="--"
        ),
        "past": dict(
            arrowstyle="->",
            color="white",
            lw=1.5,
            alpha=0.8,
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=0.1"
        ),
    },

    # --- Acquisition function trace (1D) ---
    "acqf": {
        "line_1d": dict(
            color=EDGE,
            s=S_GT,
            label="Acqf."
        ),
    },

    # --- Metrics panels (best value, hypervolume, HVI, elapsed time) ---
    "metrics": {
        "line": dict(
            marker="o",
            markerfacecolor=PARETO,
            markeredgecolor=EDGE,
            markersize=MARKERSIZE_OBS,
            linestyle="-",
            linewidth=LINEWIDTH_METRICS,
        ),
    },

    # --- Evolution panels (one per parameter / objective / constraint) ---
    "evolution": {
        "scatter": dict(
            marker="o",
            color=PARETO,
            edgecolor=EDGE,
            s=S_OBS,
            alpha=1,
            linestyle="-",
            linewidth=LINEWIDTH_EVOL,
        ),
        "interconnection": dict(
            linestyle="-",
            linewidth=LINEWIDTH_EVOL,
            alpha=0.8
        ),
    },

    # --- Reference lines ---
    "refline": {
        "initial_samples": dict(
            linestyle="--",
            linewidth=1,
            color=EDGE,
            alpha=0.5
        ),
    },
}
