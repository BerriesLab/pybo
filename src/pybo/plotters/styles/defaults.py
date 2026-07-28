"""pyBO figure settings — the base every figure starts from.

A *style* in this folder is a partial override of this module, chosen with ``--style``
(default: ieee_double). Resolution is a two-step deep merge:
    defaults.SETTINGS -> <style>.SETTINGS
See pybo/plotters/style.py.

Every entry under the semantic sections is the exact keyword mapping matplotlib
receives; the plotters spread it unchanged (``**fig_cfg["gp"]["mean"]``) and hold no
style values of their own.
"""

# --- Semantic palette --------------------------------------------------------
# The single source of truth for color; every block below refers to these names.
#
# COLOURBLIND SAFETY — read before editing, and re-measure if you do.
# Pairwise separation measured in OKLab (dE x100) under deuteranopia/protanopia:
#   feasible / pareto        9.1     floor  - legible ONLY via distinct markers
#   feasible / infeasible   11.2     floor  - legible ONLY via distinct markers
#   gp_mean  / future       11.7     floor  - legible via linestyle + marker
# Six simultaneous categorical hues cannot all reach the dE >= 15 comfort target under
# deuteranopia; that is a property of the color space, not a fixable choice. The floor
# (dE >= 8) is therefore held by the markers below acting as secondary encoding: if you
# change a color, keep its marker distinct or the plots stop being colourblind-legible.
FEASIBLE = "#2ecc71"  # Emerald green  - feasible observations
INFEASIBLE = "#e74c3c"  # Alizarin red   - infeasible observations
PARETO = "#f39c12"  # Orange         - Pareto front / best value
GP_MEAN = "#2980b9"  # Belize blue    - GP posterior mean and bands
GROUND_TRUTH = "#2c3e50"  # Midnight blue  - dense true-function background
FUTURE = "#7B3FF2"  # Violet         - proposed next X, forward arrows
EDGE = "black"  # Marker outline

# --- Marker sizes (points^2) -------------------------------------------------
# Absolute, and deliberately not scaled by linewidth_scale: scaling an area by a linear
# factor would be wrong. A style that needs smaller markers overrides these entries.
S_OBS = 64  # a measured observation
S_GT = 1  # one point of the dense true-function background
PARETO_FACTOR = 3  # Pareto and best-value stars, relative to the above
S_PARETO = S_OBS * PARETO_FACTOR
S_GT_PARETO = S_GT * PARETO_FACTOR

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
            markersize=8,
            linestyle="-"
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
            linestyle="-"
        ),
        "interconnection": dict(
            color="gray",
            linestyle="-",
            linewidth=1,
            alpha=0.2
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
