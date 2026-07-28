"""Matplotlib keyword dicts, derived entirely from the resolved figure settings.

This module holds no style values of its own. Every colour, marker, size and label
comes from ``fig_cfg`` (pybo/plotters/figure_settings_app/defaults.yaml, overlaid by
the active publisher and user styles), so changing a style file changes the plots.
Its only job is the composition YAML cannot express: pairing a role's colour with its
marker, and scaling the Pareto stars off the base marker size.
"""
from pybo.plotters.figure_settings.config import fig_cfg

_c = fig_cfg["colors"]
_m = fig_cfg["markers"]
_s = fig_cfg["scatter"]

C_FEASIBLE = _c["feasible"]
C_INFEASIBLE = _c["infeasible"]
C_PARETO = _c["pareto"]
C_GP_MEAN = _c["gp_mean"]
C_GT = _c["ground_truth"]
C_FUTURE = _c["future"]
C_EVOLUTION = _c["evolution"]

S_OBS = _s["marker_size"]
S_GT = _s["ground_truth_size"]
S_PARETO = S_OBS * _s["pareto_factor"]
S_GT_PARETO = S_GT * _s["pareto_factor"]
EDGE_DARK = _s["edge_color"]
EDGE_WIDTH = _s["edge_width"]


def _obs(role: str, color: str, size: float) -> dict:
    return {
        "marker": _m[role],
        "facecolor": color,
        "edgecolor": EDGE_DARK,
        "s": size,
        **fig_cfg["observation"][role],
    }


def _gt(role: str, color: str, size: float) -> dict:
    cfg = dict(fig_cfg["ground_truth"][role])
    edge_width = cfg.pop("edge_width", None)
    kwargs = {
        "marker": _m["ground_truth_infeasible" if role == "infeasible" else role],
        "facecolor": color,
        "edgecolor": EDGE_DARK if edge_width else None,
        "s": size,
        **cfg,
    }
    if edge_width:
        kwargs["linewidths"] = edge_width
    return kwargs


experiment_scatter_observations_feasible = _obs("feasible", C_FEASIBLE, S_OBS)
experiment_scatter_observations_infeasible = _obs("infeasible", C_INFEASIBLE, S_OBS)
experiment_scatter_observations_pareto_front = _obs("pareto", C_PARETO, S_PARETO)
experiment_scatter_observations_best_value = _obs("best_value", C_PARETO, S_PARETO)

experiment_scatter_gnd_truth_feasible = _gt("feasible", C_GT, S_GT)
experiment_scatter_gnd_truth_infeasible = _gt("infeasible", C_INFEASIBLE, S_GT)
experiment_scatter_gnd_truth_pareto_front = _gt("pareto", C_PARETO, S_GT_PARETO)
experiment_scatter_gnd_truth_best_value = _gt("best_value", C_PARETO, S_GT_PARETO)

gp_mean = {"color": C_GP_MEAN, **fig_cfg["gp"]["mean"]}
gp_confidence_interval_1sigma = {"color": C_GP_MEAN, **fig_cfg["gp"]["band_1sigma"]}
gp_confidence_interval_2sigma = {"color": C_GP_MEAN, **fig_cfg["gp"]["band_2sigma"]}
gp_confidence_interval_3sigma = {"color": C_GP_MEAN, **fig_cfg["gp"]["band_3sigma"]}

next_X_1d = {"color": C_FUTURE, **fig_cfg["next_X"]["line_1d"]}
next_X_2d = {
    "marker": _m["next_X"],
    "facecolor": C_FUTURE,
    "edgecolors": EDGE_DARK,
    "s": S_OBS,
    **fig_cfg["next_X"]["marker_2d"],
}

arrow_future = {"color": C_FUTURE, **fig_cfg["arrow"]["future"]}
arrow_past = dict(fig_cfg["arrow"]["past"])

acqf_1d = {"s": S_GT, **fig_cfg["acqf"]["line_1d"]}

metrics_line2d = {"markerfacecolor": C_PARETO, **fig_cfg["metrics"]["line"]}

evolution_scatter = {"color": C_EVOLUTION, "s": S_OBS, **fig_cfg["evolution"]["scatter"]}
evolution_interconnections = dict(fig_cfg["evolution"]["interconnection"])

general_axline = dict(fig_cfg["refline"]["initial_samples"])
