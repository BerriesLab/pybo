C_FEASIBLE = '#2ecc71'  # Emerald Green
C_INFEASIBLE = '#e74c3c'  # Alizarin Red
C_PARETO = '#f39c12'  # Orange
C_GP_MEAN = '#2980b9'  # Belize Blue
C_GT = '#2c3e50'  # Midnight Blue (for Ground Truth)
C_FUTURE = '#9b59b6'  # Amethyst Purple (for Next X/Trajectory)

ms = 8
ms_gt = ms / 2

scatter_observations_feasible = {
    'facecolor': C_FEASIBLE,
    'edgecolor': 'white',
    'linewidth': 0.8,
    'marker': 'o',
    's': ms ** 2,
    'label': "Feasible Obs.",
    'alpha': 0.9,
}

scatter_observations_infeasible = {
    'facecolor': 'none',
    'edgecolor': C_INFEASIBLE,
    'linewidth': 1.5,
    'marker': 'x',
    's': ms ** 2,
    'label': "Infeasible Obs.",
    'alpha': 0.7,
}

optimum = {
    'facecolor': 'gold',
    'edgecolor': 'black',
    'marker': '*',  # Star for the best found
    's': ms ** 2 * 1.5,
    'label': "Current Best",
    'zorder': 5
}

feasible_pareto_objectives_kwargs = {
    'color': 'tab:orange',
    'marker': 'D',
    's': ms ** 2,
    'edgecolors': 'black',
    'alpha': 0.7,
    'label': 'Pareto Obs.'
}
feasible_non_pareto_objectives_kwargs = {
    'color': "tab:green",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Non-Pareto Obs.'
}
infeasible_objectives_kwargs = {
    'color': "tab:red",
    'marker': "x",
    's': ms ** 2,
    "alpha": 0.7,
    'label': 'Inf. Obs.'
}
ref_point_kwargs = {
    'color': 'tab:red',
    "edgecolors": "black",
    'marker': 's',
    's': ms ** 2,
    'alpha': 0.7,
    'label': 'Ref. Point'
}
feasible_pareto_ground_truth_kwargs = {
    'color': "black",
    'marker': "D",
    's': ms ** 2 / 5,
    "alpha": 1,
    'label': 'Pareto GT'
}
feasible_non_pareto_ground_truth_kwargs = {
    'color': "black",
    'marker': "o",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Non-Pareto GT'
}
infeasible_ground_truth_kwargs = {
    'color': "red",
    'marker': "x",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Inf. GT'
}
posterior_pareto_kwargs = {
    'fmt': 'o',
    'edgecolors': 'tab:blue',
    'alpha': 0.3,
    'label': r'Post. $\mu \pm 3 \sigma$',
    'capsize': 3,
}
xy_plot_kwargs = {
    'marker': 'o',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolor': 'black',
    'alpha': 1,
    'linestyle': '-',
}
line2d_plot_kwargs = {
    'marker': 'o',
    'markersize': ms,
    'color': 'tab:orange',
    'markeredgecolor': 'black',
    'alpha': 1,
    'linestyle': '-',
}

gp_mean = {
    'color': 'blue',
    'label': r'GP $\mu$',
}
gp_confidence_interval_1sigma = {
    'color': 'blue',
    'alpha': 0.1,
    'label': r'GP $\pm 1 \sigma$',
}
gp_confidence_interval_2sigma = {
    'color': 'blue',
    'alpha': 0.05,
    'label': r'GP $\pm 2 \sigma$',
}
gp_confidence_interval_3sigma = {
    'color': 'blue',
    'alpha': 0.02,
    'label': r'GP $\pm 3 \sigma$',
}
next_X_1d = {
    'color': 'green',
    'linestyle': ':',
    'linewidth': 2,
    'label': r"New $X$",
}
next_X_2d = {
    "facecolor": 'red',
    "edgecolors": 'black',
    "marker": '*',
    "s": ms ** 2,
    "label": 'Next X',
    "alpha": 0.8,
}
optimum = {
    'color': 'orange',
    'marker': 'D',
    's': ms ** 2,
    'edgecolors': 'black',
    'label': "Optimum",
}
acqf_1d = {
    'color': 'black',
    'linewidth': 1.5,
    'label': 'Acqf.'
}
arrow_future = {
    'arrowstyle': '->',
    'color': 'red',
    'lw': 1.5,
    'alpha': 0.8,
    'shrinkA': 3,
    'shrinkB': 3,
    'connectionstyle': "arc3,rad=0.1",
    'ls': '--'
}
arrow_past = {
    'arrowstyle': '->',
    'color': 'white',
    'lw': 1.5,
    'alpha': 0.8,
    'shrinkA': 3,
    'shrinkB': 3,
    'connectionstyle': "arc3,rad=0.1",
}
scatter_observations_infeasible = {
    'facecolor': 'red',
    'edgecolor': 'black',
    'marker': 'X',
    's': ms ** 2,
    'label': "Infeasible obs.",
    'alpha': 0.8,
}
scatter_observations_feasible = {
    'facecolor': 'white',
    'edgecolor': 'black',
    'marker': 'o',
    's': ms ** 2,
    'label': "Feasible obs.",
    'alpha': 0.8,
}
scatter_gnd_truth_feasible = {
    'facecolor': 'black',
    'edgecolor': None,
    'marker': "o",
    's': ms_gt ** 2,
    'label': "Feasible GT",
    'alpha': 0.5,
}
scatter_gnd_truth_infeasible = {
    'facecolor': 'red',
    'edgecolor': None,
    'marker': "o",
    's': ms_gt ** 2,
    'label': "Infeasible GT",
}
scatter_gnd_truth_pareto_front = {
    'facecolor': 'tab:orange',
    'edgecolors': 'black',
    'marker': 'o',
    's': ms_gt ** 2,
    'alpha': 0.7,
    'label': 'Pareto GT.'
}
contour_gnd_truth = {
    "levels": 50,
    "cmap": 'viridis',
    "alpha": 0.8
}
contour_gnd_truth_infeasible = {
    "levels": [0.5, 1.0],  # Shades areas where value is between 0.5 and 1.0
    "colors": ['red'],
    "alpha": 0.3,  # Transparency for the "shaded" look
}
