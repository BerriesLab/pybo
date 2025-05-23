import matplotlib.pyplot as plt
from mobo.mobo import Mobo
from utils.io import *

ms = 7

feasible_non_pareto_observations_kwargs = {
    'color': "tab:green",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Non-Pareto'
}

infeasible_observations_kwargs = {
    'color': "tab:red",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Infeasible'
}

feasible_pareto_observations_kwargs = {
    'marker': 'o',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolor': 'black',
    'alpha': 0.7,
    'label': 'Pareto Front'
}

ref_point_kwargs = {
    'color': 'red',
    'marker': 'x',
    's': ms ** 2,
    'alpha': 0.8,
    'label': 'Ref. Point'
}

ground_truth_feas_kwargs = {
    'color': "black",
    'marker': "o",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Ground Truth'
}

ground_truth_inf_kwargs = {
    'color': "black",
    'marker': "x",
    's': ms ** 2 / 5,
    "alpha": 0.1,
    'label': 'Infeas. G.T.'
}

posterior_pareto_kwargs = {
    'fmt': 'o',
    'ecolor': 'tab:blue',
    'alpha': 0.3,
    'label': r'Post. $\mu \pm 3 \sigma$',
    'capsize': 3,
}

optimization_figures_kwargs = {
    'marker': 'o',
    's': ms ** 2,
    'color': 'tab:orange',
    'edgecolor': 'black',
    'alpha': 1,
}


# TODO: implement function
def plot_objective_from_R1_to_R1():
    raise NotImplementedError("Function not yet implemented.")


# TODO: implement function
def plot_objective_from_R2_to_R1():
    raise NotImplementedError("Function not yet implemented.")


def plot_multi_objective_from_RN_to_R2(
        mobo: Mobo,
        x: torch.Tensor,
        f1_lims=None,
        f2_lims=None,
        f1_label="$f_{01}$",
        f2_label="$f_{02}$",
        show_ground_truth=False,
        show_posterior=False,
        show_ref_point=False,
        show_observations=True,
        display_figures=True):

    # Initialize figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.set_xlabel(f1_label)
    axes.set_ylabel(f2_label)
    axes.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^2$')
    if f1_lims is not None and isinstance(f1_lims, tuple):
        axes.set_xlim(f1_lims[0], f1_lims[1])
    if f2_lims is not None and isinstance(f2_lims, tuple):
        axes.set_ylim(f2_lims[0], f2_lims[1])

    """ Plot ground truth """
    if show_ground_truth:
        if mobo.get_true_objective() is None:
            raise ValueError("Ground truth not available.")

        x = x.to(mobo.get_device(), mobo.get_dtype())
        ground_truth_obj = mobo.get_true_objective()(x)

        # Apply constraint mask
        if mobo.get_constraints() is None:
            ground_truth_feas_mask = torch.ones_like(ground_truth_obj, dtype=torch.bool).all(dim=-1)
        else:
            ground_truth_con = -mobo.get_true_objective().evaluate_slack(x)
            ground_truth_feas_mask = (ground_truth_con <= 0).all(dim=-1)

        # Plot feasible ground truth
        ground_truth_feas_f1 = ground_truth_obj[ground_truth_feas_mask, 0].detach().cpu().numpy()
        ground_truth_feas_f2 = ground_truth_obj[ground_truth_feas_mask, 1].detach().cpu().numpy()
        axes.scatter(ground_truth_feas_f1, ground_truth_feas_f2, **ground_truth_feas_kwargs)

        # Plot infeasible points - Could be enabled
        # ground_truth_inf_mask = torch.logical_not(ground_truth_feas_mask)
        # ground_truth_inf_f1 = ground_truth[ground_truth_inf_mask, 0].detach().cpu().numpy()
        # ground_truth_inf_f2 = ground_truth[ground_truth_inf_mask, 1].detach().cpu().numpy()
        # axes.scatter(ground_truth_inf_f1, ground_truth_inf_f2, **ground_truth_inf_kwargs)

    """ Plot posterior pareto """
    if show_posterior:
        # Predict over test grid
        x = x.to(mobo.get_device(), mobo.get_dtype())
        posterior = mobo.get_model().posterior(x)
        mean = posterior.mean.detach().cpu().numpy()
        std = posterior.variance.sqrt().detach().cpu().numpy()

        # Plot mean with error bars
        axes.errorbar(
            mean[:, 0],
            mean[:, 1],
            xerr=3 * std[:, 0],
            yerr=3 * std[:, 1],
            **posterior_pareto_kwargs,
        )

    """ Plot reference point """
    if show_ref_point is True:
        show_ref_point = mobo.get_ref_point()
        ref_point_f1 = show_ref_point[0].detach().cpu().numpy()
        ref_point_f2 = show_ref_point[1].detach().cpu().numpy()
        plt.scatter(ref_point_f1, ref_point_f2, **ref_point_kwargs)

    """ Plot observations """
    if show_observations is True:
        y_obj = mobo.get_Yobj()

        # Compute masks
        obs_feas_mask = mobo.get_con_mask()
        obs_par_mask = mobo.get_par_mask()
        obs_inf_mask = torch.logical_not(mobo.get_con_mask())
        obs_feas_and_par_mask = torch.logical_and(obs_feas_mask, obs_par_mask)
        obs_feas_and_not_par_mask = torch.logical_and(obs_feas_mask, torch.logical_not(obs_par_mask))

        # Plot infeasible observations
        if mobo.get_Ycon() is not None and torch.any(obs_inf_mask):
            obs_inf_f1 = y_obj[obs_inf_mask, 0].detach().cpu().numpy()
            obs_inf_f2 = y_obj[obs_inf_mask, 1].detach().cpu().numpy()
            axes.scatter(obs_inf_f1, obs_inf_f2, **infeasible_observations_kwargs)

        # Plot feasible non-pareto-front observations
        if torch.any(obs_feas_and_not_par_mask):
            obs_feas_non_par_f1 = y_obj[obs_feas_and_not_par_mask, 0].detach().cpu().numpy()
            obs_feas_non_par_f2 = y_obj[obs_feas_and_not_par_mask, 1].detach().cpu().numpy()
            axes.scatter(obs_feas_non_par_f1, obs_feas_non_par_f2, **feasible_non_pareto_observations_kwargs)

        # Plot feasible pareto-front observations
        if torch.any(obs_feas_and_par_mask):
            obs_feas_par_f1 = y_obj[obs_feas_and_par_mask][:, 0].detach().cpu().numpy() #[torch.sort(y_obj[feas_and_par_mask][:, 0]).indices]
            obs_feas_par_f2 = y_obj[obs_feas_and_par_mask][:, 1].detach().cpu().numpy()
            axes.scatter(obs_feas_par_f1, obs_feas_par_f2, **feasible_pareto_observations_kwargs)

    # Add legend
    plt.legend()

    filepath = compose_figure_filename(iteration_number=mobo.get_iteration_number())
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if display_figures:
        plt.show()
    plt.close(fig)


# TODO: implement a parallel coordinate plot
def plot_multi_objective_from_RN_to_R3():
    raise NotImplementedError("Parallel coordinate plots are not yet implemented.")


def plot_log_hypervolume_improvement(mobo, show=False):
    """ Note: for numerical stability, the minimum value displayed is log10(epsilon) where epsilon=1e-6."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("Number of observations (beyond initial points)")
    ax.set_ylabel("Log Relative Hypervolume Improvement")
    ax.set_title("Log Relative Hypervolume Improvement from Initial Front")

    # Get hypervolume values
    hv = np.array(mobo.get_hypervolume())
    x = np.arange(len(hv)) * mobo.get_batch_size()

    # Use first value as the reference point
    epsilon = 1e-6
    hv_0 = hv[0]
    hv_diff = (hv - hv_0) / (hv_0 + epsilon)

    # Mask values <= 0 
    mask = hv_diff > 0
    x_masked = x[mask]
    hv_diff_masked = hv_diff[mask]

    # Compute log improvement
    log_hv_diff = np.log10(hv_diff_masked + epsilon)

    # Plot
    ax.scatter(x_masked, log_hv_diff, **optimization_figures_kwargs)

    # Save or show
    plt.tight_layout()
    filepath = compose_figure_filename(postfix="hv_improvement")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)


def plot_elapsed_time(mobo: Mobo, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("Number of observations (beyond initial points)")
    ax.set_ylabel("Elapsed Time (s)")
    x = np.array(range(len(mobo.get_hypervolume()))) * mobo.get_batch_size()
    y = mobo.get_elapsed_time()
    ax.scatter(x, y, **optimization_figures_kwargs)

    plt.tight_layout()
    filepath = compose_figure_filename(postfix="elapsed_time")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)


def plot_allocated_memory(mobo: Mobo, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("Number of observations (beyond initial points)")
    ax.set_ylabel("Allocated memory (MB)")
    x = np.array(range(len(mobo.get_hypervolume()))) * mobo.get_batch_size()
    y = mobo.get_allocated_memory()
    ax.scatter(x, y, **optimization_figures_kwargs)

    plt.tight_layout()
    filepath = compose_figure_filename(postfix="allocated_memory")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)
