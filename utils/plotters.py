import matplotlib.pyplot as plt
from botorch.utils.multi_objective import is_non_dominated
from mobo.mobo import Mobo
from utils.io import *
from utils.types import OptimizationProblemType

ms = 7

accepted_observations_kwargs = {
    'color': "tab:green",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Non-Pareto Obs.'
}

rejected_observations_kwargs = {
    'color': "tab:red",
    'marker': "o",
    's': ms ** 2,
    "alpha": 0.7,
    "edgecolors": "black",
    'label': 'Rejected Obs.'
}

observed_pareto_kwargs = {
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

ground_truth_kwargs = {
    'color': "black",
    'marker': "o",
    's': ms ** 2 / 5,
    "alpha": 0.1
}

posterior_pareto_kwargs = {
    'fmt': 'o',
    'ecolor': 'tab:blue',
    'alpha': 0.3,
    'label': r'Post. $\mu \pm 3 \sigma$',
    'capsize': 3,
}


# TODO: implement function
def plot_objective_from_R1_to_R1():
    raise NotImplementedError("Function not yet implemented.")


# TODO: implement function
def plot_objective_from_R2_to_R1():
    raise NotImplementedError("Function not yet implemented.")


def plot_multi_objective_from_RN_to_R2(
        mobo: Mobo,
        X: torch.Tensor,
        f1_lims=None,
        f2_lims=None,
        f1_label="$f_{01}$",
        f2_label="$f_{02}$",
        show_ground_truth=False,
        show_posterior=False,
        show_ref_point=False,
        show_rejected_observations=False,
        show_accepted_non_pareto_observations=True,
        show_accepted_pareto_observations=True,
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

    # Plot posterior pareto front if requested
    if show_posterior:
        # Ensure that x is on the right device
        x = X.to(mobo.get_device(), mobo.get_dtype())
        posterior = mobo.get_model().posterior(x)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        # Apply constraints on outputs (mean)
        constraint_vals = [c(mean) for c in mobo.get_constraints()]
        con_mask = torch.stack([(cv <= 0) for cv in constraint_vals]).all(dim=0)
        mean_feasible = mean[con_mask]
        std_feasible = std[con_mask]

        # Get mask of non-dominated points among feasible ones
        feasible_mask = is_non_dominated(
            Y=mean_feasible,
            maximize=mobo.get_optimization_problem_type().value
        )

        # Calculate pareto front for mean and samples
        mean_pareto = mean_feasible[feasible_mask].detach().cpu().numpy()
        std_pareto = std_feasible[feasible_mask].detach().cpu().numpy()
        if mobo.get_optimization_problem_type() == OptimizationProblemType.Maximization:
            sorted_idx  = np.argsort(-mean_pareto[:, 0])
        else:
            sorted_idx  = np.argsort(mean_pareto[:, 0])
        mean_pareto = mean_pareto[sorted_idx]
        std_pareto = std_pareto[sorted_idx]

        # Plot mean with error bars
        axes.errorbar(
            mean_pareto[:, 0],
            mean_pareto[:, 1],
            xerr=3 * std_pareto[:, 0],
            yerr=3 * std_pareto[:, 1],
            **posterior_pareto_kwargs,
        )

    # TODO: ground truth should also take into account for rejected values and accepted values??
    # Plot ground truth if requested
    if show_ground_truth:
        if not mobo.get_true_objective():
            raise ValueError("Ground truth not available.")
        x = X.to(mobo.get_device(), mobo.get_dtype())
        y = mobo.get_true_objective()(x)
        constraint_vals = [c(y) for c in mobo.get_constraints()]
        mask = torch.stack([(cv <= 0) for cv in constraint_vals]).all(dim=0)
        f1 = y[mask, 0].detach().cpu().numpy()
        f2 = y[mask, 1].detach().cpu().numpy()
        axes.scatter(f1, f2, **ground_truth_kwargs)

    # Plot reference point if requested
    if show_ref_point:
        show_ref_point = mobo.get_ref_point()
        f1 = show_ref_point[0].detach().cpu().numpy()
        f2 = show_ref_point[1].detach().cpu().numpy()
        plt.scatter(f1, f2, **ref_point_kwargs)

    # Plot rejected observations
    if show_rejected_observations:
        if mobo.get_Ycon() is None:
            print("Warning: Rejected observations not available - this is an unconstrained problem.")
        else:
            y_obj = mobo.get_Yobj()
            mask = torch.logical_not(mobo.get_con_mask())
            if torch.any(mask):
                f1 = y_obj[mask, 0].detach().cpu().numpy()
                f2 = y_obj[mask, 1].detach().cpu().numpy()
                axes.scatter(f1, f2, **rejected_observations_kwargs)

    # Plot accepted observations that do not belong to the pareto-front
    if show_accepted_non_pareto_observations:
        y_obj = mobo.get_Yobj()
        par_mask = mobo.get_par_mask()
        con_mask = mobo.get_con_mask()
        mask = torch.logical_and(con_mask, torch.logical_not(par_mask))
        if torch.any(mask):
            f1 = y_obj[mask, 0].detach().cpu().numpy()
            f2 = y_obj[mask, 1].detach().cpu().numpy()
            axes.scatter(f1, f2, **accepted_observations_kwargs)

    # Plot accepted observations that belong to the Pareto Front
    if show_accepted_pareto_observations:
        y_obj = mobo.get_Yobj()
        con_mask = mobo.get_con_mask()
        par_mask = mobo.get_par_mask()
        mask = torch.logical_and(con_mask, par_mask)
        if torch.any(mask):
            f = y_obj[mask][torch.sort(y_obj[mask][:, 0]).indices]
            f1 = f[:, 0].detach().cpu().numpy()
            f2 = f[:, 1].detach().cpu().numpy()
            axes.scatter(f1, f2, **observed_pareto_kwargs)

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


def plot_log_hypervolume_difference(mobo: Mobo, show=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel("Number of observations (beyond initial points)")
    ax.set_ylabel("Log Hypervolume Difference")
    x = np.array(range(len(mobo.get_hypervolume()))) * mobo.get_batch_size()
    hv = np.array(mobo.get_hypervolume())
    # max_hv = mobo.get_true_objective().max_hv
    dy = np.log10(hv[0] - hv)
    ax.errorbar(x, dy, linewidth=2, )

    plt.tight_layout()
    filepath = compose_figure_filename(postfix="hv_diff")
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
    ax.errorbar(x, y, linewidth=2, )

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
    ax.errorbar(x, y, linewidth=2, )

    plt.tight_layout()
    filepath = compose_figure_filename(postfix="allocated_memory")
    fig.savefig(filepath, dpi=300, bbox_inches='tight', format='png')

    if show:
        plt.show()
    plt.close(fig)
