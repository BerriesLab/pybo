import matplotlib.pyplot as plt
from botorch.utils.multi_objective import is_non_dominated
from botorch.utils.transforms import unnormalize, normalize
from mobo.mobo import Mobo
from mobo.samplers import draw_samples
from utils.io import *
from utils.types import OptimizationProblemType, SamplerType

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
    'color': 'tab:blue',
    'linestyle': '-',
    'linewidth': 1.5,
    'marker': 'o',
    'markersize': 0,
    'markerfacecolor': 'tab:blue',
    'markeredgecolor': 'black',
    'alpha': 0.6,
    'label': 'Post. Mean'
}


# TODO: implement function
def plot_objective_from_R1_to_R1():
    raise NotImplementedError("Function not yet implemented.")


# TODO: implement function
def plot_objective_from_R2_to_R1():
    raise NotImplementedError("Function not yet implemented.")


def plot_multi_objective_from_RN_to_R2(
        mobo: Mobo,
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
    """ X is an 'n x d' feature matrix, Y is an 'n x 2' objective matrix, where n is the number of samples,
    and d is the number of dimensions. Pareto is a boolean array indicating which samples are Pareto optimal."""
    # Initialize figure
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.set_xlabel(f1_label)
    axes.set_ylabel(f2_label)
    axes.set_title(r'Multi objective Bayesian Optimization for $\mathbf{f_0}:\mathbb{R}^N \rightarrow \mathbb{R}^2$')
    if f1_lims is not None and isinstance(f1_lims, tuple):
        axes.set_xlim(f1_lims[0], f1_lims[1])
    if f2_lims is not None and isinstance(f2_lims, tuple):
        axes.set_ylim(f2_lims[0], f2_lims[1])

    # Plot posterior if requested
    if show_posterior:
        # Extract N random samples
        dims = mobo.get_X().shape[-1]
        x = draw_samples(
            sampler_type=SamplerType.Sobol,
            bounds=mobo.get_bounds().cpu(),
            n_samples=int(1e2),
            n_dimensions=dims
        ).to(mobo.get_device(), mobo.get_dtype())
        x = normalize(x, mobo.get_bounds())
        # x = unnormalize(x, mobo.get_bounds().cpu()).to(mobo.get_X().device)
        # Calculate posterior mean and std. dev
        posterior = mobo.get_model().posterior(x)
        mean = posterior.mean
        std = posterior.variance.sqrt()

        # Calculate pareto front for mean and samples
        mean_mask = is_non_dominated(Y=mean, maximize=mobo.get_optimization_problem_type().value)
        mean_pareto = mean[mean_mask].detach().cpu().numpy()
        std_pareto = std[mean_mask].detach().cpu().numpy()
        if mobo.get_optimization_problem_type() == OptimizationProblemType.Maximization:
            mean_sorted = np.argsort(-mean_pareto[:, 0])
        else:
            mean_sorted = np.argsort(mean_pareto[:, 0])

        mean_pareto = mean_pareto[mean_sorted]
        std_pareto = std_pareto[mean_sorted]

        # Plot std dev bands
        for i in range(3, 0, -1):
            axes.fill_between(mean_pareto[:, 0],
                              mean_pareto[:, 1] - i * std_pareto[:, 1],
                              mean_pareto[:, 1] + i * std_pareto[:, 1],
                              alpha=0.15,
                              color='tab:blue',
                              label=f'{i}σ' if i == 1 else None)

        axes.plot(mean_pareto[:, 0], mean_pareto[:, 1], **posterior_pareto_kwargs)

    # Plot ground truth if requested
    if show_ground_truth:
        if not mobo.get_true_objective():
            raise ValueError("Ground truth not available.")

        # Draw a number of random samples to plot the ground truth
        dims = mobo.get_X().shape[-1]
        x = draw_samples(
            sampler_type=SamplerType.Sobol,
            bounds=mobo.get_bounds().cpu(),
            n_samples=int(1e3),
            n_dimensions=dims
        )
        x = unnormalize(x, mobo.get_bounds().cpu())
        y = mobo.get_true_objective()(x)
        f1 = y[:, 0].detach().cpu().numpy()
        f2 = y[:, 1].detach().cpu().numpy()
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
    #plt.close(fig)


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
