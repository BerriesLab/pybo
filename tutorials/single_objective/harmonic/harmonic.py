import os
from datetime import datetime
from botorch.acquisition import *
from gpytorch.kernels import *
from gpytorch.constraints import Interval
from objectives.single_objective.harmonic import Harmonic
from plotters.acqf import Acqf1DPlotter
from plotters.single_objective_experiment import Experiment1DPlotter
from samplers.samplers import *
from plotters.evolution import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Instantiate true objective """
    objective = Harmonic(device=DEVICE, dtype=DTYPE, )

    """ Instantiate kernel """
    kernel = ScaleKernel(
        base_kernel=CosineKernel(
            period_length_prior=None,
            period_length_constraint=Interval(1 / 12 * 0.8, 1 / 12 * 1.2),
        )
    )

    """ Generate initial dataset """
    # Create a random sampler and draw an initial set of points within the objective bounds.
    # Compute the true objective values at the sampled points.
    sampler = SobolSampler(
        device=DEVICE,
        dtype=DTYPE,
        bounds=objective.bounds,
        n_dimensions=objective.num_objectives,
        normalize=False,
        nonlinear_inequality_constraints=objective.nonlinear_inequality_input_constraints,
        seed=45,
    )
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Instantiate Bayesian optimizer """
    bo = BayesianOptimizer(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acqf=qLogExpectedImprovement,
        kernel=kernel,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        if i > 0 and bo.is_converged(patience=32):
            break

        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        """ Optimize and get new X """
        bo.optimize()

        """ Plot """
        x_lims, y_lims = (-3, 3), (-2, 2)
        lims = [x_lims, y_lims]
        Experiment1DPlotter(bayesian_optimizer=bo, lims=lims).plot().save_figure().close_figure()
        Acqf1DPlotter(bayesian_optimizer=bo).plot().save_figure().close_figure()
        ElapsedTimePlotter(bayesian_optimizer=bo).plot().save_figure().close_figure()
        BestValuePlotter(bayesian_optimizer=bo).plot().save_figure().close_figure()
        ParameterPlotter(bayesian_optimizer=bo).plot().save_figure().close_figure()
        ObjectivePlotter(bayesian_optimizer=bo).plot().save_figure().close_figure()

        """ Evaluate posterior and acquisition function at new X """
        new_X = bo.new_X
        bo.compute_acquisition_function_value_at_X(new_X)
        bo.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_path = Path.cwd() / "data" / date_time
    main_path.mkdir(parents=True, exist_ok=True)

    batch_sizes = [1]
    for batch_size in batch_sizes:
        main(n_samples=32, q=batch_size, output_dir=main_path)
