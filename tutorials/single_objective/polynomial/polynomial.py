import os
from datetime import datetime

import torch
from plotters.single_objective import SingleObjectivePlotter
from samplers.samplers import Sampler
from utils.types import AcquisitionFunctionType, SamplerType, KernelType
from plotters.evolution import *
from objectives.single_objective.polynomial import Polynomial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Define the true_objective """
    objective = Polynomial(
        device=DEVICE,
        dtype=DTYPE,
    )

    """ Instantiate a random generator """
    sampler = Sampler(
        device=DEVICE,
        dtype=DTYPE,
        sampler_type=SamplerType.Sobol,
        bounds=objective.bounds,
        n_dimensions=objective.num_objectives,
        normalize=False,
        nonlinear_inequality_constraints=objective.nonlinear_inequality_input_constraints,
    )

    """ Generate initial dataset """
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Initialize optimizer """
    bayesian_optimizer = BayesianOptimizer(
        experiment_name="quadratic",
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_factory=AcquisitionFunctionType.qLogEI,
        kernel_factory=KernelType.RBF,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        """ Optimize and get new X """
        bayesian_optimizer.optimize()

        """ Plot """
        (SingleObjectivePlotter(bayesian_optimizer=bayesian_optimizer).plot_objective().plot_ground_truth().
         plot_mean().plot_confidence().plot_optimum().plot_next_X().save_figure().close_figure())
        ElapsedTimePlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        BestValuePlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        ParameterPlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        ObjectivePlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        # ConstraintPlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        # TrackerPlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()

        """ Evaluate posterior and acquisition function at new X """
        new_X = bayesian_optimizer.new_X
        bayesian_optimizer.compute_acquisition_function_value_at_X(new_X)
        bayesian_optimizer.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bayesian_optimizer.update_XY(new_X=new_X, new_Y_obj=new_Y_obj)

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_path = Path.cwd() / "data" / date_time
    main_path.mkdir(parents=True, exist_ok=True)

    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=32, q=batch_size, output_dir=main_path)
