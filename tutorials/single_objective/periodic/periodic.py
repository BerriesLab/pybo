import os
from datetime import datetime
import torch
from bayesian_optimizer.acquisition_function import AcquisitionFunctionFactory
from bayesian_optimizer.kernel import KernelFactory, RBFConfig, PeriodicConfig
from objectives.single_objective.periodic import Periodic
from plotters.acquisition_function import AcquisitionPlotter
from plotters.single_objective import SingleObjectivePlotter
from samplers.samplers import Sampler
from utils.bo_types import AcquisitionFunctionType, SamplerType, KernelType
from plotters.evolution import *
from gpytorch.constraints import Interval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Define the true_objective """
    objective = Periodic(
        device=DEVICE,
        dtype=DTYPE,
    )

    # """ Instantiate a kernel constructor"""
    # kernel_factory = KernelFactory(
    #     kernel_type=KernelType.RBF,
    #     ard_num_dims=objective.num_objectives,
    #     config=RBFConfig(
    #         # lengthscale_constraint=Interval(0.01, 0.1),
    #     )
    # )

    kernel_factory = KernelFactory(
        kernel_type=KernelType.PERIODIC,
        ard_num_dims=objective.num_objectives,
        config=PeriodicConfig(
            period_length_constraint=Interval(0.9, 1.1)
        )
    )

    # kernel = KernelFactory(
    #     kernel_type=KernelType.RBF_TIMES_PERIODIC,
    #     ard_num_dims=objective.num_objectives,
    #     config=RBFTimesPeriodicConfig(
    #         rbf=RBFConfig(),
    #         periodic=PeriodicConfig(
    #             period_length_constraint=Interval(0.8, 1.2)
    #         )
    #     )
    # )

    # kernel_factory = KernelFactory(
    #     kernel_type=KernelType.RBF_PLUS_PERIODIC,
    #     ard_num_dims=objective.num_objectives,
    #     config=RBFTimesPeriodicConfig(
    #         rbf=RBFConfig(),
    #         periodic=PeriodicConfig(
    #             period_length_constraint=Interval(0.8, 1.2)
    #         )
    #     )
    # )

    """ Instantiate an acquisition function constructor"""
    acquisition_function_factory = AcquisitionFunctionFactory(
        acqf_type=AcquisitionFunctionType.qLogNEI,
    )

    """ Generate initial dataset """
    # Create a random sampler and draw an initial set of points within the objective bounds.
    # Compute the true objective values at the sampled points.
    sampler = Sampler(
        device=DEVICE,
        dtype=DTYPE,
        sampler_type=SamplerType.Sobol,
        bounds=objective.bounds,
        n_dimensions=objective.num_objectives,
        normalize=False,
        nonlinear_inequality_constraints=objective.nonlinear_inequality_input_constraints,
        seed=45,
    )
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Initialize optimizer """
    bayesian_optimizer = BayesianOptimizer(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_factory=acquisition_function_factory,
        kernel_factory=kernel_factory,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_samples / q)):
        if i > 0 and bayesian_optimizer.is_converged(patience=32):
            break

        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        """ Optimize and get new X """
        bayesian_optimizer.optimize()

        """ Plot """
        x_lims, y_lims = (-3, 3), (-2, 2)
        lims = [x_lims, y_lims]
        SingleObjectivePlotter(bayesian_optimizer=bayesian_optimizer, lims=lims).plot().save_figure().close_figure()
        AcquisitionPlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        ElapsedTimePlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        BestValuePlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        ParameterPlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()
        ObjectivePlotter(bayesian_optimizer=bayesian_optimizer).plot().save_figure().close_figure()

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

    batch_sizes = [1, 2]
    for batch_size in batch_sizes:
        main(n_samples=32, q=batch_size, output_dir=main_path)
