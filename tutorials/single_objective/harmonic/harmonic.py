import os
from datetime import datetime
import torch

from builders.kernel import *
from builders.acqf import qLogNEIBuilder
from builders.sampler import *
from objectives.single_objective.harmonic import Harmonic

from plotters.acquisition_function import AcquisitionPlotter
from plotters.single_objective import SingleObjectivePlotter
from samplers.samplers import Sampler
from utils.bo_types import SamplerType, AcquisitionFunctionType
from plotters.evolution import *
from gpytorch.constraints import Interval

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Define the true_objective """
    objective = Harmonic(device=DEVICE, dtype=DTYPE, )

    """ Instantiate a Kernel builder """
    kernel_builder = RBFKernelBuilder()
    kernel_builder.base_params.ard_num_dims = objective.num_objectives
    kernel_builder.base_params.lengthscale_constraint = Interval(1 / 6 * 0.8, 1 / 6 * 1.2)

    # kernel_builder = CosineKernelBuilder()
    # kernel_builder.base_params.ard_num_dims = objective.num_objectives
    # kernel_builder.base_params.period_length_constraint = Interval(1 / 12 * 0.8, 1 / 12 * 1.2)

    # kernel_builder = PeriodicKernelBuilder()
    # kernel_builder.base_params.ard_num_dims = objective.num_objectives
    # kernel_builder.base_params.period_length_constraint = Interval(1 / 6 * 0.8, 1 / 6 * 1.2)

    """ Instantiate an acqf builder """
    acqf_builder = qLogNEIBuilder()

    acqf = AcqfType.qEHVI
    acqf_cfg = {}

    kernel = KernelType.RBF
    kernel_cfg = {}

    sampler = SobolSampler
    sampler_cfg = {}

    """ Instantiate a sampler builder """
    builder = SamplerBuilder(SobolSampler)
    sampler_config = SamplerConfig(
        n_dimensions=objective.dim,
        device=DEVICE,
        dtype=DTYPE,
        bounds=objective.bounds,
        seed=45
    )
    sampler = builder.build(sampler_config)

    sampler_builder = SobolSamplerBuilder()
    sampler_builder = LatinHypercubeSamplerBuilder()
    sampler_builder.runtime_params.dimension = objective.dim
    sampler_builder.runtime_params.scramble = True
    sampler_builder.runtime_params.random_seed = 45

    """ Generate initial dataset """
    # Create a random sampler and draw an initial set of points within the objective bounds.
    # Compute the true objective values at the sampled points.

    # sampler = Sampler(
    #     device=DEVICE,
    #     dtype=DTYPE,
    #     sampler_type=SamplerType.Sobol,
    #     bounds=objective.bounds,
    #     n_dimensions=objective.num_objectives,
    #     normalize=False,
    #     nonlinear_inequality_constraints=objective.nonlinear_inequality_input_constraints,
    #     seed=45,
    # )

    sampler = sampler_builder.build()
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Initialize optimizer """
    bayesian_optimizer = BayesianOptimizer(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acquisition_function_builder=acqf_builder,
        kernel_builder=kernel_builder,
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

    batch_sizes = [1]
    for batch_size in batch_sizes:
        main(n_samples=32, q=batch_size, output_dir=main_path)
