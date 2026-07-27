import os
from datetime import datetime
from botorch.acquisition import *
from gpytorch.kernels import *
from gpytorch.constraints import Interval, LessThan
from tutorials.single_objective.wave_packet.objective import WavePacket
from pybo.plotters.acqf import Acqf1DPlotter
from pybo.plotters.experiment import Experiment1DPlotter
from pybo.samplers.samplers import *
from pybo.plotters.evolution import *
from pybo.plotters.metrics import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_evals=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Instantiate true objective """
    objective = WavePacket(device=DEVICE, dtype=DTYPE, )

    """ Instantiate kernel """
    rbf = RBFKernel(ard_num_dims=objective.num_obj, lengthscale_constraint=LessThan(1 / 8))
    periodic = PeriodicKernel(period_length_constraint=Interval(1 / 4 * 0.8, 1 / 4 * 1.2))
    kernel = ScaleKernel(base_kernel=rbf * periodic)

    """ Generate initial dataset """
    sampler = SobolSampler(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        seed=45,
    )
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

    """ Instantiate Bayesian optimizer """
    bo = BayesianOptimizer(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acqf=qLogNoisyExpectedImprovement,
        kernel=kernel,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        batch_size=q,
    )

    """ Main optimization loop """
    for i in range(int(n_evals / q)):
        if i > 0 and bo.is_converged(patience=32):
            break

        print()
        print(f"*** Iteration {i + 1}/{int(n_evals / q)} ***")

        """ Optimize and get new X """
        bo.optimize()

        """ Plot """
        Experiment1DPlotter(bo=bo).plot().save_figure().close_figure()
        Acqf1DPlotter(bo=bo, z=("obj", 0)).plot().save_figure().close_figure()
        ElapsedTimePlotter(bo=bo).plot().save_figure().close_figure()
        BestValuePlotter(bo=bo).plot().save_figure().close_figure()
        EvolutionPlotter(bo=bo, y=("obj", 0)).plot().save_figure().close_figure()
        EvolutionPlotter(bo=bo, y=("par", 0)).plot().save_figure().close_figure()

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

    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        main(n_evals=32, q=batch_size, output_dir=main_path)
