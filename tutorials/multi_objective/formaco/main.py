import os
from datetime import datetime
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, RBFKernel
from tutorials.multi_objective.formaco.objective import FormACO
from plotters.experiment import *
from plotters.acqf import *
from plotters.metrics import *
from plotters.evolution import *

DEVICE = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Instantiate true objective """
    objective = FormACO(device=DEVICE, dtype=DTYPE)

    """ Instantiate kernel """
    kernel = ScaleKernel(
        base_kernel=RBFKernel(
            ard_num_dims=objective.num_par,
            lengthscale_constraint=Interval(1e-3, 1e0),
        ),
        outputscale_constraint=Interval(1e-3, 1e1),
    )

    """ Generate initial dataset """
    sampler = SobolSampler(device=DEVICE, dtype=DTYPE, objective=objective)
    X = sampler.draw_samples(n=2 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X=X)
    Y_track = objective.evaluate_tracker(X=X)

    """ Instantiate Bayesian optimizer """
    bo = BayesianOptimizer(
        device=DEVICE,
        dtype=DTYPE,
        objective=objective,
        acqf=qLogNoisyExpectedHypervolumeImprovement,
        kernel=kernel,
        X=X,
        Y_obj=Y_obj,
        Y_obj_var=None,
        Y_con=None,
        Y_con_var=None,
        Y_trk=Y_track,
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
        bo.to_csv()
        bo.to_file()

        """ Plot """
        ParetoFront2DPlotter(
            bo=bo,
            x=("obj", 0),
            y=("obj", 1),
            z=("trk", 0),
            seed=254,
        ).plot().save_figure().close_figure()
        plot_and_save_metrics(bo=bo)
        plot_and_save_evolutions(bo=bo)

        """ Evaluate posterior and acquisition function at new X """
        new_X = bo.new_X
        bo.compute_acquisition_function_value_at_X(new_X)
        bo.compute_posterior_mean_at_X(new_X)

        """ Simulate experiment at new X """
        new_Y_obj = objective.evaluate_true_objective(new_X)
        new_Y_trk = objective.evaluate_tracker(new_X)
        print(f"New Y_obj: {new_Y_obj.detach().cpu().numpy()}")
        bo.update_XY(new_X=new_X, new_Y_obj=new_Y_obj, new_Y_trk=new_Y_trk)

    print("Optimization Finished.")


if __name__ == "__main__":
    print(f"Running on {DEVICE}.")
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_path = Path.cwd() / "data" / date_time
    main_path.mkdir(parents=True, exist_ok=True)

    batch_sizes = [1, 2, 4]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size, output_dir=main_path)
