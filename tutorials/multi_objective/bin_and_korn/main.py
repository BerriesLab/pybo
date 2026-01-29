import os
from datetime import datetime
import torch
from pathlib import Path
from botorch.acquisition.multi_objective import qLogNoisyExpectedHypervolumeImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel
from bayesian_optimizer.optimizer import BayesianOptimizer
from plotters.evolution import EvolutionPlotter
from plotters.experiment import ParetoFront2DPlotter
from samplers.samplers import SobolSampler
from plotters.metrics import HypervolumePlotter, ElapsedTimePlotter, HypervolumeImprovementPlotter
from tutorials.multi_objective.bin_and_korn.objective import BinhAndKorn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def main(n_samples=64, q: int = 1, output_dir: Path = None):
    run_dir = output_dir / f"batch_{q}"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    """ Instantiate true objective """
    objective = BinhAndKorn(device=DEVICE, dtype=DTYPE)

    """ Instantiate kernel """
    kernel = ScaleKernel(base_kernel=RBFKernel(ard_num_dims=objective.num_obj))

    """ Generate initial dataset """
    sampler = SobolSampler(device=DEVICE, dtype=DTYPE, objective=objective, seed=2063)
    X = sampler.draw_samples(n=5 * (objective.dim + 1))
    Y_obj = objective.evaluate_true_objective(X)

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
        # Plot experiment
        exp_plt = ParetoFront2DPlotter(
            bo=bo,
            y=("obj", "Korn"),
            x=("obj", "Binh"),
            z=("par", "P2")
        )
        exp_plt.plot().save_figure().close_figure()

        ElapsedTimePlotter(bo=bo).plot().save_figure().close_figure()
        HypervolumePlotter(bo=bo).plot().save_figure().close_figure()
        HypervolumeImprovementPlotter(bo=bo).plot().save_figure().close_figure()

        # Plot parameters evolution
        if bo.objective.par_cfg is not None:
            for par in bo.objective.par_cfg:
                evo_plt = EvolutionPlotter(
                    bo=bo,
                    y=("par", par.label)
                )
                evo_plt.plot().save_figure().close_figure()

        # Plot objectives evolution
        if bo.objective.obj_cfg is not None:
            for obj in bo.objective.obj_cfg:
                evo_plt = EvolutionPlotter(
                    bo=bo,
                    y=("obj", obj.label)
                )
                evo_plt.plot().save_figure().close_figure()

        # Plot constraints evolution
        if bo.objective.ineq_Y_con_cfg is not None:
            for con in bo.objective.ineq_Y_con_cfg:
                evo_plt = EvolutionPlotter(
                    bo=bo,
                    y=("con", con.label)
                )
                evo_plt.plot().save_figure().close_figure()

        # Plot trackers evolution
        if bo.objective.trk_cfg is not None:
            for trk in objective.trk_cfg:
                evo_plt = EvolutionPlotter(
                    bo=bo,
                    y=("trk", trk.label)
                )
                evo_plt.plot().save_figure().close_figure()

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
        main(n_samples=32, q=batch_size, output_dir=main_path)
