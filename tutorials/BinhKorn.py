import os
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective
from abc import ABC
from botorch.test_functions.base import ConstrainedBaseTestProblem
from mobo.mobo import Mobo
from mobo.samplers import Sampler
from utils.io import *
from utils.make_video import create_video_from_images
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_improvement, plot_elapsed_time, \
    plot_allocated_memory


class BinhAndKorn(ConstrainedBaseTestProblem, ABC):
    r"""Binh and Korn test problem for multi-objective optimization with constraints."""

    dim = 2
    num_objectives = 2
    num_constraints = 2
    _bounds = [(0.0, 5.0), (0.0, 3.0)]  # [(lower_0, upper_0), (lower_1, upper_1)]

    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(x: torch.Tensor) -> torch.Tensor:
        return (4 * x[..., 0]**2 + 4 * x[..., 1]**2).unsqueeze(-1)

    @staticmethod
    def f2(x: torch.Tensor) -> torch.Tensor:
        return ((x[..., 0] - 5)**2 + (x[..., 1] - 5)**2).unsqueeze(-1)

    def evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        r"""Return the true (noise-free) objective values at X."""
        return torch.cat([self.f1(X), self.f2(X)], dim=-1)

    def evaluate_slack_true(self, X: torch.Tensor) -> torch.Tensor:
        r""" Evaluate the true slack for each constraint at X.

        Constraints:
        1. g1(x) = (x0 - 5)^2 + x1^2 <= 25 → slack = 25 - ((x0 - 5)^2 + x1^2)
        2. g2(x) = (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 → slack = ((x0 - 8)^2 + (x1 + 3)^2) - 7.7
        """
        g1_slack = 25 - ((X[..., 0] - 5) ** 2 + X[..., 1] ** 2)
        g2_slack = (X[..., 0] - 8) ** 2 + (X[..., 1] + 3) ** 2 - 7.7
        return torch.stack([g1_slack, g2_slack], dim=-1)

def c1(x):
    return torch.square((x[..., -2] - 5)) + torch.square(x[..., -1]) - 25

def c2(x):
    return 7.7 - torch.square((x[..., -2] - 8)) - torch.square((x[..., -1] + 3))

def main(n_samples=64, q: int = 1, ):
    data_dir = main_dir / "data"
    experiment_name = f"test_binh_and_korn_64iter_{q}q_512mc_256rs_qnehvi"
    directory = create_experiment_directory(data_dir, experiment_name)
    os.chdir(directory)

    """ Define the true_objective """
    true_objective = BinhAndKorn()

    """ Instantiate a random generator """
    sampler = Sampler(
        sampler_type=SamplerType.Sobol,
        bounds=true_objective.bounds,
        n_dimensions=true_objective.dim,
        normalize=False
    )

    """ Generate initial dataset and random samples for posterior and ground truth evaluation """
    X = sampler.draw_samples(n=2*(true_objective.dim+1))
    rnd_X = sampler.draw_samples(n=1000)

    """ Main optimization loop """
    mobo = Mobo(
        experiment_name=experiment_name,
        X=X,
        Yobj=true_objective(X),
        Yobj_var=None,
        Ycon=None,
        Ycon_var=None,
        bounds=true_objective.bounds,
        optimization_problem_type=OptimizationProblemType.Minimization,
        true_objective=true_objective,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        constraints=[c1, c2],
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        sampler_type=SamplerType.Sobol,
        raw_samples=256,
        mc_samples=512,
        batch_size=q,
    )

    for i in range(int(n_samples / q)):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / q)} ***")

        mobo.optimize()
        mobo.to_file()
        plot_multi_objective_from_RN_to_R2(
            mobo=mobo,
            show_ref_point=True,
            show_ground_truth=True,
            show_observations=True,
            f1_lims=(0, 140),
            f2_lims=(0, 50),
            display_figures=False,
            x=rnd_X,
        )

        """ Simulate experiment at new X """
        new_X = mobo.get_new_X()
        new_Yobj = true_objective(new_X)
        print(f"New Yobj: {new_Yobj}")

        """ Save to csv """
        mobo.update_XY(new_X=new_X, new_Yobj=new_Yobj)
        mobo.save_dataset_to_csv()
        print(f"GPU Memory Allocated: {mobo.get_allocated_memory()[-1]:.2f} MB")

    plot_log_hypervolume_improvement(mobo, show=False)
    plot_elapsed_time(mobo, show=False)
    plot_allocated_memory(mobo, show=False)
    create_video_from_images()
    print("Optimization Finished.")


if __name__ == "__main__":
    main_dir = Path.cwd().parent
    batch_sizes = [1, 2, 4, 8]
    for batch_size in batch_sizes:
        main(n_samples=64, q=batch_size)
