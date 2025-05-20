import os

import torch
from botorch.acquisition.multi_objective import MCMultiOutputObjective, IdentityMCMultiOutputObjective
from abc import ABC

from mobo.mobo import Mobo
from mobo.samplers import draw_samples
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_difference, plot_elapsed_time, \
    plot_allocated_memory


class BinhAndKorn(MCMultiOutputObjective, ABC):
    def __init__(self):
        super().__init__()

    def f1(self, x: torch.Tensor):
        f1 = 4 * torch.square(x[..., 0]) + 4 * torch.square(x[..., 1])
        return f1.unsqueeze(-1)

    def f2(self, x: torch.Tensor):
        f2 = torch.square((x[..., 0] - 5)) + torch.square((x[..., 1] - 5))
        return f2.unsqueeze(-1)

    def forward(self, samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        x_eval = X if X is not None else samples
        obj1 = self.f1(x_eval)
        obj2 = self.f2(x_eval)
        return torch.cat([obj1, obj2], dim=-1)

def c1(x):
    return torch.square((x[..., 0] - 5)) + torch.square(x[..., 1]) - 25

def c2(x):
    return 7.7 - torch.square((x[..., 0] - 8)) - torch.square((x[..., 1] + 3))

def main(n_samples=64, batch_size: int = 1, ):
    main_directory = Path(f"../data")
    experiment_name = f"test_binh_and_korn_64iter_{batch_size}q_512mc_256rs_qlognehvi"
    directory = create_experiment_directory(main_directory, experiment_name)
    os.chdir(directory)

    """ Generate initial dataset """
    X = draw_samples(
        sampler_type=SamplerType.Sobol,
        bounds=torch.tensor([[0, 0], [5, 3]]),
        n_samples=2 * (2 + 1),
        n_dimensions=2,
        normalize=False
    )
    Yobj = BinhAndKorn()(X)

    """ Main optimization loop """
    mobo = Mobo(
        experiment_name=experiment_name,
        X=X,
        Yobj=Yobj,
        Yobj_var=None,
        Ycon=None,
        Ycon_var=None,
        bounds=torch.tensor([[0, 0], [5, 3]]),
        optimization_problem_type=OptimizationProblemType.Minimization,
        true_objective=BinhAndKorn(),
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        constraints=[c1, c2],
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        sampler_type=SamplerType.Sobol,
        raw_samples=256,
        mc_samples=512,
        batch_size=batch_size,
    )

    for i in range(int(n_samples / batch_size)):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / batch_size)} ***")

        mobo.optimize()
        mobo.to_file()
        plot_multi_objective_from_RN_to_R2(
            mobo=mobo,
            show_ref_point=True,
            show_ground_truth=True,
            show_posterior=True,
            show_rejected_observations=True,
            show_accepted_pareto_observations=True,
            show_accepted_non_pareto_observations=True,
            f1_lims=(0, 140),
            f2_lims=(0, 50),
            display_figures=False
        )

        """ Simulate experiment at new X """
        new_X = mobo.get_new_X()
        new_Yobj = BinhAndKorn()(new_X)
        print(f"New Yobj: {new_Yobj}")

        """ Save to csv """
        mobo.update_XY(new_X=new_X, new_Yobj=new_Yobj)
        mobo.save_dataset_to_csv()
        print(f"GPU Memory Allocated: {mobo.get_allocated_memory()[-1]:.2f} MB")

    plot_log_hypervolume_difference(mobo, show=False)
    plot_elapsed_time(mobo, show=False)
    plot_allocated_memory(mobo, show=False)
    print("Optimization Finished.")


if __name__ == "__main__":
    batch_sizes = [1]
    for batch_size in batch_sizes:
        main(n_samples=64, batch_size=batch_size)
