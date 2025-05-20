import os
from botorch.acquisition.multi_objective import MCMultiOutputObjective, IdentityMCMultiOutputObjective
from abc import ABC
from mobo.mobo import Mobo
from mobo.samplers import draw_samples
from utils.io import *
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_difference, plot_elapsed_time, \
    plot_allocated_memory


#TODO: check class
class Kursawe(MCMultiOutputObjective, ABC):
    def __init__(self):
        super().__init__()

    def f1(self, x: torch.Tensor):
        n = x.shape[-1]
        f1 = 0
        for i in range(n - 1):
            f1 += -10 * torch.exp(-0.2 * torch.sqrt(torch.square(x[..., i]) + torch.square(x[..., i + 1])))
        return f1.unsqueeze(-1)

    def f2(self, x: torch.Tensor):
        n = x.shape[-1]
        f2 = 0
        for i in range(n):
            f2 += torch.pow(torch.abs(x[..., i]), 0.8) + 5 * torch.sin(torch.pow(x[..., i], 3))
        return f2.unsqueeze(-1)

    def forward(self, samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        x_eval = X if X is not None else samples
        obj1 = self.f1(x_eval)
        obj2 = self.f2(x_eval)
        return torch.cat([obj1, obj2], dim=-1)


def main(n_samples=64, batch_size: int = 1, ):
    main_directory = Path(f"../data")
    experiment_name = f"test_kursawe_64iter_{batch_size}q_512mc_256rs_qlognehvi"
    directory = create_experiment_directory(main_directory, experiment_name)
    os.chdir(directory)

    """ Generate initial dataset """
    X = draw_samples(
        sampler_type=SamplerType.Sobol,
        bounds=torch.tensor([[-5, -5], [5, 5]]),
        n_samples=2 * (2 + 1),
        n_dimensions=2,
        normalize=False
    )
    Yobj = Kursawe()(X)

    """ Main optimization loop """
    mobo = Mobo(
        experiment_name=experiment_name,
        X=X,
        Yobj=Yobj,
        Yobj_var=None,
        Ycon=None,
        Ycon_var=None,
        bounds=torch.tensor([[-5, -5], [5, 5]]),
        optimization_problem_type=OptimizationProblemType.Minimization,
        true_objective=Kursawe(),
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        constraints=None,
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
            f1_lims=(-20, -14),
            f2_lims=(-12, 2),
            display_figures=False
        )

        """ Simulate experiment at new X """
        new_X = mobo.get_new_X()
        new_Yobj = Kursawe()(new_X)
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
