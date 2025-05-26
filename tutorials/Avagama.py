""""
This tutorial shows how to use the MOBO class to optimize a multi-objective problem.
The problem is the Avagama function, which is a multi-objective optimization problem.
The tutorial assumes that the dataset is already available in the data folder, and that
the data is organized in columns as [I_M (A), I_P (A), tau_R (us), t_M (min), t_O (min), W (um)]
"""

import os
from abc import ABC
from botorch.acquisition.multi_objective import IdentityMCMultiOutputObjective, MCMultiOutputObjective
from mobo.output_constraints import LowerBound
from mobo.mobo import Mobo
from mobo.samplers import Sampler
from utils.io import *
from utils.make_video import create_video_from_images
from utils.types import AcquisitionFunctionType, SamplerType, OptimizationProblemType
from utils.plotters import plot_multi_objective_from_RN_to_R2, plot_log_hypervolume_improvement, plot_elapsed_time, \
    plot_allocated_memory


class CustomMultiObjective(MCMultiOutputObjective, ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def f1(Z: torch.Tensor) -> torch.Tensor:
        i_m = Z[..., 0]
        i_p = Z[..., 1]
        tau_r = Z[..., 2]

        t_on = torch.full_like(i_m, 78.0)

        energy = i_p * tau_r + 0.5 * tau_r * (i_m - i_p) + i_m * (t_on - tau_r)
        energy = energy / torch.max(torch.abs(energy))
        noise = torch.rand_like(energy) * torch.max(energy) / 10
        # theta = torch.arccos(tau_r / torch.sqrt(tau_r ** 2 + (i_m - i_p) ** 2))
        wear = energy + noise
        return 250 * wear.unsqueeze(-1)

    @staticmethod
    def f2(Z: torch.Tensor) -> torch.Tensor:
        i_m = Z[..., 0]
        i_p = Z[..., 1]
        tau_r = Z[..., 2]

        t_on = torch.full_like(i_m, 78.0)
        energy = i_p * tau_r + 0.5 * tau_r * (i_m - i_p) + i_m * (t_on - tau_r)
        time = 1 / energy
        time = time / torch.max(torch.abs(time))
        noise = torch.rand_like(time) * torch.max(time) / 10
        time = time + noise
        return 300 * time.unsqueeze(-1)

    def forward(self, samples: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
        x_eval = X if X is not None else samples
        obj1 = self.f1(x_eval)
        obj2 = self.f2(x_eval)
        return torch.cat([obj1, obj2], dim=-1)


def main(n_samples=1, batch_size=1):
    experiment_name = f"test_avagama_64samples_1q_1024mc_512rs_qnehvi"
    main_directory = f"../data"
    directory = create_experiment_directory(main_directory, experiment_name)
    os.chdir(directory)

    """ Instantiate a random generator """
    sampler = Sampler(
        sampler_type=SamplerType.Sobol,
        bounds=torch.Tensor([[7.5, 3, 0.1], [15, 7.5, 1]]),
        n_dimensions=3,
        normalize=False
    )

    """ Instantiate a Mobo object """
    mobo = Mobo(
        experiment_name=experiment_name,
        X=None,
        Yobj=None,
        Yobj_var=None,
        Ycon=None,
        Ycon_var=None,
        bounds=torch.Tensor([[7.5, 3, 0.1], [15, 7.5, 1]]),
        optimization_problem_type=OptimizationProblemType.Minimization,
        true_objective=None,
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1]),
        output_constraints=[LowerBound(40, index=2)],
        acquisition_function_type=AcquisitionFunctionType.qNEHVI,
        sampler_type=SamplerType.Sobol,
        raw_samples=512,
        mc_samples=1024,
        batch_size=1,
    )

    """ Load dataset and generate random samples for posterior and ground truth evaluation """
    mobo.load_dataset_from_csv(
        filepath=Path.cwd().parent / "avagama_dataset.csv",
        input_space_dim=3,
        objective_space_dim=2,
        constraint_space_dim=1,
        objective_variance=False,
        constraint_variance=False,
        skiprows=1,
        skipcols=0,
    )
    rnd_X = sampler.draw_samples(n=1000)


    for i in range(int(n_samples / batch_size)):
        print("\n\n")
        print(f"*** Iteration {i + 1}/{int(n_samples / batch_size)} ***")

        mobo.optimize()
        mobo.to_file()

        plot_multi_objective_from_RN_to_R2(
            mobo=mobo,
            show_ref_point=True,
            show_ground_truth=False,
            show_posterior=True,
            x=rnd_X,
            show_observations=True,
            f1_label=r'Wear ($\mu$m)',
            f2_label=r'Machining Time (min)',
            # f1_lims=(-1.6, 0.1),
            # f2_lims=(-1.6, 0.1),
            display_figures=False
        )

        new_X = mobo.get_new_X()

        """ Simulate experiment at new X """
        # new_Yobj = true_objective(new_X)
        # new_Ycon = -true_objective.evaluate_slack(new_X)
        # print(f"New Yobj: {new_Yobj}")
        # print(f"New Ycon: {new_Ycon}")

        """ Save to csv """
        # mobo.update_XY(new_X=new_X, new_Yobj=new_Yobj, new_Ycon=new_Ycon)
        # mobo.save_dataset_to_csv()
        # print(f"GPU Memory Allocated: {mobo.get_allocated_memory()[-1]:.2f} MB")

    plot_log_hypervolume_improvement(mobo, show=False)
    plot_elapsed_time(mobo, show=False)
    plot_allocated_memory(mobo, show=False)
    create_video_from_images()
    print("Optimization Finished.")


if __name__ == "__main__":
    main_dir = Path.cwd().parent
    main()
