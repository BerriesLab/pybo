import numpy as np
import torch

from tqdm import tqdm
from bayesian_optimization_for_gpu.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from bayesian_optimization_for_gpu.utils import *
from enums import AcquisitionFunction
from enums import Kernel

from botorch.test_functions.multi_objective import BraninCurrin
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement

n_samples = 20
problem = BraninCurrin(negate=True)
bounds = problem._bounds
n_opt_iter = 20

""" Sample domain """
lhs = LatinHypercubeSampling()
lhs.set_n_samples(n_samples)
lhs.set_bounds(bounds)
samples = lhs.sample_domain()
np.savetxt("../data/mobo_dataset.csv", samples, delimiter=",", comments="")

""" Simulate experiments """
X = torch.tensor(samples)
Y = problem(X)

""" Main optimization loop """
for i in tqdm(range(n_opt_iter)):

    if i == 0:
        # Instantiate a new MOBO
        mobo = Mobo(
            experiment_name="test_mobo_from_R3_to_R2",
            bounds=bounds,
            acquisition_function=AcquisitionFunction.qEHVI,
            f0=problem,
            kernel=Kernel.RBF,
            X=X,
            Y=Y,
            Yvar=None,
            n_acqf_iter=100,
        )
    else:
        # Load an existing MOBO
        mobo = Mobo.from_pickle("../data/mobo_from_R3_to_R2.pkl")

    mobo.optimize()

# Evaluate the objective function at the new point(s).
new_obj = evaluate_objective(candidates)  # Changed to evaluate_objective
new_yvar = torch.zeros_like(new_obj)  # Assume no noise

# Update the models with the new data.
train_x = torch.cat([train_x, candidates], dim=0)
train_obj = torch.cat([train_obj, new_obj], dim=0)
train_yvar = torch.cat([train_yvar, new_yvar], dim=0)
models, mlls = define_model(train_x, train_obj, train_yvar, bounds)

# Print some info
non_dominated = is_non_dominated(train_obj)
print(f"  Number of non-dominated points: {non_dominated.sum().item()}")
print(f"  Current non-dominated solutions (x, obj):")
for nd_idx in torch.where(non_dominated)[0]:
    print(f"    x: {train_x[nd_idx].tolist()}, obj: {train_obj[nd_idx].tolist()}")

inner_loop_end_time = time.time()  # End timing
inner_loop_time = inner_loop_end_time - inner_loop_start_time
print(f"  Iteration time: {inner_loop_time:.2f} seconds")

# Return the final training data and models.
