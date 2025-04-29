import numpy as np
import torch
from bayesian_optimization_for_gpu.latin_hypercube_sampling import LatinHypercubeSampling
from bayesian_optimization_for_gpu.mobo_gpu import Mobo
from utils.cuda import *
from utils.types import AcquisitionFunctionType, SamplerType
from botorch.test_functions.multi_objective import BraninCurrin

n_samples = 50
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
Yvar = torch.randn_like(Y) ** 2
bounds = torch.tensor(bounds)

""" Main optimization loop """
for i in range(n_opt_iter):

    if i == 0:
        # Instantiate a new MOBO
        mobo = Mobo(experiment_name="test_mobo_from_R3_to_R2")
        mobo.set_X(X)
        mobo.set_Y(Y)
        mobo.set_Yvar(Yvar)
        mobo.set_bounds(bounds)
        mobo.set_acquisition_function(AcquisitionFunctionType.qLogEHVI)
        mobo.set_sampler(SamplerType.Sobol)
        mobo.set_batch_size(4)

    else:
        # Load an existing MOBO
        mobo = Mobo.from_file("../data/mobo_from_R3_to_R2.pkl")

    mobo.optimize()

# Evaluate the objective function at the new point(s).
new_Y = problem(mobo.get_new_X())
# new_Yvar = torch.zeros_like(new_obj)  # Assume no noise

# Update the models with the new data.
X = torch.cat([X, mobo.get_new_X()], dim=0)
Y = torch.cat([Y, new_Y], dim=0)
# Yvar = torch.cat([Yvar, new_Yvar], dim=0)
#models, mlls = define_model(train_x, train_obj, train_yvar, bounds)
#
# Print some info
# non_dominated = is_non_dominated(train_obj)
# print(f"  Number of non-dominated points: {non_dominated.sum().item()}")
# print(f"  Current non-dominated solutions (x, obj):")
# for nd_idx in torch.where(non_dominated)[0]:
#     print(f"    x: {train_x[nd_idx].tolist()}, obj: {train_obj[nd_idx].tolist()}")
#
# inner_loop_end_time = time.time()  # End timing
# inner_loop_time = inner_loop_end_time - inner_loop_start_time
# print(f"  Iteration time: {inner_loop_time:.2f} seconds")

# Return the final training data and models.
