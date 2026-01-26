import torch


def compute_feasible_mask(objective, X, Y_obj, Y_con):
    feasible_input_mask = compute_feasible_input_mask(objective, X)
    feasible_output_mask = compute_feasible_output_mask(objective, Y_obj, Y_con)
    feasible_mask = feasible_input_mask & feasible_output_mask
    return feasible_mask


def compute_feasible_input_mask(objective, X):
    return objective.is_input_feasible(X)


def compute_feasible_output_mask(objective, Y_obj, Y_con):
    if objective.constraints is None:
        Y_feasible = torch.ones(Y_obj, dtype=torch.bool, device=self._device)
    else:
        Y_full = torch.cat([Y_obj, Y_con], dim=-1)
        Y_feasible = objective.is_output_feasible(Y_full)
    return Y_feasible
