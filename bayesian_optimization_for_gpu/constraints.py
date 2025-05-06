import torch


class UpperBound:
    def __init__(self, threshold, constraint_index=-1):
        self.threshold = threshold
        self.objective_index = constraint_index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return samples[..., self.objective_index] - self.threshold


class LowerBound:
    def __init__(self, threshold, constraint_index=-1):
        self.threshold = threshold
        self.objective_index = constraint_index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        return self.threshold - samples[..., self.objective_index]

