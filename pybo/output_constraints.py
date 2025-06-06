import torch


class UpperBound:
    def __init__(self, threshold: float, index: int = -1):
        self.threshold = threshold
        self.index = index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # c(x) = samples[..., index] <= threshold
        return samples[..., self.index] - self.threshold


class LowerBound:
    def __init__(self, threshold: float, index: int = -1):
        self.threshold = threshold
        self.index = index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # c(x) = threshold <= samples[..., index]
        return self.threshold - samples[..., self.index]
