import torch


class UpperBound:
    def __init__(self, threshold: float, index: int = -1):
        self.threshold = threshold
        self.index = index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # Returns 0 if under threshold, and the distance if over
        return torch.relu(samples[..., self.index] - self.threshold)


class LowerBound:
    def __init__(self, threshold: float, index: int = -1):
        self.threshold = threshold
        self.index = index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # Returns 0 if over threshold, and the distance if under
        return torch.relu(self.threshold - samples[..., self.index])


class Identity:
    def __init__(self, index: int = -1):
        self.index = index

    def __call__(self, Z):
        return Z[..., self.index]
