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


class Interval:
    """Feasible inside [low, high]: 0 when the value is in the band, and the
    distance from the nearer edge when it is outside. Same convention as the
    bounds above, which also return 0 once satisfied.

    Equivalent to a LowerBound and an UpperBound applied together, but as a single
    callable - so one measured quantity stays one constraint, with one column in
    the record and one output for the model to fit.
    """

    def __init__(self, low: float, high: float, index: int = -1):
        self.low = low
        self.high = high
        self.index = index

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        # At most one side can be violated, so the larger of the two shortfalls is
        # the distance from the band, and it is negative exactly when inside it.
        Y = samples[..., self.index]
        return torch.relu(torch.maximum(self.low - Y, Y - self.high))
