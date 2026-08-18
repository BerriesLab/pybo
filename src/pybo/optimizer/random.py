from pybo.optimizer.sobol import SobolOptimizer
from pybo.samplers.random import RandomSampler


class RandomOptimizer(SobolOptimizer):
    """An optimizer that proposes by drawing independent uniform points.

    The weakest of the three arms: Sobol spreads its draws by construction, so a Sobol
    run already beats chance at covering the space. What this one measures is how much
    of an arm's advantage comes from that coverage and how much from the model.
    """

    optimizer_type = "random"
    sampler_class = RandomSampler
