from abc import ABC, abstractmethod
import torch
from botorch.utils.transforms import unnormalize
from torch import Tensor
from pybo.objectives.base_class import MCObjectiveBase
from pybo.utils.helpers import project_linear_equalities


class SamplerBase(ABC):
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            objective: MCObjectiveBase,
            seed: int | None = None
    ):
        self.device = device
        self.dtype = dtype
        self.objective = objective
        self.seed = seed

    @abstractmethod
    def _generate_base_samples(self, n: int) -> Tensor:
        """ Generates n random samples of shape [n, dim] in the range [0, 1].
        It must be implemented by each subclass."""
        pass

    def draw_samples(self, n: int) -> torch.Tensor:
        """ Draws n random samples of shape [n, dim] subject to X constraints."""
        valid_X = []
        n_valid = 0
        num_attempts = 0
        max_attempts = 1000

        while n_valid < n and num_attempts < max_attempts:
            num_attempts += 1

            norm_X = self._generate_base_samples(n=n)
            X = unnormalize(norm_X, self.objective.bounds)

            if self.objective.lin_eq_X_con:
                X = project_linear_equalities(X=X, lin_eq_cons=self.objective.lin_eq_X_con)

            feasible_mask = self.objective.is_X_feasible(X=X)
            feasible_X = X[feasible_mask]
            if feasible_X.shape[0] > 0:
                valid_X.append(feasible_X)
                n_valid += feasible_X.shape[0]

        if not valid_X:
            raise RuntimeError(f"No valid samples found after {max_attempts} attempts.")

        return torch.cat(valid_X, dim=0)[:n]
