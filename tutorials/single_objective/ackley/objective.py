import torch
from objectives.base_class import MCSingleObjectiveBase

from objectives.data_base import Config, VariableRegistry


class Ackley(MCSingleObjectiveBase):
    """ Unconstrained single objective problem. """

    class Obj(VariableRegistry):
        ACKLEY = Config(label="Ackley", index=0, bounds=(15.0, 0.0), dtype=torch.float64, to_minimize=True)

    class Par(VariableRegistry):
        P1 = Config(label="P1", index=0, bounds=(-5.0, 5.0), dtype=torch.float64)
        P2 = Config(label="P2", index=1, bounds=(-5.0, 5.0), dtype=torch.float64)

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(device=device, dtype=dtype)

    def term1(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[:, self.Par.P1.index]
        x2 = X[:, self.Par.P2.index]
        arg = -0.2 * torch.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
        return -20 * torch.exp(arg)

    def term2(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[:, self.Par.P1.index]
        x2 = X[:, self.Par.P2.index]
        arg = 0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))
        return - torch.exp(arg)

    def term3(self) -> torch.Tensor:
        return torch.e + 20

    def evaluate_true_objective(self, X: torch.Tensor, add_noise=False) -> torch.Tensor:
        return (self.term1(X) + self.term2(X) + self.term3()).unsqueeze(-1)
