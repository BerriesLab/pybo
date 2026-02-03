import torch
from objectives.base_class import MCSingleObjectiveBase

from objectives.variable_registry import *


class Ackley(MCSingleObjectiveBase):
    """ Unconstrained single objective problem. """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-5.0, 5.0)),
                ParCfg(label="P2", index=1, bounds=(-5.0, 5.0)),
            ],
            obj_cfg=[
                ObjCfg(label="Ackley", index=0, bounds=(0.0, 15.0), to_minimize=True, f=self._ackley)
            ]
        )

    def _term1(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[:, self.get_par_idx("P1")]
        x2 = X[:, self.get_par_idx("P2")]
        arg = -0.2 * torch.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
        return -20 * torch.exp(arg)

    def _term2(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[:, self.get_par_idx("P1")]
        x2 = X[:, self.get_par_idx("P2")]
        arg = 0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))
        return - torch.exp(arg)

    def _term3(self) -> torch.Tensor:
        return torch.e + 20

    def _ackley(self, X: torch.Tensor) -> torch.Tensor:
        return (self._term1(X) + self._term2(X) + self._term3()).unsqueeze(-1)
