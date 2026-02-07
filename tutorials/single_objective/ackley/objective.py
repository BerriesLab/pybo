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
                ParCfg(bounds=(-5.0, 5.0)),
                ParCfg(bounds=(-5.0, 5.0)),
            ],
            obj_cfg=[
                ObjCfg(label="Ackley", bounds=(0.0, 15.0), ref_point=16.0, to_minimize=True)
            ]
        )

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[:, 0]
        x2 = X[:, 1]
        arg1 = -0.2 * torch.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
        term1 = -20 * torch.exp(arg1)
        arg2 = 0.5 * (torch.cos(2 * torch.pi * x1) + torch.cos(2 * torch.pi * x2))
        term2 = - torch.exp(arg2)
        term3 = torch.e + 20
        return (term1 + term2 + term3).unsqueeze(-1)
