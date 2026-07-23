import torch
from pybo.objectives.base_class import MCSingleObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg


class WavePacket(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(bounds=(-1.0, 1.0))
            ],
            obj_cfg=[
                ObjCfg(to_minimize=True, bounds=(-1.5, 1.5))
            ],
        )

        self.p = 1 / 2
        self.sigma = 0.4
        self.k0 = 2 * torch.pi / self.p
        self.x0 = 0

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        term1 = torch.exp(-0.5 * ((X - self.x0) / self.sigma) ** 2)
        term2 = torch.sin(self.k0 * X)
        f = term1 * term2
        return f
