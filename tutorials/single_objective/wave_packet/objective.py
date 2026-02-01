import torch
from objectives.base_class import MCSingleObjectiveBase

from objectives.variable_registry import ParCfg, ObjCfg


class WavePacket(MCSingleObjectiveBase):
    def __init__(self, device: torch.device, dtype: torch.dtype, ):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="P1", index=0, bounds=(-1.0, 1.0))
            ],
            obj_cfg=[
                ObjCfg(label="F1", index=0, to_minimize=True, bounds=(-1.5, 1.5), f=self._f1)
            ],
        )

        self.p = 1 / 2
        self.sigma = 0.4
        self.k0 = 2 * torch.pi / self.p
        self.x0 = 0

    def _f11(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((X - self.x0) / self.sigma) ** 2)

    def _f12(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.k0 * X)

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        f1 = self._f11(X=X)
        f2 = self._f12(X=X)
        f = f1 * f2
        return f
