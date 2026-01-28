from torch import Tensor
from objectives.base_class import MCMultiObjectiveBase
from objectives.variable_registry import *


class BinhAndKorn(MCMultiObjectiveBase):
    """ Two objective problem composed of the Binh and Korn functions. """

    class Obj(VariableRegistry):
        BINH = Config(label="Binh", index=0, bounds=(0.0, 140.0), dtype=torch.float64, to_minimize=True, ref_point=150)
        KORN = Config(label="Korn", index=1, bounds=(0.0, 50.0), dtype=torch.float64, to_minimize=True, ref_point=60)

    class Par(VariableRegistry):
        P1 = Config(label="P1", index=0, bounds=(0.0, 5.0), dtype=torch.float64)
        P2 = Config(label="P2", index=1, bounds=(0.0, 3.0), dtype=torch.float64)

    class InputCon(VariableRegistry):
        C1 = Config(label="C1", index=0, bounds=(0.0, 5.0), dtype=torch.float64, f=self._input_c1)
        C2 = Config(label=)
        C3.

    class OutputCon


    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device, dtype=dtype,
            nonlinear_inequality_input_constraints=[
                (self._input_c1, True),
                (self._input_c2, True)
            ],
        )

    def _f1(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return 4 * x1 ** 2 + 4 * x2 ** 2

    def _f2(self, X: torch.Tensor) -> torch.Tensor:
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return (x1 - 5) ** 2 + (x2 - 5) ** 2

    def _input_c1(self, X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 5)^2 + x1^2 <= 25 """
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return 25 - ((x1 - 5) ** 2 + x2 ** 2)

    def _input_c2(self, X: torch.Tensor) -> torch.Tensor:
        """ A constraint on the input: (x0 - 8)^2 + (x1 + 3)^2 >= 7.7 """
        x1 = X[..., self.Par.P1.index]
        x2 = X[..., self.Par.P2.index]
        return (x1 - 8) ** 2 + (x2 + 3) ** 2 - 7.7

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        f1 = self._f1(X=X)
        f2 = self._f2(X=X)
        return torch.stack([f1, f2], dim=-1)
