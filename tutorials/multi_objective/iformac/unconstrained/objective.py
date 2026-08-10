import torch
from pybo.constraints.output_constraints import Identity, LowerBound, UpperBound
from pybo.objectives.base_class import MCMultiObjectiveBase
from pybo.objectives.variable_registry import ParCfg, ObjCfg, IneqYConCfg, TrkCfg


class IFormAC(MCMultiObjectiveBase):
    """
    3x objectives:
        - Machining Time: Originally intended for minimization.
        - Tool Wear: Originally intended for minimization.
        - Orbiting Time Penalty: Intended for mininization.

    3x parameters:
        - Maximum Current [7.5, 15]
        - Pedestal Current [3.0, 7.5]
        - Maximum Ramp Time [0.1*78000, 1*78000] ns, where ON time = 78000 ns

    Reference point:
        - Machining Time: 300 min
        - Tool Wear: 150 um
        - Orbiting Penalty Time: 50 min
    """

    _t_on = 78000  # ns
    _orbiting_target = 40
    _delta = _orbiting_target / 100 * 5

    # The three polynomials below were fitted against the ramp time in microseconds, while
    # the parameter itself is carried in nanoseconds - the unit the rig logs and reports it
    # in. Each converts where it consumes tau, rather than at the class boundary, so what X
    # holds stays the unit everything outside this class speaks.

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            par_cfg=[
                ParCfg(label="Maximum Current (A)", bounds=(7.5, 15.0)),
                ParCfg(label="Pedestal Current (A)", bounds=(3.0, 7.5)),
                ParCfg(label="Maximum Ramp Time (ns)", bounds=(0.1 * self._t_on, self._t_on))
            ],
            obj_cfg=[
                ObjCfg(label="Machining Time (min)", to_minimize=True, ref_point=360.0, bounds=(0, 350)),
                ObjCfg(label="Tool Wear (μm)", to_minimize=True, ref_point=160.0, bounds=(0, 150)),
                ObjCfg(label="orbiting time penalty", to_minimize=True, ref_point=50.0, bounds=None)
            ],
            trk_cfg=[
                TrkCfg(label="orbiting time deviation", bounds=(-20, 20))
            ],
        )

    @staticmethod
    def _electrode_wear(X: torch.Tensor) -> torch.Tensor:
        """
        Simulates wear in microns based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2] * 1e-3  # ns -> us, the unit the fit was made in

        val = (
                26.301475188947435
                - 19.166867643857774 * i_max
                + 48.32116975101596 * i_p
                - 2.2004820692692393 * tau_r_max
                + 1.610831887686114 * i_max ** 2
                - 1.7060582358070433 * i_max * i_p
                - 0.09448612682328417 * i_max * tau_r_max
                - 2.2369331180580914 * i_p ** 2
                + 0.12893509602180986 * i_p * tau_r_max
                + 0.02736891179134915 * tau_r_max ** 2
        )

        return torch.clamp(input=val, min=0)

    @staticmethod
    def _machining_time(X: torch.Tensor) -> torch.Tensor:
        """
        Simulates machining time (down phase) in seconds based on a pre-fitted
        polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2] * 1e-3  # ns -> us, the unit the fit was made in

        val = (
                616.3490679119025
                - 39.079346209938606 * i_max
                - 46.683313051874606 * i_p
                + 1.732712663059158 * tau_r_max
                + 0.17007512603265695 * i_max ** 2
                + 0.5782395309343626 * i_max * i_p
                + 0.5065065733380472 * i_max * tau_r_max
                + 3.069882379450696 * i_p ** 2
                - 0.4865913603357419 * i_p * tau_r_max
                - 0.046096819818593815 * tau_r_max ** 2
        )

        return torch.clamp(input=val, min=0)

    @staticmethod
    def _orbiting_time(X: torch.Tensor) -> torch.Tensor:
        """
        Simulates orbiting time in seconds based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2] * 1e-3  # ns -> us, the unit the fit was made in

        val = (
                188.4485094756797
                - 21.28654897663603 * i_max
                - 4.222217726699118 * i_p
                + 0.17654656533899832 * tau_r_max
                + 0.6689645319172054 * i_max ** 2
                + 0.6548427659792726 * i_max * i_p
                - 0.024689990372160464 * i_max * tau_r_max
                - 0.6374105462985316 * i_p ** 2
                + 0.03550705380735647 * i_p * tau_r_max
                - 0.00016572486105292938 * tau_r_max ** 2
        )

        return torch.clamp(input=val, min=0)

    def _linear_orbiting_penalty(self, X: torch.Tensor) -> torch.Tensor:
        """
        Define a linear orbiting penalty for X != 40.
        """
        orbiting_time = self._orbiting_time(X)
        penalty = (orbiting_time - self._orbiting_target).abs()
        return torch.clamp(penalty, max=1000.0)

    def _exponential_orbiting_penalty(self, X: torch.Tensor, k: float = 1) -> torch.Tensor:
        """
        Define an exponential orbiting penalty for X != 40.
        """
        orbiting_time = self._orbiting_time(X)
        deviation = (orbiting_time - self._orbiting_target).abs()
        penalty = torch.exp(deviation / k) - 1.0
        return torch.clamp(penalty, max=1000.0)

    def _quadratic_orbiting_penalty(self, X: torch.Tensor) -> torch.Tensor:
        """
        Define a quadratic orbiting penalty for X != 40.
        """
        orbiting_time = self._orbiting_time(X)
        deviation = torch.square(orbiting_time - self._orbiting_target).abs()
        return torch.clamp(input=deviation, max=1000.0)

    def evaluate_true_objective(self, X: torch.Tensor) -> torch.Tensor:
        machining_time = self._machining_time(X=X)
        electrode_wear = self._electrode_wear(X=X)
        # orbiting_time_penalty = self._linear_orbiting_penalty(X=X)
        orbiting_time_penalty = self._quadratic_orbiting_penalty(X=X)
        # orbiting_time_penalty = self._exponential_orbiting_penalty(X=X)
        return torch.stack([machining_time, electrode_wear, orbiting_time_penalty], dim=-1)

    def evaluate_tracker(self, X: torch.Tensor) -> torch.Tensor:
        Y = self._orbiting_time(X=X).unsqueeze(dim=-1)
        return Y - self._orbiting_target
