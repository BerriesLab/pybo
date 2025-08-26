import torch
from torch import Tensor
from constraints.output_constraints import Identity
from objectives.base_class import MCMultiOutputBase


class FormACOMCMultiOutputConstrainedObjective(MCMultiOutputBase):
    """
    2x objectives:
        - Machining Time: Originally intended for minimization.
        - Electrode Wear: Originally intended for minimization.

    1x constraints:
        - Orbiting Time: Requires values > 40 mins.

    3x parameters:
        - Maximum Current [7.5, 15]
        - Pedestal Current [3.0, 7.5]
        - Maximum Ramp Time [0.1*78, 1*78], where ON time = 78 us

    Reference point:
        - Machining Time: 300 min
        - Electrode Wear: 150 um
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=3,
            parameter_names=[
                "Maximum Current (A)",
                "Pedestal Current (A)",
                "Maximum Ramp Time (μs)"
            ],
            num_objectives=2,
            objective_names=[
                "Machining Time (min)",
                "Electrode Wear (μm)",
            ],
            num_constraints=1,
            constraint_names=[
                "Orbiting Time (min)"
            ],
            num_trackers=1,
            tracker_names=[
                "Orbiting Time (min)"
            ],
            bounds=[(7.5, 15), (3, 7.5), (0.1 * 78, 78)],
            obj_to_minimize=[True, True],
            ref_point=[300, 150],
            num_outcomes=2,
            outcomes=[0, 1],
            gt_noise_std=None,
            max_hv=None,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=[Identity(index=-1)],
            estimate_max_hv=True,
        )

    @staticmethod
    def _electrode_wear(X: Tensor) -> Tensor:
        """
        Simulates wear in microns based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return torch.clamp(
            input=26.301475188947435
                  - 19.166867643857774 * i_max
                  + 48.32116975101596 * i_p
                  - 2.2004820692692393 * tau_r_max
                  + 1.610831887686114 * i_max ** 2
                  - 1.7060582358070433 * i_max * i_p
                  - 0.09448612682328417 * i_max * tau_r_max
                  - 2.2369331180580914 * i_p ** 2
                  + 0.12893509602180986 * i_p * tau_r_max
                  + 0.02736891179134915 * tau_r_max ** 2,
            min=0
        )

    @staticmethod
    def _machining_time(X: Tensor) -> Tensor:
        """
        Simulates machining time (down phase) in seconds based on a pre-fitted
        polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return torch.clamp(
            input=616.3490679119025
                  - 39.079346209938606 * i_max
                  - 46.683313051874606 * i_p
                  + 1.732712663059158 * tau_r_max
                  + 0.17007512603265695 * i_max ** 2
                  + 0.5782395309343626 * i_max * i_p
                  + 0.5065065733380472 * i_max * tau_r_max
                  + 3.069882379450696 * i_p ** 2
                  - 0.4865913603357419 * i_p * tau_r_max
                  - 0.046096819818593815 * tau_r_max ** 2,
            min=0
        )

    @staticmethod
    def _orbiting_time(X: Tensor) -> Tensor:
        """
        Simulates orbiting time in seconds based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return torch.clamp(
            input=188.4485094756797
                  - 21.28654897663603 * i_max
                  - 4.222217726699118 * i_p
                  + 0.17654656533899832 * tau_r_max
                  + 0.6689645319172054 * i_max ** 2
                  + 0.6548427659792726 * i_max * i_p
                  - 0.024689990372160464 * i_max * tau_r_max
                  - 0.6374105462985316 * i_p ** 2
                  + 0.03550705380735647 * i_p * tau_r_max
                  - 0.00016572486105292938 * tau_r_max ** 2,
            min=0
        )

    def evaluate_true_objective(self, X: Tensor, add_noise=False) -> Tensor:
        machining_time = self._machining_time(X=X)
        electrode_wear = self._electrode_wear(X=X)
        return torch.stack([machining_time, electrode_wear], dim=-1)

    def evaluate_true_constraint(self, X: Tensor) -> Tensor:
        """
        orbiting_time >= 40 min -> 40 - orbiting_time <= 0
        """
        c = 40 - self._orbiting_time(X=X)
        return c.unsqueeze(dim=-1)

    def evaluate_trackers(self, X: Tensor) -> Tensor:
        return self._orbiting_time(X=X).unsqueeze(dim=-1)

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected


class FormACOMCMultiOutputObjective(MCMultiOutputBase):
    """
    3x objectives:
        - Machining Time: Originally intended for minimization.
        - Electrode Wear: Originally intended for minimization.
        - Orbiting Time: Requires values > 40 mins -> intended for maximization.

    3x parameters:
        - Maximum Current [7.5, 15]
        - Pedestal Current [3.0, 7.5]
        - Maximum Ramp Time [0.1*78, 1*78], where ON time = 78 us

    Reference point:
        - Machining Time: 300 min
        - Electrode Wear: 150 um
        - Orbiting Time: 10 min
        - Orbiting Penalty Time: -50 min (see Note below)

    Note: this objective exhibits the typical user flexibility in defining objective penalties. Specifically,
    "evaluate_true_objective" can be defined to return the "orbiting_time" or the "orbiting_time_penalty". If it
    returns the "orbiting_time", then the penalty must be included in the forward method, and it is subject to
    rescale. If it returns the "orbiting_time_penalty", then the forward method should pass the unaltered values
    without adding any penalty. Both strategies are legit. However, when evaluate_true_objective returns the
    orbiting_time_penalty, the actual orbiting time is lost forever, and there is no way for the user to recover
    the experimental value without any workarounds. A third option would be to add the penalty as a fourth optimization
    objective, and leave the 3rd objective as free to explore: not to optimize.
    The reference point must reflect the choice of objective.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__(
            device=device,
            dtype=dtype,
            dim=3,
            parameter_names=[
                "Maximum Current (A)",
                "Pedestal Current (A)",
                "Maximum Ramp Time (μs)"
            ],
            num_objectives=3,
            objective_names=[
                "Machining Time (min)",
                "Electrode Wear (μm)",
                "Orbiting Time Penalty (min)",
            ],
            bounds=[(7.5, 15), (3, 7.5), (0.1 * 78, 78)],
            obj_to_minimize=[True, True, False],
            ref_point=[300, 150, -50],
            num_outcomes=3,
            outcomes=[0, 1, 2],
            num_trackers=1,
            tracker_names=[
                "Orbiting Time (min)"
            ],
            num_constraints=0,
            constraint_names=None,
            gt_noise_std=None,
            max_hv=None,
            linear_equality_input_constraints=None,
            linear_inequality_input_constraints=None,
            nonlinear_inequality_input_constraints=None,
            output_constraints=None,
            estimate_max_hv=True,
        )

    @staticmethod
    def _electrode_wear(X: Tensor) -> Tensor:
        """
        Simulates wear in microns based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return torch.clamp(
            input=26.301475188947435
                  - 19.166867643857774 * i_max
                  + 48.32116975101596 * i_p
                  - 2.2004820692692393 * tau_r_max
                  + 1.610831887686114 * i_max ** 2
                  - 1.7060582358070433 * i_max * i_p
                  - 0.09448612682328417 * i_max * tau_r_max
                  - 2.2369331180580914 * i_p ** 2
                  + 0.12893509602180986 * i_p * tau_r_max
                  + 0.02736891179134915 * tau_r_max ** 2,
            min=0
        )

    @staticmethod
    def _machining_time(X: Tensor) -> Tensor:
        """
        Simulates machining time (down phase) in seconds based on a pre-fitted
        polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return torch.clamp(
            input=616.3490679119025
                  - 39.079346209938606 * i_max
                  - 46.683313051874606 * i_p
                  + 1.732712663059158 * tau_r_max
                  + 0.17007512603265695 * i_max ** 2
                  + 0.5782395309343626 * i_max * i_p
                  + 0.5065065733380472 * i_max * tau_r_max
                  + 3.069882379450696 * i_p ** 2
                  - 0.4865913603357419 * i_p * tau_r_max
                  - 0.046096819818593815 * tau_r_max ** 2,
            min=0
        )

    @staticmethod
    def _orbiting_time(X: Tensor) -> Tensor:
        """
        Simulates orbiting time in seconds based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return torch.clamp(
            input=188.4485094756797
                  - 21.28654897663603 * i_max
                  - 4.222217726699118 * i_p
                  + 0.17654656533899832 * tau_r_max
                  + 0.6689645319172054 * i_max ** 2
                  + 0.6548427659792726 * i_max * i_p
                  - 0.024689990372160464 * i_max * tau_r_max
                  - 0.6374105462985316 * i_p ** 2
                  + 0.03550705380735647 * i_p * tau_r_max
                  - 0.00016572486105292938 * tau_r_max ** 2,
            min=0
        )

    def _linear_orbiting_penalty(self, X: Tensor) -> Tensor:
        """
        Define a linear orbiting penalty for X < 40.
        """
        orbiting_time = self._orbiting_time(X)
        return torch.clamp(orbiting_time - 40.0, max=0.0)

    def _exponential_orbiting_penalty(self, X: Tensor, k: float = 1) -> Tensor:
        """
        Define an exponential orbiting penalty for X < 40.
        """
        orbiting_time = self._orbiting_time(X)
        violation = 40.0 - orbiting_time
        return - torch.clamp(input=torch.exp(violation / k), max=1000)

    def _quadratic_orbiting_penalty(self, X: Tensor) -> Tensor:
        """
        Define a quadratic orbiting penalty for X < 40.
        """
        orbiting_time = self._orbiting_time(X)
        return -torch.square(input=torch.clamp(input=orbiting_time - 40.0, max=0.0))

    def evaluate_true_objective(self, X: Tensor) -> Tensor:
        machining_time = self._machining_time(X=X)
        electrode_wear = self._electrode_wear(X=X)
        orbiting_time_penalty = self._linear_orbiting_penalty(X=X)
        # orbiting_time_penalty = self._quadratic_orbiting_penalty(X=X)
        # orbiting_time_penalty = self._exponential_orbiting_penalty(X=X)
        return torch.stack([machining_time, electrode_wear, orbiting_time_penalty], dim=-1)

    def evaluate_trackers(self, X: Tensor) -> Tensor:
        return self._orbiting_time(X=X).unsqueeze(dim=-1)

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.obj_to_minimize] *= -1
        return selected
