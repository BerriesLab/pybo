import torch
from botorch.utils.multi_objective import Hypervolume
from torch import Tensor
from abc import ABC
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.exceptions import BotorchTensorDimensionError, BotorchError, InputDataError
from botorch.utils.transforms import normalize_indices
from botorch.utils.multi_objective.pareto import is_non_dominated


class AvagamaMCMultiOutputConstrainedObjective(MCMultiOutputObjective, ABC):
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
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = 3
        self.num_objectives = 2
        self.num_constraints = 1
        self.negate = [True, True]
        self.ref_point = [300, 150]
        self.noise_std: float or list[float] or None = None
        self.bounds = [(7.5, 15), (3, 7.5), (0.1 * 78, 78)]  # 2 x d: I_MAX, I_P, tau 78 us
        self.outcomes = [0, 1]
        self.num_outcomes = 2
        self.max_hv: float or None = None
        # Bounds validation and registration
        if len(self.bounds) != self.dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {len(self.bounds)=}."
            )

        # Outcomes validation and index conversion
        if self.outcomes is not None:
            if len(self.outcomes) < 2:
                raise BotorchTensorDimensionError("Must specify at least two outcomes for MOO.")
            if any(i < 0 for i in self.outcomes):
                if self._num_outcomes is None:
                    raise BotorchError("num_outcomes is required if any outcomes are less than 0.")
                self._outcomes = normalize_indices(self.outcomes, self.num_outcomes)

        # Noise validation and conversion
        if self.noise_std is not None:
            if isinstance(self.noise_std, list):
                if len(self.noise_std) != self.num_objectives:
                    raise ValueError(
                        f"noise_std list must have length {self.num_objectives}, "
                        f"but got {len(self.noise_std)}"
                    )
                self.noise_std = torch.tensor(self.noise_std)
            elif isinstance(self.noise_std, (float, int)):
                self.noise_std = torch.tensor([self.noise_std] * self.num_objectives)
            else:
                raise TypeError("noise_std must be a float, int, or list of floats.")
        else:
            self.noise_std = None

        # Create tensors
        self.negate = torch.tensor(data=self.negate, device=self.device, dtype=torch.bool)
        self.ref_point = torch.tensor(data=self.ref_point, device=self.device, dtype=self.dtype)
        self.outcomes = torch.tensor(self.outcomes, device=self.device)
        self.bounds = torch.tensor(self.bounds, device=self.device, dtype=self.dtype).transpose(-1, -2)

        # Compute max hv
        if hasattr(self, "estimate_max_hv_mc"):
            self.estimate_max_hv_mc()

    @staticmethod
    def _electrode_wear(X: Tensor) -> Tensor:
        """
        Simulates wear in microns based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return (
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

    @staticmethod
    def _machining_time(X: Tensor) -> Tensor:
        """
        Simulates machining time (down phase) in seconds based on a pre-fitted
        polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return (
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

    @staticmethod
    def _orbiting_time(X: Tensor) -> Tensor:
        """
        Simulates orbiting time in seconds based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return (
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

    @staticmethod
    def _orbiting_time_penalty(X: Tensor) -> Tensor:
        threshold = 40  # in minutes
        return torch.where(X[..., 2] >= threshold, -X[..., 2], -threshold)

    def evaluate_true(self, X: Tensor) -> Tensor:
        """ Evaluate the true (noise-free, unnegated) objective functions at the given input locations.
        It serves as the ground-truth evaluation of the problem and is typically used for benchmarking,
        visualization (e.g., plotting the true Pareto front), or performance assessment of optimization algorithms. """
        machining_time = self._machining_time(X=X)
        electrode_wear = self._electrode_wear(X=X)
        return torch.stack([machining_time, electrode_wear], dim=-1)

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        """ Evaluate the true (noise-free, unnegated) constraint slack at the given input locations.
        A positive slack indicates a feasible point, while a negative value indicates a violation.
        This is useful for computing constraint satisfaction, especially for benchmarking or visualizing
        feasible regions in multi-objective optimization problems."""
        return torch.Tensor(40 - self._orbiting_time(X)).unsqueeze(-1)

    def estimate_max_hv_mc(self, n_samples: int = 100_000, verbose=True):
        """ Estimates the maximum theoretical hypervolume using Sobol-based Monte Carlo sampling.
            Parameters:
                - n_samples (int): Number of MC samples for objective space.
                - verbose (bool): If True, prints progress and results.
            Sets:
                - self._max_hv: Estimated maximum hypervolume value.
            """

        if verbose:
            print("Estimating maximum theoretical hypervolume... ", end="")

        # 1. Sample input space to get Pareto front
        n_design = min(10_000, n_samples // 100)
        lb, ub = self.bounds[0], self.bounds[1]
        X = (ub - lb) * torch.rand(n_design, self.dim, device=self.device, dtype=self.dtype) + lb

        # 2. Evaluate and filter feasible points only if constraints exist
        Y = self.evaluate_true(X)
        if self.num_constraints > 0:
            constraint_slack = self.evaluate_slack_true(X)  # shape: (n_samples, num_constraints)
            feasible_mask = constraint_slack >= 0  # all constraints satisfied
            feasible_mask = feasible_mask.all(dim=-1)
            feasible_Y = Y[feasible_mask]
        else:
            feasible_Y = Y
        if feasible_Y.shape[0] == 0:
            raise ValueError("No feasible samples found to estimate hypervolume.")

        # 3. Work in maximization space: flip only objectives marked as "negate"
        feasible_Y_max = feasible_Y.clone()
        feasible_Y_max[..., self.negate] *= -1

        # 4. Compute Pareto front among feasible samples (in maximization space)
        pareto_mask = is_non_dominated(feasible_Y_max)
        pareto_front_max = feasible_Y_max[pareto_mask]

        # 5. Compute hypervolume in maximization space
        ref_point_max = self.ref_point.clone()
        ref_point_max[..., self.negate] *= -1
        hv = Hypervolume(ref_point_max).compute(pareto_front_max)
        self.max_hv = hv

        if verbose:
            print(f"{self.max_hv:.4f}")

    def add_noise(self, Y: Tensor) -> Tensor:
        """ A method to add noise to the observations. """
        if self.noise_std is not None:
            noise = self.noise_std.to(Y.device) * torch.randn_like(Y)
            return Y + noise
        return Y

    def evaluate(self, X: Tensor, noise: bool = True) -> Tensor:
        """ Evaluate the objective function on input X, adding noise and applying negation
        if specified. This method returns the version of the objective used for optimization. While this
        method is not used for optimization, it can be conveniently used to generate the initial
        population in test problems. """
        Y = self.evaluate_true(X)
        Y = self.add_noise(Y) if noise else Y
        Y[..., self.negate] *= -1
        return Y

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        """ Transform Monte Carlo samples from the model's posterior according to the specified
        objective configuration. It selects the relevant output dimensions (if `outcomes` are specified),
        applies penalties, and optionally applies negation if the objective is formulated as a minimization
        problem but needs to be maximized internally (as is common in acquisition functions like qNEHVI)."""
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)
        selected[..., self.negate] *= -1
        return selected


class AvagamaMCMultiOutputObjective(MCMultiOutputObjective, ABC):
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
        - Orbiting penalty Time: -50 min
    """

    def __init__(self, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = 3
        self.num_objectives = 3
        self.num_constraints = 0
        self.negate = [True, True, False]
        self.ref_point = [300, 150, -50]
        self.noise_std: float or list[float] or None = None
        self.bounds = [(7.5, 15), (3, 7.5), (0.1 * 78, 78)]  # 2 x d: I_MAX, I_P, tau 78 us
        self.outcomes = [0, 1, 2]
        self.num_outcomes = 3
        self.max_hv = None
        self.max_hv: float or None = None

        # Bounds validation and registration
        if len(self.bounds) != self.dim:
            raise InputDataError(
                "Expected the bounds to match the dimensionality of the domain. "
                f"Got {self.dim=} and {len(self.bounds)=}."
            )

        # Outcomes validation and index conversion
        if self.outcomes is not None:
            if len(self.outcomes) < 2:
                raise BotorchTensorDimensionError("Must specify at least two outcomes for MOO.")
            if any(i < 0 for i in self.outcomes):
                if self._num_outcomes is None:
                    raise BotorchError("num_outcomes is required if any outcomes are less than 0.")
                self._outcomes = normalize_indices(self.outcomes, self.num_outcomes)

        # Noise validation and conversion
        if self.noise_std is not None:
            if isinstance(self.noise_std, list):
                if len(self.noise_std) != self.num_objectives:
                    raise ValueError(
                        f"noise_std list must have length {self.num_objectives}, "
                        f"but got {len(self.noise_std)}"
                    )
                self.noise_std = torch.tensor(self.noise_std)
            elif isinstance(self.noise_std, (float, int)):
                self.noise_std = torch.tensor([self.noise_std] * self.num_objectives)
            else:
                raise TypeError("noise_std must be a float, int, or list of floats.")
        else:
            self.noise_std = None

        # Create tensors
        self.negate = torch.tensor(data=self.negate, device=self.device, dtype=torch.bool)
        self.ref_point = torch.tensor(data=self.ref_point, device=self.device, dtype=self.dtype)
        self.outcomes = torch.tensor(self.outcomes, device=self.device)
        self.bounds = torch.tensor(self.bounds, device=self.device, dtype=self.dtype).transpose(-1, -2)

        # Compute max hv
        if hasattr(self, "estimate_max_hv_mc"):
            self.estimate_max_hv_mc()

    @staticmethod
    def _electrode_wear(X: Tensor) -> Tensor:
        """
        Simulates wear in microns based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return (
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

    @staticmethod
    def _machining_time(X: Tensor) -> Tensor:
        """
        Simulates machining time (down phase) in seconds based on a pre-fitted
        polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return (
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

    @staticmethod
    def _orbiting_time(X: Tensor) -> Tensor:
        """
        Simulates orbiting time in seconds based on a pre-fitted polynomial function.
        """
        i_max = X[..., 0]
        i_p = X[..., 1]
        tau_r_max = X[..., 2]

        return (
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

    @staticmethod
    def _orbiting_time_penalty(X: Tensor) -> Tensor:
        threshold = 40  # in minutes
        return torch.where(X[..., 2] >= threshold, -X[..., 2], -threshold)

    # Optional — not required by MCMultiOutputObjective
    def evaluate_true(self, X: Tensor) -> Tensor:
        """ Evaluate the true (noise-free, unnegated) objective functions at the given input locations.
        It serves as the ground-truth evaluation of the problem and is typically used for benchmarking,
        visualization (e.g., plotting the true Pareto front), or performance assessment of optimization algorithms."""
        machining_time = self._machining_time(X=X)
        electrode_wear = self._electrode_wear(X=X)
        orbiting_time = self._orbiting_time(X=X)
        return torch.stack([machining_time, electrode_wear, orbiting_time], dim=-1)

    def estimate_max_hv_mc(self, n_samples: int = 100_000, verbose=True):
        """ Estimates the maximum theoretical hypervolume using Sobol-based Monte Carlo sampling.
            Parameters:
                - n_samples (int): Number of MC samples for objective space.
                - verbose (bool): If True, prints progress and results.
            Sets:
                - self._max_hv: Estimated maximum hypervolume value.
            """

        if verbose:
            print("Estimating maximum theoretical hypervolume... ", end="")

        # 1. Sample input space to get Pareto front
        n_design = min(10_000, n_samples // 100)
        lb, ub = self.bounds[0], self.bounds[1]
        X = (ub - lb) * torch.rand(n_design, self.dim, device=self.device, dtype=self.dtype) + lb

        # 2. Evaluate and filter feasible points only if constraints exist
        Y = self.evaluate_true(X)
        if self.num_constraints > 0:
            constraint_slack = self.evaluate_slack_true(X)  # shape: (n_samples, num_constraints)
            feasible_mask = constraint_slack >= 0  # all constraints satisfied
            feasible_mask = feasible_mask.all(dim=-1)
            feasible_Y = Y[feasible_mask]
        else:
            feasible_Y = Y
        if feasible_Y.shape[0] == 0:
            raise ValueError("No feasible samples found to estimate hypervolume.")

        # 3. Work in maximization space: flip only objectives marked as "negate"
        feasible_Y_max = feasible_Y.clone()
        feasible_Y_max[..., self.negate] *= -1

        # 4. Compute Pareto front among feasible samples (in maximization space)
        pareto_mask = is_non_dominated(feasible_Y_max)
        pareto_front_max = feasible_Y_max[pareto_mask]

        # 5. Compute hypervolume in maximization space
        ref_point_max = self.ref_point.clone()
        ref_point_max[..., self.negate] *= -1
        hv = Hypervolume(ref_point_max).compute(pareto_front_max)
        self.max_hv = hv

        if verbose:
            print(f"{self.max_hv:.4f}")

    # Optional — not required by MCMultiOutputObjective
    def add_noise(self, Y: Tensor) -> Tensor:
        """ A method to add noise to the observations. """
        if self.noise_std is not None:
            noise = self.noise_std.to(Y.device) * torch.randn_like(Y)
            return Y + noise
        return Y

    # Optional — not required by MCMultiOutputObjective
    def evaluate(self, X: Tensor, noise: bool = True) -> Tensor:
        """ Evaluate the objective function on input X, optionally adding noise and applying negation
        if specified. This method returns the version of the objective used for optimization. While this
        method is not used for optimization, it can be conveniently used to generate the initial
        population in test problems. """
        Y = self.evaluate_true(X)
        Y = self.add_noise(Y) if noise else Y
        Y[..., self.negate] *= -1
        return Y

    def forward(self, samples: Tensor, X: Tensor = None) -> Tensor:
        """ Transform Monte Carlo samples from the model's posterior according to the specified
        objective configuration. It selects the relevant output dimensions (if `outcomes` are specified),
        applies penalties, and optionally applies negation if the objective is formulated as a minimization
        problem but needs to be maximized internally (as is common in acquisition functions like qNEHVI)."""
        selected = samples.clone()
        if self.outcomes is not None:
            selected = selected.index_select(-1, self.outcomes)

        # Transform the orbiting time objective to a "satisfaction" metric.
        # The objective is 0 if time >= 40, and (time - 40) if time < 40.
        orbiting_time = selected[..., 2]

        # === Exponential penalty ===
        # violation = torch.clamp(40.0 - orbiting_time, min=0.0)
        # satisfaction = - torch.exp(violation / 1)  # k=5.0

        # === Linear penalty ===
        satisfaction = 1000 * torch.min(torch.zeros_like(orbiting_time), orbiting_time - 40.0)

        selected[..., 2] = satisfaction
        # Negate objectives for minimization
        selected[..., self.negate] *= -1
        return selected

