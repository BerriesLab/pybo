import torch
from torch import Tensor
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine
from utils.types import SamplerType
from botorch.utils.transforms import unnormalize
from collections.abc import Callable


# TODO: fix constraints
class Sampler:
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            sampler_type: SamplerType = SamplerType.Sobol,
            bounds: torch.Tensor | None = None,
            n_dimensions: int = 2,
            normalize: bool = True,
            linear_equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
            linear_inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
            non_linear_inequality_constraints: list[Callable] | None = None
    ):

        self.device = device
        self.dtype = dtype
        self.sampler_type = sampler_type
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.normalize = normalize
        self.linear_equality_constraints = linear_equality_constraints
        self.linear_inequality_constraints = linear_inequality_constraints
        self.non_linear_inequality_constraints = non_linear_inequality_constraints

    def _parse_linear_equality_constraints(self):
        r""""
        Parses the list of equality constraints into a single A and b tensor.
        Each constraint is of the form (indices, coefficients, rhs) and encodes:

            sum( X[indices[i]] * coefficients[i] ) >= rhs

        This matches the BoTorch convention.
        """
        num_constraints = len(self.linear_equality_constraints)
        A = torch.zeros(num_constraints, self.n_dimensions, dtype=self.dtype)
        b = torch.zeros(num_constraints, dtype=self.dtype)

        for i, (indices, coefficients, rhs) in enumerate(self.linear_equality_constraints):
            # The constraint is `Ax <= b`, so we use the given coefficients and rhs
            A[i, indices.long()] = coefficients
            b[i] = rhs

        return A, b

    def _parse_linear_inequality_constraints(self):
        r""""
        Parses the list of inequality constraints into a single A and b tensor.
        Each constraint is of the form (indices, coefficients, rhs) and encodes:

            sum( X[indices[i]] * coefficients[i] ) >= rhs

        This matches the BoTorch convention.
        """
        num_constraints = len(self.linear_inequality_constraints)
        A = torch.zeros(num_constraints, self.n_dimensions, dtype=self.dtype)
        b = torch.zeros(num_constraints, dtype=self.dtype)

        for i, (indices, coefficients, rhs) in enumerate(self.linear_inequality_constraints):
            # The constraint is `Ax <= b`, so we use the given coefficients and rhs
            A[i, indices.long()] = coefficients
            b[i] = rhs

        return A, b

    @staticmethod
    def _project_onto_linear_equality_manifold(X: Tensor, A: Tensor, b: Tensor) -> Tensor:
        """
        Projects samples from the full space onto the linear equality hyperplane.
                    X_proj = x - A.T * (A * A.T)^-1 * (A * x - b)
        """
        # Calculate the projection matrix using torch.linalg.solve for stability
        ATA_inv = torch.linalg.solve(A @ A.T, torch.eye(A.shape[0], dtype=A.dtype))
        # Calculate the error term (A*x - b) for each sample
        error = X @ A.T - b
        # Calculate the correction term
        correction = error @ ATA_inv @ A
        # Project the samples
        X_proj = X - correction
        return X_proj

    def draw_samples(self, n) -> torch.Tensor:
        valid_x = []
        num_attempts = 0
        max_attempts = 1000
        tol = 1e-6

        while len(valid_x) < n and num_attempts < max_attempts:
            num_attempts += 1

            # Draw raw samples
            if self.sampler_type == SamplerType.LatinHypercube:
                sampler = LatinHypercube(d=self.n_dimensions)
                samples = sampler.random(n=n)
                X = torch.tensor(samples, device=self.device, dtype=self.dtype)

            elif self.sampler_type == SamplerType.Sobol:
                sampler = SobolEngine(dimension=self.n_dimensions, scramble=True)
                X = sampler.draw(n=n).to(device=self.device, dtype=self.dtype)

            else:
                raise ValueError("Invalid sampler type.")

            # Unnormalize if needed
            if not self.normalize:
                if self.bounds is None:
                    raise ValueError("If normalize is True, then bounds cannot be None.")
                X = unnormalize(X, bounds=self.bounds)

            # Initialize a combined constraint mask
            constraint_mask = torch.ones(X.shape[0], dtype=torch.bool)

            # Project samples to satisfy linear equality constraints
            if self.linear_equality_constraints is not None:
                A, b = self._parse_linear_equality_constraints()
                X = self._project_onto_linear_equality_manifold(X, A, b)

            # Apply parsed linear inequality constraints (Ax <= b)
            if self.linear_inequality_constraints is not None:
                A, b = self._parse_linear_inequality_constraints()
                linear_inequality_constraint_values = X @ A.T
                # Check if values are less than or equal to the RHS
                linear_in_mask = (linear_inequality_constraint_values >= b).all(dim=-1)
                constraint_mask &= linear_in_mask

            # Apply non-linear constraints
            if self.non_linear_inequality_constraints:
                for constraint_fn in self.non_linear_inequality_constraints:
                    constraint_mask &= (constraint_fn(X) <= 0)

            X = X[constraint_mask]
            valid_x.append(X)

        valid_samples = torch.cat(valid_x, dim=0)
        if valid_samples.shape[0] < n:
            raise RuntimeError(f"Only {valid_samples.shape[0]} valid samples found after {num_attempts} attempts.")

        return valid_samples[:n]
