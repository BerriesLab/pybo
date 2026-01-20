import torch
from torch import Tensor
from botorch.utils.transforms import unnormalize
from collections.abc import Callable
from abc import ABC, abstractmethod
from scipy.stats.qmc import LatinHypercube
from torch.quasirandom import SobolEngine


class SamplerBase(ABC):
    def __init__(
            self,
            device: torch.device,
            dtype: torch.dtype,
            bounds: torch.Tensor | None = None,
            n_dimensions: int = 2,
            normalize: bool = True,
            linear_equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
            linear_inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
            nonlinear_inequality_constraints: list[tuple[Callable, bool]] | None = None,
            seed: int | None = None
    ):
        self.device = device
        self.dtype = dtype
        self.bounds = bounds
        self.n_dimensions = n_dimensions
        self.normalize = normalize
        self.linear_equality_constraints = linear_equality_constraints
        self.linear_inequality_constraints = linear_inequality_constraints
        self.nonlinear_inequality_constraints = nonlinear_inequality_constraints
        self.seed = seed

    @abstractmethod
    def _generate_base_samples(self, n: int) -> Tensor:
        """ Must be implemented by each subclass."""
        pass

    def _parse_constraints(self, constraint_list):
        num_constraints = len(constraint_list)
        A = torch.zeros(num_constraints, self.n_dimensions, device=self.device, dtype=self.dtype)
        b = torch.zeros(num_constraints, device=self.device, dtype=self.dtype)
        for i, (indices, coefficients, rhs) in enumerate(constraint_list):
            A[i, indices.long()] = coefficients
            b[i] = rhs
        return A, b

    def _project_onto_linear_equality_manifold(self, X: Tensor, A: Tensor, b: Tensor) -> Tensor:
        error = (X @ A.T) - b
        ATA = A @ A.T
        correction = torch.linalg.solve(ATA, error.T).T @ A
        return X - correction

    def draw_samples(self, n: int) -> torch.Tensor:
        valid_x = []
        num_attempts = 0
        max_attempts = 1000
        n_to_draw = int(n)

        while len(valid_x) < n and num_attempts < max_attempts:
            num_attempts += 1

            # 1. Generazione (Metodo specifico della sottoclasse)
            X = self._generate_base_samples(n_to_draw)

            # 2. Unnormalize
            if not self.normalize:
                if self.bounds is None:
                    raise ValueError("If normalize is False, bounds must be provided.")
                X = unnormalize(X, bounds=self.bounds)

            # 3. Proiezione Uguaglianze
            if self.linear_equality_constraints is not None:
                A_eq, b_eq = self._parse_constraints(self.linear_equality_constraints)
                X = self._project_onto_linear_equality_manifold(X, A_eq, b_eq)

            # 4. Filtro Disuguaglianze e Bounds
            constraint_mask = torch.ones(X.shape[0], device=self.device, dtype=torch.bool)

            if self.linear_inequality_constraints is not None:
                A_in, b_in = self._parse_constraints(self.linear_inequality_constraints)
                constraint_mask &= (X @ A_in.T >= b_in).all(dim=-1)

            if self.nonlinear_inequality_constraints:
                for (constraint_fn, _) in self.nonlinear_inequality_constraints:
                    constraint_mask &= (constraint_fn(X) >= 0)

            if self.bounds is not None and not self.normalize:
                within_bounds = (X >= self.bounds[0]).all(dim=-1) & (X <= self.bounds[1]).all(dim=-1)
                constraint_mask &= within_bounds

            X = X[constraint_mask]
            if X.shape[0] > 0:
                valid_x.append(X)

            # Strategia adattiva: se molti campioni vengono scartati, ne generiamo di più al giro dopo
            n_to_draw = n * 2

        if not valid_x:
            raise RuntimeError(f"No valid samples found after {max_attempts} attempts.")

        return torch.cat(valid_x, dim=0)[:n]


class SobolSampler(SamplerBase):
    def _generate_base_samples(self, n: int) -> torch.Tensor:
        engine = SobolEngine(dimension=self.n_dimensions, scramble=True, seed=self.seed)
        return engine.draw(n=n).to(device=self.device, dtype=self.dtype)


class LatinHypercubeSampler(SamplerBase):
    def _generate_base_samples(self, n: int) -> torch.Tensor:
        sampler = LatinHypercube(d=self.n_dimensions, seed=self.seed)
        samples = sampler.random(n=n)
        return torch.tensor(samples, device=self.device, dtype=self.dtype)


class UniformGridSampler(SamplerBase):
    def _generate_base_samples(self, n: int) -> torch.Tensor:
        # Genera una griglia uniforme (molto utile per basse dimensioni)
        points_per_dim = int(n ** (1 / self.n_dimensions))
        grid_axes = [torch.linspace(0, 1, points_per_dim, device=self.device, dtype=self.dtype)
                     for _ in range(self.n_dimensions)]
        grid = torch.meshgrid(*grid_axes, indexing='ij')
        return torch.stack(grid, dim=-1).reshape(-1, self.n_dimensions)
