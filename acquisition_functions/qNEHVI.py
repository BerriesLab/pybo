from typing import Callable, List
import torch
from botorch.acquisition.multi_objective import MCMultiOutputObjective
from botorch.models.model import Model
from botorch.models.transforms import Normalize
from botorch.sampling import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from torch import Tensor
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement


# This class extends qNoisyExpectedHypervolumeImprovement to incorporate an
# exploration component. The exploration is driven by the uncertainty (standard
# deviation) of the model's predictions for the candidate points.
class qExplorationWeightedNEHVI(qNoisyExpectedHypervolumeImprovement):
    """
    q-Noisy Expected Hypervolume Improvement that favors exploration.

    This class extends `qNoisyExpectedHypervolumeImprovement` by adding an
    exploration component to the acquisition function. The exploration component
    is based on the uncertainty (standard deviation) of the model's predictions
    for the candidate points, encouraging the selection of points in less
    certain regions of the objective space.
    """

    def __init__(
            self,
            model: Model,
            ref_point: list[float] | Tensor,
            X_baseline: Tensor,
            sampler: MCSampler | None = None,
            objective: MCMultiOutputObjective | None = None,
            constraints: list[Callable[[Tensor], Tensor]] | None = None,
            X_pending: Tensor | None = None,
            eta: Tensor | float = 1e-3,
            fat: bool = False,
            prune_baseline: bool = False,
            alpha: float = 0.0,
            cache_pending: bool = True,
            max_iep: int = 0,
            incremental_nehvi: bool = True,
            cache_root: bool = True,
            marginalize_dim: int | None = None,
            exploration_weight: float = 0.1,  # New parameter: controls exploration emphasis
    ) -> None:
        """
        Initializes the qExplorationWeightedNEHVI acquisition function.

        Args:
            model: A fitted model (e.g., a Gaussian Process model).
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
                This is a reference point for the objective values (idx.e. after
                applying `objective` to the samples).
            X_baseline: A `r x d`-dim Tensor of `r` design points that have already
                been observed. These points are considered as potential approximate
                pareto-optimal design points.
            sampler: The sampler used to draw base samples. If not given,
                a sampler is generated using `get_sampler`.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated. Defaults to `IdentityMCMultiOutputObjective()`.
            constraints: A list of callables, each mapping a Tensor of dimension
                `sample_shape x batch-shape x q x m` to a Tensor of dimension
                `sample_shape x batch-shape x q`, where negative values imply
                feasibility. The acquisition function will compute expected feasible
                hypervolume.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that
                have points that have been submitted for function evaluation, but
                have not yet been evaluated.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints.
            fat: A Boolean flag indicating whether to use the heavy-tailed approximation
                of the constraint indicator.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the pareto optimal and better than the
                reference point.
            alpha: The hyperparameter controlling the approximate non-dominated
                partitioning.
            cache_pending: A boolean indicating whether to use cached box
                decompositions (CBD) for handling pending points.
            max_iep: The maximum number of pending points before the box
                decompositions will be recomputed.
            incremental_nehvi: A boolean indicating whether to compute the
                incremental NEHVI from the `idx`th point where `idx=1, ..., q`
                under sequential greedy optimization, or the full qNEHVI over
                `q` points.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline` and use low-rank updates.
            marginalize_dim: A batch dimension that should be marginalized.
            exploration_weight: A non-negative float that controls the emphasis on
                exploration. A higher value leads to more exploration.
                Set to `0.0` to disable exploration and behave like standard qNEHVI.
        """
        # Call the constructor of the parent class (qNoisyExpectedHypervolumeImprovement)
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )
        # Validate and store the new exploration_weight parameter
        if not isinstance(exploration_weight, (float, int)) or exploration_weight < 0:
            raise ValueError("`exploration_weight` must be a non-negative float.")
        self.exploration_weight = float(exploration_weight)

    def forward(self, X: Tensor) -> Tensor:
        """
        Computes the q-Noisy Expected Hypervolume Improvement with an added
        exploration component.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `q` design points for which
                to compute the acquisition value.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values.
        """
        # 1. Calculate the base qNEHVI value using the parent's forward method.
        # This handles the core logic of hypervolume improvement, including
        # model posterior calculations, sampling, and handling of baseline/pending points.
        base_nehvi = super().forward(X)

        # 2. If exploration_weight is 0, no need to compute an exploration term.
        # In this case, the behavior is identical to the original qNEHVI.
        if self.exploration_weight == 0.0:
            return base_nehvi

        # 3. Compute the posterior for only the new candidate points `X`.
        posterior_X = self.model.posterior(X)

        # 4. Extract the standard deviation of the posterior predictions for `X`.
        # Sum across the `q` and `m` dimensions to get a single scalar or
        # batch-wise exploration value. This sum serves as a simple measure of
        # the overall uncertainty for the proposed batch of points.
        exploration_term = posterior_X.stddev.sum(dim=[-1, -2])

        # 5. Add the weighted exploration term to the base NEHVI.
        # The `exploration_weight` scales the contribution of the uncertainty.
        # A higher weight means more emphasis on exploring uncertain regions.
        # The `exploration_term` will be broadcasted if `base_nehvi` has a
        # batch dimension and `exploration_term` is scalar or has a compatible
        # batch shape.
        return base_nehvi + self.exploration_weight * exploration_term


# This class extends qNoisyExpectedHypervolumeImprovement to incorporate a
# diversity-promoting penalty based on distance to previously observed points.
class qDiversityWeightedNEHVI(qNoisyExpectedHypervolumeImprovement):
    """
    q-Noisy Expected Hypervolume Improvement that penalizes points too close
    to previously observed data.

    This class extends `qNoisyExpectedHypervolumeImprovement` by adding a
    penalty to the acquisition function value if a candidate point (`X`) is
    within a specified minimum distance (`min_dist_radius`) of any point in
    the `X_baseline` (observed data). This encourages exploration by
    discouraging sampling in already well-explored regions.
    """

    def __init__(
            self,
            model: Model,
            ref_point: List[float] | Tensor,
            X_baseline: Tensor,
            sampler: MCSampler | None = None,
            objective: MCMultiOutputObjective | None = None,
            constraints: List[Callable[[Tensor], Tensor]] | None = None,
            X_pending: Tensor | None = None,
            eta: Tensor | float = 1e-3,
            fat: bool = False,
            prune_baseline: bool = False,
            alpha: float = 0.0,
            cache_pending: bool = True,
            max_iep: int = 0,
            incremental_nehvi: bool = True,
            cache_root: bool = True,
            marginalize_dim: int | None = None,
            min_dist_radius: float = 0.05,  # New parameter: minimum allowed distance
            distance_penalty_weight: float = 1.0,  # New parameter: strength of the penalty
    ) -> None:
        """
        Initializes the qDiversityWeightedNEHVI acquisition function.

        Args:
            model: A fitted model (e.g., a Gaussian Process model).
            ref_point: A list or tensor with `m` elements representing the reference
                point (in the outcome space) w.r.t. to which compute the hypervolume.
            X_baseline: A `r x d`-dim Tensor of `r` design points that have already
                been observed.
            sampler: The sampler used to draw base samples.
            objective: The MCMultiOutputObjective under which the samples are
                evaluated.
            constraints: A list of callables for feasibility.
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that
                have points that have been submitted for function evaluation.
            eta: The temperature parameter for the sigmoid function used for the
                differentiable approximation of the constraints.
            fat: A Boolean flag indicating whether to use the heavy-tailed approximation
                of the constraint indicator.
            prune_baseline: If True, remove points in `X_baseline` that are
                highly unlikely to be the pareto optimal.
            alpha: The hyperparameter controlling the approximate non-dominated
                partitioning.
            cache_pending: A boolean indicating whether to use cached box
                decompositions (CBD) for handling pending points.
            max_iep: The maximum number of pending points before the box
                decompositions will be recomputed.
            incremental_nehvi: A boolean indicating whether to compute the
                incremental NEHVI.
            cache_root: A boolean indicating whether to cache the root
                decomposition over `X_baseline`.
            marginalize_dim: A batch dimension that should be marginalized.
            min_dist_radius: A non-negative float. Candidate points within this
                radius of any point in `X_baseline` will be penalized.
            distance_penalty_weight: A non-negative float that controls the
                strength of the distance-based penalty. A higher value means
                a stronger penalty for points too close to `X_baseline`.
        """
        # Call the constructor of the parent class (qNoisyExpectedHypervolumeImprovement)
        super().__init__(
            model=model,
            ref_point=ref_point,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            X_pending=X_pending,
            eta=eta,
            fat=fat,
            prune_baseline=prune_baseline,
            alpha=alpha,
            cache_pending=cache_pending,
            max_iep=max_iep,
            incremental_nehvi=incremental_nehvi,
            cache_root=cache_root,
            marginalize_dim=marginalize_dim,
        )
        # Validate and store the new diversity parameters
        if not isinstance(min_dist_radius, (float, int)) or min_dist_radius < 0:
            raise ValueError("`min_dist_radius` must be a non-negative float.")
        self.min_dist_radius = float(min_dist_radius)

        if not isinstance(distance_penalty_weight, (float, int)) or distance_penalty_weight < 0:
            raise ValueError("`distance_penalty_weight` must be a non-negative float.")
        self.distance_penalty_weight = float(distance_penalty_weight)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Computes the q-Noisy Expected Hypervolume Improvement with an added
        diversity penalty.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of `q` design points for which
                to compute the acquisition value.

        Returns:
            A `batch_shape`-dim Tensor of acquisition values.
        """
        # 1. Calculate the base qNEHVI value using the parent's forward method.
        #    Assuming `super().forward(X)` correctly computes this.
        base_nehvi = super().forward(X)

        # 2. If penalty weight or radius is zero, no need to compute a penalty term.
        if self.distance_penalty_weight == 0.0 or self.min_dist_radius == 0.0:
            return base_nehvi

        # Ensure X_baseline has a batch dimension for consistent broadcasting.
        X_baseline_processed = self.X_baseline
        if X_baseline_processed.ndim == 2:
            X_baseline_processed = X_baseline_processed.unsqueeze(0)  # Becomes (1 x r x d)

        # Expand X to (batch_shape x q x 1 x d) for element-wise subtraction with X_baseline, .
        X_expanded = X.unsqueeze(-2)

        # Normalize X - Note that X_baseline is already normalized.
        normalizer = Normalize(d=X_expanded.shape[-1], bounds=self.objective.bounds)
        X_expanded = normalizer(X_expanded)

        # Calculate squared Euclidean distances between each candidate point in X
        # and all points in X_baseline.
        squared_distances = (X_expanded - X_baseline_processed).pow(2).sum(dim=-1)

        # Take the square root to get Euclidean distances.
        distances = torch.sqrt(squared_distances)

        # Find the minimum distance from each candidate point (in X) to any point in X_baseline.
        min_distances_to_baseline, _ = distances.min(dim=-1)

        # Calculate the penalty factor: max(0, min_dist_radius - min_distance).
        # This means the penalty is 0 if distance >= min_dist_radius, and positive otherwise.
        # The penalty increases as the point gets closer to an existing point.
        penalty_factor = torch.relu(self.min_dist_radius - min_distances_to_baseline)
        penalty_factor = normalizer.untransform(penalty_factor)

        # Sum the penalties for all `q` points in the batch and apply the weight.
        # The sum ensures that if multiple points in the batch are too close,
        # their combined penalty is considered.
        total_penalty = penalty_factor.sum(dim=-1)

        # Subtract the total penalty from the base NEHVI value.
        return base_nehvi - self.distance_penalty_weight * total_penalty
