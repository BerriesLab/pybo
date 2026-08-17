from pybo.optimizer.base_class import OptimizerBase
from pybo.samplers.sobol import SobolSampler


class SobolOptimizer(OptimizerBase):
    """An optimizer that proposes by sampling, never by modelling."""

    optimizer_type = "sobol"

    # acqf and kernel are accepted and ignored: the tutorials pick the arm at runtime and
    # pass the modeling arguments to whichever class they chose, so this one has to
    # swallow them. Dropping them makes every --strategy sobol run fail in the base
    # __init__ on an unexpected keyword.
    def __init__(self, sampler: SobolSampler | None = None, acqf=None, kernel=None, **kwargs):
        super().__init__(**kwargs)
        self._candidate_sampler = sampler

    @property
    def sampler(self) -> SobolSampler | None:
        """The sampler that draws the candidates, or None until optimize() builds one."""
        return self._candidate_sampler

    def _instantiate_sampler(self):
        super()._validate_objective()
        if self._candidate_sampler is None:
            self._candidate_sampler = SobolSampler(
                device=self._device, dtype=self._dtype, objective=self.objective
            )

    def optimize(self, verbose=True):
        self._instantiate_sampler()
        super().optimize(verbose=verbose)

    def _propose(self, verbose=True):
        if verbose:
            print("Drawing random candidates... ", end="")

        self._new_X = self._candidate_sampler.draw_samples(n=self._batch_size)

        if verbose:
            self._print_success(msg=f"New X: {self._new_X.detach().cpu().numpy()}")

    def _validate_state(self):
        super()._validate_state()
        self._validate_sampler()

    def _validate_sampler(self):
        # optimize() builds one before validating, so None here means _validate_state was
        # called on its own. Nothing to check either way.
        if self._candidate_sampler is None:
            return

        if not isinstance(self._candidate_sampler, SobolSampler):
            raise TypeError(
                f"sampler must be a SobolSampler, got {type(self._candidate_sampler).__name__}.")

        if self._candidate_sampler.objective is not self.objective:
            raise ValueError(
                "sampler.objective is not the objective being optimized: the sampler would "
                "draw candidates from a different search space.")
