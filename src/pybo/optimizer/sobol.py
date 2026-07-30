from pybo.optimizer.base_class import OptimizerBase
from pybo.samplers.base_class import SamplerBase
from pybo.samplers.sobol import SobolSampler


class SobolOptimizer(OptimizerBase):
    """An optimizer that proposes by sampling, never by modelling."""

    def __init__(self, sampler: SamplerBase | None = None, acqf=None, kernel=None, **kwargs):
        super().__init__(**kwargs)
        self._candidate_sampler = sampler

    @property
    def sampler(self) -> SamplerBase | None:
        """The sampler that draws the candidates, or None until the first proposal
        builds the default one."""
        return self._candidate_sampler

    def _propose(self, verbose=True):
        """Draw the next batch instead of modeling it.

        draw_samples() rejects against the objective's X constraints, so the baseline
        searches the same feasible region the modeling arm is restricted to.

        A sampler needs the objective, so one that was not supplied is built here rather
        than in __init__: optimize() validates the objective before proposing, which lets
        the optimizer be constructed before the objective exists."""
        if self._candidate_sampler is None:
            self._candidate_sampler = SobolSampler(
                device=self._device, dtype=self._dtype, objective=self.objective
            )

        if verbose:
            print("Drawing random candidates... ", end="")

        self._new_X = self._candidate_sampler.draw_samples(n=self._batch_size)

        if verbose:
            self._print_success(msg=f"New X: {self._new_X.detach().cpu().numpy()}")
