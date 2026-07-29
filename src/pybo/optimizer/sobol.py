"""The do-nothing arm every optimizer result is measured against.

A hypervolume of 50 means nothing on its own: the question is always how much of it a
budget of random draws would have found anyway. SobolOptimizer spends the same budget on
the same problem while ignoring everything it has learned, so the difference between the
two traces is what the surrogate and the acquisition function actually bought.

It subclasses BayesianOptimizer rather than reimplementing the loop, because the point of
the comparison is that both arms are scored identically - same feasibility rules, same
reference point, same hypervolume code, same summary.json. Only the choice of the next
point differs. optimize() keeps the metric phase verbatim and replaces the surrogate and
the acquisition function with a draw from the sampler.

    from pybo.optimizer.sobol import SobolOptimizer
    bo = SobolOptimizer(sampler=sampler, device=DEVICE, dtype=DTYPE, objective=objective,
                        X=X, Y_obj=Y_obj, batch_size=q)

Give it the same sampler instance that drew the initial design: SobolSampler holds one
engine, so the run continues the sequence instead of redrawing points near the ones
already spent. That also makes this a deliberately strong baseline - a low-discrepancy
sweep covers the box more evenly than i.i.d. uniform draws would, so it is harder to beat
than "random" suggests.
"""
import time

from pybo.optimizer.bayesian import BayesianOptimizer
from pybo.samplers.base_class import SamplerBase


class SobolOptimizer(BayesianOptimizer):
    """A BayesianOptimizer that proposes by sampling, never by modelling."""

    def __init__(self, sampler: SamplerBase, **kwargs):
        # Callers pass the same arguments they would give the optimizer, so a tutorial
        # switches arms without rewriting its setup. The two that describe machinery this
        # arm never builds are dropped, so summary.json does not record a kernel that was
        # never fitted or an acquisition function that was never optimized.
        kwargs["acqf"] = None
        kwargs["kernel"] = None
        super().__init__(**kwargs)
        # Not _sampler: that name is taken by the MC sampler the acquisition function
        # integrates over, which this arm never builds either.
        self._candidate_sampler = sampler

    def optimize(self, verbose=True):
        """Score the data, then draw the next batch instead of modelling it.

        Phases 1 of BayesianOptimizer.optimize() verbatim (feasibility, acquisition
        reference, metrics - none of which touch the model), then draw_samples() in place
        of phases 2 and 3. The elapsed time recorded is therefore the honest cost of this
        arm, which is nearly all metric computation.
        """
        if self._X is None or self._Y_obj is None:
            raise RuntimeError("No data to optimize: set X and Y_obj before calling optimize().")
        t0 = time.monotonic()

        self._compute_feasible_mask(verbose=verbose)
        self._compute_acquisition_function_reference(verbose=verbose)
        self._compute_metrics(verbose=verbose)

        if verbose:
            print("Drawing random candidates... ", end="")
        # draw_samples() rejects against the objective's X constraints, so the baseline
        # searches the same feasible region the optimizer is restricted to.
        self._new_X = self._candidate_sampler.draw_samples(n=self._batch_size)
        if verbose:
            self._print_success(msg=f"New X: {self._new_X.detach().cpu().numpy()}")

        t1 = time.monotonic()
        self._elapsed_time.append(t1 - t0)

        if verbose:
            print(f"Sampling step completed in {t1 - t0:.2f}s")

    # The tutorial loop probes the acquisition function and the posterior at each new
    # point. Neither exists here, and that is the whole point of the arm rather than an
    # error, so these report nothing instead of raising - a tutorial runs either arm
    # without knowing which it holds.

    def compute_acquisition_function_value_at_X(self, X, verbose=True):
        if verbose:
            print("Acquisition function value: none - this run samples at random.")
        return None

    def compute_posterior_mean_at_X(self, X, verbose=True):
        if verbose:
            print("Posterior mean: none - this run fits no model.")
        return None

    def compute_posterior_variance_at_X(self, X, verbose=True):
        if verbose:
            print("Posterior variance: none - this run fits no model.")
        return None
