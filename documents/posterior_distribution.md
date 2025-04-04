# Posterior Distribution

The posterior predictive distribution for a new data point $y^\prime$ after observing a collection of data
points collectively denoted as $\mathcal{D}$. We would like to assess how the future data would be distributed and what
value of $y^\prime$
we would likely observe if we were to run the experiment and acquire another data point again, given that we have
observed some actual data. That is, we want to calculate the posterior predictive
distribution $p(y^\prime \vert \mathcal{D})$. We can calculate the posterior predictive distribution by treating it as a
marginal distribution (conditioned on the collected dataset $\mathcal{D}$) and applying the same technique as before,
namely
$$
p(y' \vert \mathcal{D}) = \int{p(y', \theta \vert \mathcal{D}) d\theta} =
\int{p(y'\vert \theta, \mathcal{D} ) p(\theta \vert \mathcal{D})d\theta}
$$
where we have used the product rule of probability $p(y', \theta) = p(y' \cap \theta) = p(y' \vert \theta) p(\theta)$,
and where the second term $p(\theta \vert \mathcal{D})$ is the posterior distribution of the parameter $\theta$ that can
be calculated by applying Bayes’ rule.

Unfortunately, the term $p(y'\vert \theta, \mathcal{D})$ is involved and hard to calculate. When assessing a new data
point after observing some existing data points, a common assumption is that they are **conditionally independent**
given a particular value of $\theta$. Such conditional independence implies that $p(y'\vert \theta, \mathcal{D}) = p(
y'\vert \theta)$ ,
which happensto be the likelihood term. Thus, we can simplify the posterior predictive distribution asfollows:
$$
p(y' \vert \mathcal{D}) = \int{p(y'\vert \theta) p(\theta \vert \mathcal{D})d\theta}
$$
where $p(y' \vert \theta)$ is the likelihood of observing the data under a specific parameter $\theta$,
and $p(\theta \vert \mathcal{D})$ is the parameter's posterior distribution. This follows the same pattern of
calculation compared to the prior predictive distribution, just now the parameter $\theta$ is conditioned on the
dataset $\mathcal{D}$.

---

**Example: The posterior predictive distribution under a normal posterior and normal likelihood**  
Similarly to the example for the prior [predictive distribution](prior_distribution.md), under the assumption of normal
likelihood $p(y' \vert \mathcal{D}) = \mathcal{N}(\theta, \sigma^2)$ and normal
posterior $p(\theta \vert \mathcal{D}) = \mathcal{N}(\theta', \sigma_\theta^{'2})$, the predictive posterior
distribution is also normal:
$$y' = \mathcal{N}(\theta', \sigma^2+\sigma_\theta^{'2})$$