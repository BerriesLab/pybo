# Prior Predictive Distribution

Before collecting new data, it is necessary to work with a prior predictive distribution that considers
all possible values of the underlying parameter $\theta$. That is, the prior predictive distribution $p(y)$ for the
variable $y$ is a marginal probability distribution that could be calculated by integrating out all dependencies on the
parameter $\theta$:
$$
p(y) = \int{p(y, \theta)d\theta} = \int{p(y \vert \theta)p(\theta)d\theta}
$$
where $p(y \vert \theta)$ is the likelihood of observing the data under a specific parameter $\theta$, and $p(\theta)$
is the prior distribution of the parameter. This is the exact
definition of the evidence term in Bayes’ formula.

---

**Example: The prior predictive distribution under a normal prior and normal likelihood**  
Before the experiment starts, we assume the observation model for the likelihood of the data $y$ to follow a normal
distribution, that is $y \sim \mathcal{N}(\theta, \sigma^2)$,
or $p(y \vert \theta, \sigma^2) = \mathcal{N}(\theta, \sigma^2)$, where $\theta$ is the underlying parameter
and $\sigma^2$ is a fixed variance. For example, $\theta$ could be an objective function and $\sigma^2$ additive
Gaussian noise. However, the distribution of $y$ depends on $\theta$, which is also an unknown or uncertain quantity. We
then assume that also the parameter $\theta$ follows a normal distribution, that
is $\theta \sim \mathcal{N}\left(\theta_0, \sigma_\theta^2 \right)$
or $p(\theta) = \mathcal{N}\left(\theta_0, \sigma_\theta^2\right)$, where $\theta_0$ and $\sigma_0^2$ are the mean and
variance of the prior normal distribution assumed before collecting any other data. Since we have no knowledge of the
environment of interest,we would like to understand how the data point (treated as a random variable) $y$ could be
distributed in this unknown environment under different values of $\theta$. This consists in calculating the prior
predictive distribution $p(y)$, that is:
$$p(y) = \int_{-\infty}^{+\infty}{p(y, \theta)d\theta} = \int_{-\infty}^{+\infty}{p(y \vert \theta)p(\theta)d\theta}$$
Since both the likelihood $p(y \vert \theta)$ and the parameter's prior $p(\theta)$ are Gaussian, then also the
predictive prior distribution $p(y)$ must follow a Gaussian distribution. Specifically
$$p(y) = \mathcal{N}\left(\theta_0, \sigma^2 + \sigma_0^2\right)$$