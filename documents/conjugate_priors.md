<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Conjugate Priors

When the posterior has the same functional form of the prior, the prior and the posterior are conjugated.

$$
P(\theta \vert \mathrm{data},m) = \frac{P(\mathrm{data} \vert \theta,m) \times P(\theta \vert m)}{P(\mathrm{data} \vert m)}
$$

Example: If the likelihood function $P(\mathrm{data} \vert \theta,m) \sim \mathcal{N}$ is Normal, we can choose the
prior $P(\theta \vert m) \sim \mathcal{N}$, so that the posterior $P(\theta \vert \mathrm{data},m) \sim \mathcal{N}$ is
also normal.
