

Assume that the value $y$ resulting from an observation at some point $x$ is 
distributed according to an observation model depending on the underlying 
objective function value $\phi = f(x)$:
$$
p\left(y \vert x, \phi \right)
$$

Here we model the value $y$ observed at $x$ as 
$$
y = \phi + \epsilon
$$
where $\epsilon$ represents measurement error. Errors are assumed to be 
Gaussian distributed with mean zero. This implies a Gaussian observation model
$$
p\left(y \vert x, \phi, \sigma_n \right) = \mathcal{N}\left(y; \phi, 
\sigma_n^2 \right)
$$
where $\sigma_n$ is the observation noise scale. 