# Gaussian Processes

A Gaussian Process (GP) is a collection of infinite random variables, any finite subset of which follows a multivariate
gaussian distribution. It is defined by a mean function $\mu(\mathbf{x})$ and a covariance function or
kernel $k(x, x^\prime)$, which encode the belief about the function's smoothness and structure. In formula:
$$
f(x) \sim \mathrm{GP}\left( \mu(x), \, k(x, x^\prime) \right)
$$
An informal way fo interpreting GP is to think that each function value $f(x)$ is normally distributed. A commonly used
kernel in Bayesian optimization is the Squared Exponential
$$
k(x, x^\prime) = \sigma^2 \exp{\left( -\frac{\left( x-x^\prime \right)^2}{2l^2}\right)}
$$
where $\sigma^2$ is the variance of the function, and $l$ is the length scale, which determines the correlation
distance.