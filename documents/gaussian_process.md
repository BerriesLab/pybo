<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Gaussian Process

A Gaussian Process (GP) is a stochastic process, i.e. a collection of infinitely many random variables, each associated
with a function value at a particular input point vector, the latter typically being a space or time coordinate.
Specifically, given the inputs $X=\{\mathbf{x}_i\}$, where $\mathbf{x}_i \in \mathbb{R}^N$ is the *i*-th input data
vector, the collection of random variables describing the process is $\mathbf{f} = \{f(\mathbf{x}_i)\}$. Every finite
subset of the function values $\{ f(\mathbf{x}_i), f(\mathbf{x}_{i+1}), \dots, f(\mathbf{x}_n) \}$ follows a
multivariate normal distribution,
i.e.
$$
\mathbf{f} = \begin{bmatrix} f(\mathbf{x}_1) \\ f(\mathbf{x}_2) \\ \cdots \\ f(\mathbf{x}_n) \end{bmatrix} \sim \mathcal{N}(\mathbf{m}, K)
$$
where $\mathbf{m}$ is the mean function vector and $K$ is the covariance matrix. In
other words, the GP defines a joint probability distribution over the entire function. In formula, A Gaussian Process is
usually denotes as:
$$
f(\mathbf{x}) \sim \mathrm{GP}\left( \mathbf{m}(\mathbf{x}), \, k(\mathbf{x}, \mathbf{x}^\prime) \right)
$$
where $k(\mathbf{x}, \mathbf{x}^\prime)$ is the kernel function used to calculate the covariance matrix. A commonly used
kernel in
Bayesian optimization is the Squared Exponential
$$
k(\mathbf{x}, \mathbf{x}^\prime) = \sigma^2 \exp{\left( -\frac{1}{2} \sum_{i=1}^N \frac{\left( x_i - x_i^\prime \right)^2}{l_i^2}\right)}
$$
where $\sigma^2$ is the variance of the function, and $l_i$ is the length scale for dimension $j$, which determines the
correlation distance. The kernel function is used to calculate the covariance matrix as follows
$$
K =
\begin{bmatrix}
k(\mathbf{x}_1, \mathbf{x}_1) & k(\mathbf{x}_1, \mathbf{x}_2) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\
k(\mathbf{x}_2, \mathbf{x}_1) & k(\mathbf{x}_2, \mathbf{x}_2) & \cdots & k(\mathbf{x}_2, \mathbf{x}_n) \\
\vdots & \vdots & \ddots & \vdots \\
k(\mathbf{x}_n, \mathbf{x}_1) & k(\mathbf{x}_n, \mathbf{x}_2) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n)
\end{bmatrix}
$$
where $k(\mathbf{x}_i, \mathbf{x}_i)=\sigma_n^2$ is the variance of the input.