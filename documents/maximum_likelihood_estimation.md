<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Maximum Likelihood Estimation

Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a model by maximizing the
likelihood of observing the given data under the model. In other words, MLE identifies the parameters of a parametric
distribution that are most likely to have produced the observed data. The optimization process typically involves
computing the log-likelihood of the data for numerical convenience. The optimization algorithm is based on common
numerical methods, such as Gradient Descent.

### Example 1: Gaussian Model

Given a set of observations $X=\{x_1, x_2, \dots,x_n\} \in \mathcal{R}^1$, and assuming that the observations follows a
Gaussian distribution, the likelihood function for the Gaussian distribution is the product of the individual
probabilities:
$$
P(X \vert \mu, \sigma) = \prod_i\frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left({-\frac{\left( x_i - \mu\right)^2}{2 \sigma^2}}\right)
$$
where $\sigma$ and $\mu$ are the parameters to optimize. The corresponding log-likelihood function is:
$$
\log P(X \vert \mu, \sigma) = - \frac{n}{2}\log \left( 2 \pi \sigma^2 \right) - \frac{1}{2 \sigma^2} \sum_{i=1}^n \left(x_i - \mu\right)^2
$$
To maximize the log-likelihood, one calculates the derivative of $\log P(X \vert \mu, \sigma)$ with respect
to $\mu$ and $\sigma$ and set them to zero. This yields the following results for the Maximum Likelihood Estimate:

+ The MLE for $\mu$ is the sample mean:$$\hat{\mu}=\frac{1}{n}\sum_i^n x_i$$
+ The MLE for $\sigma^2$ is the sample variance: $$\hat{\sigma}^2=\frac{1}{n}\sum_i^n (x_i -\mu)^2$$

Note: for more complicated likelihood function, it is not possible to find an analytical solution to the maximization
problem and numerical methods must be employed.

### Example 2. Multivariate Gaussian Model

Given a set of
observations $\{\mathbf{x}_1, \mathbf{x}_2, \dots \mathbf{x}_n \} \in \mathcal{R}^N$, and
assuming that the observations follow a multivariate Gaussian distribution, the likelihood function is the product of
the individual multivariate probabilities:
$$
P(X \vert \mu, K) = \prod_{i=1}^n \frac{1}{(2 \pi)^{\frac{N}{2}} \sqrt{\det(K)}}
\exp\left({-\frac{1}{2} \left( \mathbf{x}_i - \mathbf{\mu} \right)^\mathrm{T}} K^{-1} \left( \mathbf{x}_i - \mathbf{\mu} \right) \right)
$$
where $\mu \in \mathrm{R}^N$ is the mean vector, and $K \in \mathrm{R}^{N \times N}$ is the covariance matrix. The
corresponding log-likelihood function is
$$
\log P(X \vert \mu, K) = -\frac{1}{2} \sum_{i=1}^n (\mathbf{x}_i - \mathbf{\mu})^\mathrm{T} K^{-1} (\mathbf{x}_i - \mathbf{\mu}) -
\frac{n}{2} \log \det{K} - \frac{nN}{2} \log(2\pi)
$$
To maximize the log-likelihood, one calculates the derivative of $\log P(X \vert \mu, K)$ with respect
to $\mu_{i}$ and $K_{ij}$ and set them to zero. This yields the following results for the Maximum Likelihood Estimate:

+ The MLE for $\mu$ is the sample mean:$$\hat{\mu}=\frac{1}{n}\sum_i^n \mathbf{x}_i$$
+ The MLE for $K$ is the sample covariance
  matrix: $$\hat{K}=\frac{1}{n}\sum_i^n (\mathbf{x}_i -\hat{\mu})(\mathbf{x}_i -\hat{\mu})^\mathrm{T}$$

### Example 3: Gaussian Process

Assuming a set of input data points $X=\{x_1, x_2, \dots, x_n\}$ and the corresponding set of output
values $\mathbf{y}=\{y_1, y_2, \dots, y_n\}$, the likelihood function for a Gaussian
Process (GP) is derived from the fact that the joint distribution of the observations, conditioned on the inputs, is a
multivariate normal distribution
$$
P(\mathbf{y} \vert X, \theta) \sim \mathcal{N}(\mathbf{y} \vert \mathbf{m}, K + \sigma_n^2 I)
$$
where $\mathbf{m}(X)$ is the *mean function*, $K$ is the covariance matrix of the available data points, calculated
using the kernel function $k(x, x^\prime)$, and $\sigma_n^2$ is the noise variance, used to model the noise in the
observations. Therefore, the likelihood function is
$$
P(\mathbf{y} \vert X, \theta) = \frac{1}{(2\pi)^{n/2} \sqrt{\det{\left(K + \sigma_n^2 I\right)}}}
\exp \left( -\frac{1}{2} (\mathbf{y} - \mathbf{m})^\mathrm{T} \left(K + \sigma_n^2 I \right)^{-1} (\mathbf{y} - \mathbf{m}) \right)
$$
Note that this function differs from the multivariate normal distribution, where the probabilities of single data points
are multiplied. This is because in a Gaussian Process, we assume that function values at different points are correlated
through the covariance matrix $K$. Unlike a standard multivariate normal distribution where each observation is treated
as an independent sample, here the structure of $K$ K captures the relationships between points based on the chosen
kernel function $k(x, x^\prime)$. The presence of $K + \sigma_n^2 I$ in the likelihood function ensures that the model
accounts for both intrinsic function smoothness (via the kernel) and observational noise (via $\sigma_n^2 I$). This
means that the likelihood function does not factorize into independent terms for each data point but rather
treats the observations as coming from a single coherent probabilistic model. The **log-likelihood** function is derived
from the likelihood and is given by:
$$
\log P(\mathbf{y} \vert \mathbf{X}, \theta) = -\frac{1}{2} (\mathbf{y} - \mathbf{m})^T \left(K(\mathbf{X}, \mathbf{X}) +
\sigma_n^2 I \right)^{-1} (\mathbf{y} - \mathbf{m}) - \frac{1}{2} \log |K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I| -
\frac{n}{2} \log(2\pi)
$$
Then, the MLE estimates for the parameters $\mathbf{\theta}$ is:
$$
\hat{\mathbf{\theta}} = \arg\max{P(\mathbf{y} \vert X, \theta)}
$$