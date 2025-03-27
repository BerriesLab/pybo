<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Maximum Likelihood Estimation (MLE)

It is a statistical method used to estimate the parameters of a model by maximizing the likelihood of observing the
given data under the model.

In the context of Bayesian optimization and GPs: The likelihood is the probability of observing the data given the
model's parameters (the hyperparameters of the kernel). The goal is to find the values of the kernel’s hyperparameters
that maximize the likelihood of the observed data under the model.

Mathematically, the likelihood $P(\mathbf{y} \vert \mathbf{X}, \theta)$ is the probability of the observed
outputs $\mathbf{y}$ given the inputs $\mathbf{X}$ and the kernel hyperparameters $\theta$. The log-likelihood is often
used because it’s easier to work with, and it transforms the product of probabilities into a sum.

The process of fitting a GP involves:

+ Choosing a kernel.
+ Optimizing the kernel’s hyperparameters using MLE.

The optimization process typically involves computing the log-likelihood of the data given the kernel hyperparameters
and then adjusting the hyperparameters to maximize this log-likelihood. This is done using optimization techniques such
as gradient descent or other optimization algorithms.

# To fix

### 1. **General Case with Non-Zero Mean Function**

A Gaussian Process (GP) with a **non-zero mean function** is modeled as:
$$
f(x) \sim \mathcal{N}\left(m(x), K(x, x')\right)
$$
where:

- $m(x)$ is the **mean function**.
- $K(x, x')$ is the **covariance function** (or kernel).

The function values at observed inputs $\mathbf{X} = \{x_1, x_2, \dots, x_n\}$ follow a **multivariate normal
distribution** with:

- Mean vector $\mathbf{m} = [m(x_1), m(x_2), \dots, m(x_n)]$.
- Covariance matrix $K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I$, where $\sigma_n^2$ is the **noise variance** and $I$ is
  the **identity matrix**.

### 2. **Likelihood Function with Non-Zero Mean**

The **likelihood** of the observed data $\mathbf{y} = \{y_1, y_2, \dots, y_n\}$, given the inputs $\mathbf{X}$ and
kernel hyperparameters $\theta$, is given by the following **multivariate normal distribution**:
$$
P(\mathbf{y} \vert \mathbf{X}, \theta) = \frac{1}{(2\pi)^{n/2} |K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I|^{1/2}} \exp
\left( -\frac{1}{2} (\mathbf{y} - \mathbf{m})^T \left(K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I \right)^{-1} (
\mathbf{y} - \mathbf{m}) \right)
$$
Where:

- $\mathbf{y}$ is the vector of observed outputs (function values).
- $\mathbf{m}$ is the vector of the mean function evaluated at the observed
  inputs: $\mathbf{m} = [m(x_1), m(x_2), \dots, m(x_n)]$.
- $K(\mathbf{X}, \mathbf{X})$ is the covariance matrix of the inputs, computed using the kernel function.
- $\sigma_n^2$ is the **noise variance** (which accounts for measurement or observational noise).
- $I$ is the **identity matrix**.

### 3. **Log-Likelihood Function with Non-Zero Mean**

The **log-likelihood** function is derived from the likelihood and is given by:
$$
\log P(\mathbf{y} \vert \mathbf{X}, \theta) = -\frac{1}{2} (\mathbf{y} - \mathbf{m})^T \left(K(\mathbf{X}, \mathbf{X}) +
\sigma_n^2 I \right)^{-1} (\mathbf{y} - \mathbf{m}) - \frac{1}{2} \log |K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I| -
\frac{n}{2} \log(2\pi)
$$