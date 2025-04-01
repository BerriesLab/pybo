<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
  ol {line-height: 2;}
</style>

# Bayesian Optimization

Bayesian optimization is a technique to optimize (i.e. either minimize or maximize) an expensive-to-evaluate objective
function $\mathscr{L}(\mathbf{x})$, where $\mathbf{x}$ is a vector of parameters, using a probabilistic model. Unlike
traditional methods, like grid search or random search, which optimize the objective function by exploring the
hyperparameter space in a deterministic or pre-defined way, Bayesian optimization builds a probabilistic model of the
function and uses it to decide where to evaluate next in the hyperparameter space. Bayesian optimization is based
on an iterative application of [Bayes inference](bayesian_inference), and is used in several fields of engineering,
where each experiment is vey time and/or material expensive.

---

### Bayesian Optimization for Gaussian Processes

For Gaussian Processes, the objective function $\mathscr{L}(\mathbf{x})$ is modeled as
a [Gaussian Process](gaussian_process.md). The following inputs are given:

- **Objective function** $\mathscr{L}(\mathbf{x}): \mathbb{R}^N \rightarrow \mathbb{R}^1$: The function to
  optimize.
- **Kernel function** $k(\mathbf{x}, \mathbf{x}^\prime): \mathbb{R}^N \rightarrow \mathbb{R}^1$: The function to
  calculate the covariance matrix $K$.
- **Search space** or **hyperparameter space** $X \in \mathbb{R}^N$: The space of possible inputs (real-valued,
  bounded).
- **Acquisition function** $A(\mathbf{x}): \mathbb{R}^N \rightarrow \mathbb{R}^1$: A function used to decide the next
  point to evaluate (e.g., Expected Improvement, Probability of Improvement, Upper Confidence Bound).
- **Initial observations**: A relatively small set of $n$ initial data
  points
  $$
  D_{0} = \{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \dots, (\mathbf{x}_n, y_n)\} =
  \left\{
  \left( \left[ \begin{matrix} x_{10} \\ x_{11} \\ \cdots \\ x_{1N} \end{matrix} \right], y_1\right)
  \left( \left[\begin{matrix} x_{20} \\ x_{21} \\ \cdots \\ x_{2N} \end{matrix}\right], y_2 \right)
  \cdots,
  \left( \left[\begin{matrix} x_{n0} \\ x_{n1} \\ \cdots \\ x_{nN} \end{matrix}\right], y_n \right)
  \right\}
  $$
  where each $y_i = \mathscr{L}(\mathbf{x}_i)$ is the value of the objective function at point $\mathbf{x}_i$. These
  points can be chosen using Random Sampling, [Latin Hypercube Sampling](latin_hypercube_sampling.md) (LHS), or Grid
  Search.

- **Budget or stopping criteria**: The maximum number of evaluations $n_\mathrm{max}$ or a stopping condition based on
  convergence, such as $\mathscr{L}(\mathbf{x}_\mathrm{next}) - \mathscr{L}(\mathbf{x}) < \epsilon$, where $\epsilon$ is
  a very small value.

The optimization process follows five main steps:

1. **Initialize Gaussian Process (GP)**:  
   Fit a Gaussian Process model to the initial observations $D_0$ to estimate the hyperparameters of the GP model (
   e.g., length scale(s), variance). This is done by first calculating the covariance matrix $K$, and then maximizing
   the (logarithmic) [Maximum Likelihood Estimate](maximum_likelihood_estimation.md) for the model, i.e.:
   $$
   \hat{\mathbf{\theta}} = \arg\max_\theta\log{P(\mathbf{y} \vert X, \theta)}
   $$
   where $\theta$ is the set of parameters to find and
   $$
   \log P(\mathbf{y} \vert X, \theta) = -\frac{1}{2} (\mathbf{y} - \mathbf{m})^T \left(K +
   \sigma_n^2 I \right)^{-1} (\mathbf{y} - \mathbf{m}) - \frac{1}{2} \log |K + \sigma_n^2 I| -
   \frac{n}{2} \log(2\pi)
   $$
2. **Find Next Data Point $\mathbf{x}_*$**:  
   Maximize the acquisition function $A(x)$ over the search space $X$ to find the next point $\mathbf{x}_*$ where to
   evaluate the objective function $\mathscr{L}$:
   $$
   x_{\text{next}} = \arg\max_{x \in X} A(x)
   $$
3. **Evaluate the Objective Function**:  
   Evaluate the objective function at the newly selected point, $y_* = \mathscr{L}(\mathbf{x}_*)$, then add the new
   observation $\left(\mathbf{x}_*, y_* \right)$ to the existing dataset,
   i.e. $D_{t+1}=D_t \cup \left(\mathbf{x}_*, y_* \right)$. Now, the Gaussian Process is described by
   $$
   \begin{bmatrix} y \\ f \left( x_* \right) \end{bmatrix} \sim
   \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} K\left(X, X \right) +\sigma_n^2I & K \left(X, x_* \right)
   \\ K \left(x_*, X \right) & K\left( x_*, x_* \right) \end{bmatrix} \right)
   $$
4. **Update the Model**:  
   After evaluating the objective function at $\mathbf{x}_*$, update the Gaussian Process posterior $P(H \vert E)$ with
   the new data point. This involves calculating the mean function and the covariance function based on the extended
   dataset:
   $$
   m_{t+1}(\mathbf{x}) = m_t(\mathbf{x}) + \mathbf{k}(\mathbf{x}_*,X) \left(K + \sigma_n^2 I \right)^{-1} \mathbf{y}
   $$
   $$
   \sigma^2(\mathbf{x}) = \mathbf{k}(\mathbf{x}_*,X) - \left(K + \sigma_n^2 I \right)^{-1} \mathbf{k}(X, \mathbf{x})
   $$
   where $\mathbf{k}$ is the vector of covariances between the new point $\mathbf{x}_*$ and all previously observed
   points $X$.
5. **Stopping Criteria**:
   Repeat steps 2–4 iteratively until a stopping criterion is met, e.g. when a predefined number of function evaluations
   is reached or when the improvement in the objective function becomes negligible.

---

### Data Points

The experimental data points fora Bayesian optimization are assumed to include noise.
$$
y_i = f\left( x_i \right) + \epsilon_i
$$
where
$$
\epsilon_i \sim \mathcal{N}\left(0, \sigma_n^2 \right)
$$
is the normally distributed noise associated to the *i*-th measurement, having mean $\mu_n = 0$ and
variance $\sigma_n^2$. When $n$ data points are available, the *covariance matrix* $K \in [n\times n]$

---

### The Posterior Function $\mathscr{L}_*$

When adding a new test point $x_\mathrm{next}$, so that $\mathscr{L}\left( x_\mathrm{next} \right)$ is the function
evaluated at the new point, the posterior function, i.e. the new function evaluated after adding the
point $x_\mathrm{next}$, also follows a multivariate gaussian distribution
$$
\begin{bmatrix} y \\ f \left( x_* \right) \end{bmatrix} \sim
\mathcal{N}\left(\mathbf{0}, \begin{bmatrix} K\left(X, X \right) +\sigma_n^2I & K \left(X, x_* \right)
\\ K \left(x_*, X \right) & K\left( x_*, x_* \right) \end{bmatrix} \right)
$$
where $\sigma_n^2$ is due to the Gaussian noise of the experimental data points.

---

### Acquisition Function

The acquisition function uses the posterior mean and posterior variance to identify the next point $x$ to evaluate.
A popular acquisition function is the Expected Improvement ($\mathrm{EI}$) function.
$$
\mathrm{EI}(x) = \left[ f_\mathrm{best} - \mu(x)\right] F\left(\eta \right) + \sigma(x)f(\eta)
$$
where $\eta = \frac{f_\mathrm{best}-\mu(x)}{\sigma(x)}$ is the normalized improvement over the best observed value.
Here, $f_\mathrm{best} - \mu(x)$ is the difference between the current best value and the predicted mean at $x$,
evaluated before adding the new data point $x_\mathrm{next}$ to the dataset $X$.

---

### Posterior Mean $\mu_*(x)$

The posterior mean is the prediction of the value of the objective function at $x$, is given by:
$$
m(\mathbf{x}_*) = \mathbf{k}(\mathbf{x}_*)^T\left[K +\sigma_n^2 I \right]^{-1}\mathbf{y}
$$
where $\mathbf{k}(\mathbf{x}_*)$ is a column vector where each element is the covariance between the new
point $\mathbf{x}_*$ and all points in the dataset $\mathbf{x}_i$.