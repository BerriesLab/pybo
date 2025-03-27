<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Bayesian Optimization

Bayesian optimization is a technique to optimize (i.e. either minimize or maximize) an expensive-to-evaluate objective
function $\mathscr{L}(\mathbf{x})$, where $\mathbf{x}$ is a vector of parameters, using a probabilistic model. Unlike
traditional methods, like grid search or random search, which optimize the objective function by exploring the
hyperparameter space in a deterministic or pre-defined way, Bayesian optimization builds a probabilistic model of the
function and uses it to decide where to evaluate next in the hyperparameter space. Bayesian optimization is based
on [Bayes inference](bayesian_inference.md).

### Bayesian Optimization for Gaussian Processes

The objective function $\mathscr{L}(\mathbf{x})$ is modeled as a Gaussian Process, i.e. a distribution over the possible
functions, where each point of the function is normally distributed.

**Inputs:**

1. **Objective function** $f(x)$: The function you wish to optimize.
2. **Search space** $X$: The space of possible inputs (e.g., real-valued, bounded).
3. **Acquisition function** $A(x)$: A function used to decide the next point to evaluate (e.g., Expected Improvement,
   Probability of Improvement, Upper Confidence Bound).
4. **Initial observations**: A small set of initial data points $D = \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}$,
   where each $y_i = f(x_i)$ is the objective value at point $x_i$.
5. **Budget or stopping criteria**: The maximum number of evaluations or a stopping condition based on convergence.

**Steps:**

1. **Initialize Gaussian Process (GP)**:
    - Fit a Gaussian Process model to the initial observations.
    - Use the initial data points to estimate the hyperparameters of the GP model (e.g., length scale, variance).


2. **Define the Acquisition Function**:
    - Choose an acquisition function \(A(x)\) (e.g., Expected Improvement (EI), Probability of Improvement (PI), Upper
      Confidence Bound (UCB)).
    - The acquisition function quantifies the trade-off between exploring uncertain areas (high model uncertainty) and
      exploiting areas where the objective function is likely to be optimal.


3. **Iterative Process** (Repeat until stopping criteria met):

   a. **Optimize the Acquisition Function**:
    - For each iteration, maximize the acquisition function \(A(x)\) over the search space \(X\) to find the next point
      to evaluate:
      \[
      x_{\text{next}} = \arg\max_{x \in X} A(x)
      \]
    - This gives the next candidate input point \(x_{\text{next}}\) to evaluate.

   b. **Evaluate the Objective Function**:
    - Evaluate the objective function \(f(x_{\text{next}})\) at the newly selected point.
    - Observe the function value \(y_{\text{next}} = f(x_{\text{next}})\).

   c. **Update the Model**:
    - Add the new data point \((x_{\text{next}}, y_{\text{next}})\) to the existing dataset \(D\).
    - Update the GP model with the new data. This involves recalculating the posterior and possibly re-optimizing the GP
      hyperparameters.

4. **Stopping Criteria**:
    - Stop the process when a predefined number of function evaluations is reached or when the improvement in the
      objective function becomes negligible.

---

## **Outputs:**

- The point \(x_{\text{best}}\) in the search space that maximizes the acquisition function.
- The corresponding objective value \(f(x_{\text{best}})\) as the best found solution.

It is given an initial dataset $X = \left\{(x_i, y_i) \right\}$, where $x_i$ are the inputs (i.e., parameters to
optimizes)
and $y_i=\mathscr{L}(x_i)$ are the corresponding outputs (i.e., the function values), a Gaussian Process is used
to update the belief, or *prior probability* $P(H)$, about the function $f(x)$.
In this context, the evidence $E$ is the set of observed data points, the likelihood $P(E\vert H)$ is the probability of
observing the current data given the model parameters, the prior probability $P(H)$ is the initial belief about the
function before observing the data, and the evidence probability $P(E)=\int P(E\vert\theta)P(\theta)d\theta$ must be
computed in order to update the model with new data points.

1. **Define the Objective Function**:  
   Let $\mathscr{L}(\mathbf{x}): \mathcal{X} \rightarrow \mathbb{R}$ represent the objective function,
   where $\mathbf{x}$ is the vector of input variables. Assume a Gaussian Process prior for $\mathscr{L}(\mathbf{x})$
   with a mean function $\mu(\mathbf{x}) = 0$ and kernel $k(x, x')$


2. **Initial Evaluation**:  
   Evaluate $\mathscr{L}(\mathbf{x})$ at $X$,
   obtaining $\mathbf{y} = \begin{bmatrix} y_1 & y_2 & \dots & y_n \end{bmatrix}$.


3. **Fit the Gaussian Process**:  
   Calculate the posterior mean $\mu_*(x)$ and variance $\sigma_*^2(x)$.


4. **Optimize the Acquisition Function**:  
   Maximize the acquisition function (e.g., **Expected Improvement**) to determine the next point $x_{\text{next}}$
   to evaluate.


5. **Evaluate the Objective Function**:  
   Evaluate $\mathscr{L}(x_{\text{next}})$ to obtain $y_{\text{next}}$.


6. **Update the Gaussian Process**:  
   Update the posterior distribution with the new data point $(x_{\text{next}}, y_{\text{next}})$.


7. **Repeat**:  
   Iterate steps 3-6 until a stopping criterion is met.

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

### The Posterior Function $\mathscr{L}_*$

When adding a new test point $x_\mathrm{next}$, so that $\mathscr{L}\left( x_\mathrm{next} \right)$ is the function
evaluated at the new point, the posterior function, i.e. the new function evaluated after adding the
point $x_\mathrm{next}$, also follows a multivariate gaussian distribution
$$
\begin{bmatrix} y \\ f \left( x_* \right) \end{bmatrix} \sim
\mathcal{N}\left(0, \begin{bmatrix} K\left(X, X \right) +\sigma_n^2I & K \left(X, x_* \right)
\\ K \left(x_*, X \right) & K\left( x_*, x_* \right) \end{bmatrix} \right)
$$
where $\sigma_n^2$ is due to the Gaussian noise of the experimental data points.

### Acquisition Function

The acquisition function uses the posterior mean and posterior variance to identify the next point $x$ to evaluate.
A popular acquisition function is the Expected Improvement ($\mathrm{EI}$) function.
$$
\mathrm{EI}(x) = \left[ f_\mathrm{best} - \mu(x)\right] F\left(\eta \right) + \sigma(x)f(\eta)
$$
where $\eta = \frac{f_\mathrm{best}-\mu(x)}{\sigma(x)}$ is the normalized improvement over the best observed value.
Here, $f_\mathrm{best} - \mu(x)$ is the difference between the current best value and the predicted mean at $x$,
evaluated before adding the new data point $x_\mathrm{next}$ to the dataset $X$.

### Posterior Mean $\mu_*(x)$

The posterior mean is the prediction of the value of the objective function at $x$, is given by:
$$
\mu_*(x) = K(x,X)\left[K(X, X) +\sigma_n^2 I \right]^{-1}\mathbf{y}
$$