# Readme

+ [Bayes Theorem](documents/conditional_probability.md)
+ [Bayesian Inference](documents/bayesian_inference.md)
+ [Bayesian Optimization](documents/bayesian_optimization.md)


+ [Gaussian Processes](documents/gaussian_process.md)
+ [Maximum Likelihood Estimation](documents/maximum_likelihood_estimation.md)


**Input**: np.ndarray $ n \times (d + 2m + 2c) $ where 
- $n$ is the number of experiments.
- $d$ is the number of parameters.
- $m$ is the number of observables.
- $c$ is the number of constraints.

The matrix is organized as follows:

$$
\begin{bmatrix} X & ¦ & Y^{(o)} & ¦ & Y^{(c)} \end{bmatrix}
$$

where $X \in n \times d$ is the parameter matrix, $Y^{(o)} \in n \times 2m$ is the objective matrix, and $Y^{(c)} \in n \times 2c$ is the constraint matrix. Specifically:

$$
X = \begin{bmatrix} x_{01} & x_{02} & \dots & x_{0d} \\
                    x_{11} & x_{12} & \dots & x_{1d} \\
                    \vdots & \vdots & \ddots & \vdots \\
                    x_{n1} & x_{n2} & \dots & x_{nd} \end{bmatrix}
$$

$$
Y^{(o)} = \begin{bmatrix}
          y^{(o)}_{01} & y^{(o,\sigma)}_{01} & y^{(o)}_{02} & y^{(o, \sigma)}_{02} & \dots & y^{(o)}_{0d} & y^{(o, \sigma)}_{0m}\\
          y^{(o)}_{11} & y^{(o,\sigma)}_{11} & y^{(o)}_{12} & y^{(o, \sigma)}_{12} & \dots & y^{(o)}_{1d} & y^{(o, \sigma)}_{1m}\\ 
          \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
          y^{(o)}_{n1} & y^{(o,\sigma)}_{n1} & y^{(o)}_{n2} & y^{(o, \sigma)}_{n2} & \dots & y^{(o)}_{nd} & y^{(o, \sigma)}_{0m}\\ 
          \end{bmatrix}
$$
$$
Y^{(c)} = \begin{bmatrix} 
          y^{(c)}_{01} & y^{(c,\sigma)}_{01} & y^{(c)}_{02} & y^{(c, \sigma)}_{02} & \dots & y^{(c)}_{0d} & y^{(c, \sigma)}_{0m}\\
          y^{(c)}_{11} & y^{(c,\sigma)}_{11} & y^{(c)}_{12} & y^{(c, \sigma)}_{12} & \dots & y^{(c)}_{1d} & y^{(c, \sigma)}_{1m}\\ 
          \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
          y^{(c)}_{n1} & y^{(c,\sigma)}_{n1} & y^{(c)}_{n2} & y^{(c, \sigma)}_{n2} & \dots & y^{(c)}_{nd} & y^{(c, \sigma)}_{0m}\\ 
          \end{bmatrix}
$$