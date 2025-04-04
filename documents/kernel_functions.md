<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Kernel Functions

### Exponential

The covariance between two points decreases as their distance increases and approaches zero when the distance exceeds
two.
$$
k\left(x_i, x_j \right) = \exp{\left(- \vert x_i - x_j \vert \right)}
$$

### Gaussian or RBF

An exponential kernel with two additional parameters: the length $l$, which controls the smoothness of the function, and
the variance $\sigma_f^2$, which controls the vertical variation. In formula:
$$
k\left(x_i, x_j \right) = \sigma_f^2 \exp{\left(- \frac{\vert x_i - x_j \vert}{2 l^2} \right)}
$$