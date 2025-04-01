<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Joint Distributions

Assume that we have two independent univariate Gaussian variables $x_1=\mathcal{N}\left(\mu_1, \sigma^2_1 \right)$
and $x_2=\mathcal{N}\left(\mu_2, \sigma^2_2 \right)$. Their joint distribution is $P\left(x_1, x_2\right)$ is:
$$\begin{align}
P\left(x_1, x_2\right) &= P\left(x_1\right) P\left(x_2\right) \\ &=
\frac{1}{\sqrt{2 \pi \sigma_1^2}} \exp\left({-\frac{\left(x_1 - \mu_1\right)^2}{2\sigma_1^2}}\right)
\frac{1}{\sqrt{2 \pi \sigma_2^2}} \exp\left({-\frac{\left(x_2 - \mu_2\right)^2}{2\sigma_2^2}}\right) \\ &=
\frac{1}{2 \pi \sigma_1 \sigma_2} \exp\left({-\frac{1}{2} \left[ \left(x_1 - \mu_1 \right)^T (\sigma_1^2)^{-1} \left(x_1 - \mu_1 \right) + \left(x_2 - \mu_2 \right)^T (\sigma_2^2)^{-1} \left(x_2 - \mu_2 \right)\right]}\right) \\ &=
\frac{1}{2 \pi \sigma_1 \sigma_2} \exp\left({-\frac{1}{2} \left( \mathbf{x} - \mathbf{\mu} \right)^\mathrm{T} \Sigma^{-1} \left(\mathbf{x} - \mathbf{\mu} \right)}\right)
\end{align}$$

where $\Sigma = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2\end{bmatrix}$. In general, for a multivariate normal
distribution:

$$
P(x_1)P(x_2)\cdot P(x_N)=
\frac{1}{\left(2 \pi\right)^{N/2} \det{\Sigma}}
\exp\left({-\frac{1}{2} \left( \mathbf{x} - \mathbf{\mu} \right)^\mathrm{T} \Sigma^{-1} \left(\mathbf{x} - \mathbf{\mu} \right)}\right)
$$