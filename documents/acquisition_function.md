# Acquisition Function

The acquisition function is a function used to identify the next location in the input domain where to evaluate an
expensive-to-evaluate objective function. It is based on the utility concept, and specifically on the difference between
the utility of the dataset before selecting the new location $u(D_n)$, and the utility of the dataset including the
newly selected location $u(D_{n+1})$. The mathematical form of the acquisition function depends on the choice of
utility.

### Expected Improvement

The utility is the maximum observed value, that is $u(D_n)=\max{(f_{1:n})}=f_n^*$
and $u(D_{n+1})=u(D_n \cup \{x_{n+1}, f_{n+1}\})=\max{(f_{n+1}, f_n^*)}$. Taking the difference between the two gives
the increase utility due to the addition of another observation:
$$u(D_{n+1})-u(D_n)=\max{(f_{n+1}, f_n^*)} - f_n^* = \max{(f_{n+1} - f_n^*, 0)}$$
which return the marginal increment if $f_{n+1} > f_n^*$ and zero otherwise. Due to the randomness in the
observation $y_{n+1}$, it is convenient to evaluate the expected marginal gain to integrate it out:
$$\mathbb{E}\left[u\left(D_{n+1}\right)-u\left(D_n\right) \, \vert \, x_{n+1}, D_n\right] =
\int{ \max{(f_{n+1} - f_n^*, 0)}p\left(f_{n+1} \, \vert \, x_{n+1}, D_n \right) \, df_{n+1}}$$
Under the framework of Gaussian Processes, it is possible to obtain a closed-form expression of the expected
improvement acquisition function. The key is to re-parameterize $f_{n+1}$. In fact, since the observation $f_{n+1}$ at
the candidate location $x_{n+1}$ follows a normal distribution with mean $\mu_{n+1}$ and $\sigma^2_{n+1}$, that
is $f_{n+1} \sim \mathcal{N}(\mu_{n+1}, \sigma^2_{n+1})$, $f_{n+1}$ can be re-parameterized
as $f_{n+1} = \mu_{n+1} + \sigma_{n+1}\epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$ is normally distributed.

$$
\begin{align}
\mathbb{EI} &= \int_{-\infty}^{\infty}\max{ \left\{ f_{n+1} - f_n^*, 0 \right\} } p\left(f_{n+1} \, \vert \, x_{n+1}, D_n \right) \, df_{n+1} \\
&= \int_{-f^*_n}^{\infty} \left( f_{n+1} - f_n^* \right) \mathcal{N}\left(f_{n+1} \, \vert \, \mu_{n+1}, \sigma_{n+1}^2 \right) \, df_{n+1} \\
\end{align}
$$

Plugging in the expression for the Normal distribution, it can be shown that:

$$
\mathbb{EI}(x_{n+1}) =\left( \mu - f^* \right) \Phi(Z) + \sigma \phi(Z)
$$

where $Z=(\mu - f^*)/\sigma$, $\phi$ is the probability density distribution, and $\Phi$ is the cumulative probability
distribution. 