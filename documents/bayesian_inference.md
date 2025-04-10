<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Bayesian Inference

Bayesian inference is a statistical method that applies [conditional probability](conditional_probability.md) to update
the probability of a hypothesis $H$ as more evidence $E$ (or data) becomes available. Mathematically, it is expressed
as:

$$
P(H \vert E) = \frac{P(E \vert H) P(H)}{P(E)}
$$

where:

+ $P(H \vert E)$: **Posterior Probability**  
  This is the probability of the hypothesis $H$ being true **after** observing the new evidence $E$. It represents
  the updated belief about the hypothesis, taking into account both the prior knowledge and the new evidence.


+ $P(E \vert H)$: **Likelihood**  
  This is the probability of observing the evidence (or data) $E$, given the hypothesis (or model) $H$. It is called the
  likelihood
  term since it tells how likely it is to observe the data after specifying a certain observation model
  for the data.


+ $P(H)$: **Prior Belief**  
  This is the belief about the distribution of the hypothesis $H$ before observing any actual evidence, or data. It
  encodes prior knowledge or assumptions about the hypothesis, which could be based on expert knowledge, previous data,
  or even be uniform (i.e., assuming no preference or knowledge about the hypothesis).


+ $P(E)$: **Evidence**  
  This is the total probability of observing the evidence $E$ under **all** possible hypotheses. It acts as a
  normalizing constant to ensure that the posterior probability $P(H \vert E)$ is properly scaled, i.e., the sum of all
  posterior probabilities over all hypotheses equals 1. It can be calculated as $P(E)=\sum_iP(E\vert H_i)P(H_i)$, or
  more conveniently $P(E)=P(E\vert H)P(H) + P(E\vert H^\mathrm{C})P(H^\mathrm{C})$

More generally, when there is more than one hypothesis, Bayes' theorem can be cast as:
$$
P(H_i \vert E) = \frac{P(E \vert H_i) P(H_i)}{\sum_j P(E\vert H_j)P(H_j)}
$$