<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Bayesian Inference

Bayesian inference is a statistical method that applies [Bayes' Theorem](bayes_theorem.md) to update the probability of
a hypothesis $H$ as more evidence $E$ (or data) becomes available. Mathematically, it is expressed as:
$$
P(H \vert E) = \frac{P(E \vert H) P(H)}{P(E)}
$$
where:

+ $P(H \vert E)$: *Posterior Probability*  
  This is the probability of the hypothesis $H$ being true **after** observing the new evidence $E$. It represents
  the updated belief about the hypothesis, taking into account both the prior knowledge and the new evidence.


+ $P(E \vert H)$: *Likelihood*  
  This is the probability of observing the evidence $E$, given that the hypothesis $H$ is true. It measures how
  well the hypothesis explains the observed evidence. This is the model's likelihood function.


+ $P(H)$: *Prior Probability*  
  This is the initial belief or probability of the hypothesis $H$ before any evidence is observed. It reflects prior
  knowledge or assumptions about the hypothesis, which could be based on expert knowledge, previous data, or even be
  uniform (i.e., assuming no preference or knowledge about the hypothesis).


+ $P(E)$: *Evidence Probability*  
  This is the total probability of observing the evidence $E$ under **all** possible hypotheses. It acts as a
  normalizing
  constant to ensure that the posterior probability $P(H \vert E)$ is properly scaled, i.e., the sum of all posterior
  probabilities over all hypotheses equals 1. It can be calculated as $P(E)=\sum_iP(E\vert H_i)P(H_i)$



