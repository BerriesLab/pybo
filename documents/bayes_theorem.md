<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Bayes Theorem

Given two events $A$ and $B$, and given the probability $P(A)$ and $P(B)$ that the event $A$ and $B$ occur,
respectively, Bayes theorem states that
$$
P(A \cap B) = P(A \vert B)P(B) = P(B \vert A)P(A)
$$
where $P(A \cap B)$ is the probability that both events $A$ and $B$ occur, $P(A \vert B)$ is the conditional probability
that event $A$ occurs given knowledge that the event $B$ already occurred, and $P(B \vert A)$ is the conditional
probability that event $B$ occurs given knowledge that the event $A$ already occurred. Rearranging:
$$
P(A \vert B) = \frac{P(B \vert A) P(A)}{P(B)}
$$