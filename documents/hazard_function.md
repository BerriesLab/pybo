<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Hazard function

The hazard rate is the instantaneous failure rate, scaled by the faction of surviving systems at time $t$. It can be
easily derived from the Bayes theorem
$$
P(A\vert B) = \frac{P(B \vert A)P(A)}{P(B)}
$$
Assuming that

+ A is the event that a failure occurs *exactly* at time $t$.
+ B is the event that a system survived up to time $t$.

then

+ $P(A\vert B) = h(t)$
+ $P(A) = f(t)$
+ $P(B) = R(t) = 1 - F(t)$
+ $P(B \vert A) = 1$

hence
$$
h(t) = \frac{f(t)}{1-F(t)}
$$ 