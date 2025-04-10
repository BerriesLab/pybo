<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Joint, Conditional and Marginal Probability

Given two random variables $x$ and $y$, and two events $x = X$ and $y = Y$, where $X$ and $Y$ are specific values
that $x$ and $y$ may assume, respectively. Also, assume that $x$ and $y$ are dependent in some way.

The **joint probability** of two events refers to the probability of them occurring simultaneously. Typically, the joint
probability is
$$
p(X \, \mathrm{and } \, Y)=p(x=X \cap \ y=Y)=p(Y \cap X)
$$
The joint probability becomes the **joint probability distribution** when the single events $X$ and $Y$ are replaced by
the corresponding random variables $x$ and $y$, as now the probability represents all possible combinations of the two
simultaneous events.

Using the chain rule, the joint probability of two events $X$ abd $Y$ can be written as a **conditional probability**.
In fact
$$
p(X \cap Y) = p(X \vert Y)p(Y)
$$
Since the joint probability is symmetrical, i.e. $p(X \cap Y) = p(Y \cap X)$, it is not difficult to show the Bayesian
formula for two events
$$
P(X \vert Y) = \frac{p(Y \vert X)p(X)}{p(Y)}
$$
By replacing a single event $x=X$ with the random variable $x$, one gets the corresponding conditional probability
distribution $p(x \vert y=Y)$ for the variable $x$ given $y=Y$.

The probability of the event $x=X$ under all possible values of $y$ is called **marginal probability** of the
event $x=X$. The **marginal probability distribution** for a (continuous) random variable $x$ in the presence of
another (continuous) random variable $y$ can be calculated as follows:
$$
p(x) = \int p(x \vert y) p(y) dy
$$

A special case that would impact the calculation of the three probabilities is **independence**. When the event $x=X$ in
independent of the event $y=Y$, then $P(X \vert Y) = p(X)$. Consequently, their joint probability (distributions) become
the product of the individual probability (distributions), while the marginal probability of $x$ reduces to its own
probability distribution.  