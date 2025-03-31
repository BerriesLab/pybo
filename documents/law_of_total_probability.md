<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# The law of total probability

Given the set $\Omega$ into a bunch of disjoint subsets $\{B_i\}$, where $B_i \cap B_j = 0$ if $i \ne j$ and such
that $\Omega = \cup_{i=1}^n B_i$. The law of total probability states that
$$
P(A) = \sum_{i=1}^nP(A \vert B_i) P(B_i)
$$
A useful disjoint set is the complement set of $B$, called $B^\mathrm{C}$ and defined such
that $\Omega = B \cup B^\mathrm{C}$. In this case,
$$
P(A) = P(A \vert B) P(B) + P(A \vert B^\mathrm{C}) P(B^\mathrm{C})
$$