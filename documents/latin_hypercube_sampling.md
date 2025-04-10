<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
</style>

# Latin Hypercube Sampling

Latin Hypercube Sampling (LHS) is a statistical method for generating a near-random sample of parameter values from a
multidimensional distribution. It is particularly useful for efficiently exploring high-dimensional spaces.

LHS works by dividing each parameter's range into equally probable intervals (or hypercubes in higher dimensions) and
ensuring that each sample is selected such that no two samples share the same row or column in any dimension. Once the
hypercubes are defined, a sample is drawn randomly from within each selected hypercube.

This ensures that the samples are well-distributed across the parameter space, reducing clustering and improving
coverage compared to simple random sampling. sample. 