# Readme

+ [Bayes Theorem](documents/conditional_probability.md)
+ [Bayesian Inference](documents/bayesian_inference.md)
+ [Bayesian Optimization](documents/bayesian_optimization.md)


+ [Gaussian Processes](documents/gaussian_process.md)
+ [Maximum Likelihood Estimation](documents/maximum_likelihood_estimation.md)

````mermaid
graph TD

A[Start] --> B[Define inputs: Objective functions, bounds, samples, etc.]
B --> C[Perform Latin Hypercube Sampling (LHS)]
C --> D[Save sampled domain to CSV file]

D --> E[Load dataset from CSV]
E --> F[Run simulations on objective functions]
F --> G[Combine inputs and objectives into a dataset]
G --> H[Update dataset and save to CSV]

H --> I[Iterate over optimization steps (i = n_samples to n_samples + n_experiments)]
subgraph Optimization Loop
    I --> J[Initialize or load MultiObjective Bayesian Optimization (MOBO)]
    J --> K[Set MOBO parameters]
    K --> L[Optimize acquisition function to get new X]
    L --> M[Save figure, model, and data to disk]
    M --> N[Update dataset with new X]
    N --> O[Evaluate objectives (Y) for the new X]
    O --> P[Update dataset with new X and Y]
    P --> Q[Save updated dataset to CSV]
    Q --> R{More iterations?}
    R -->|Yes| I
end

R -->|No| S[End]
