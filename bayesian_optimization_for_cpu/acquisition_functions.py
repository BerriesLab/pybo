import numpy as np
from scipy.stats import norm
from skopt.learning import GaussianProcessRegressor


def ei(x, n_objectives, model: list[GaussianProcessRegressor], pareto_front):
    x = x.reshape(1, -1)
    mean = []
    sigma = []

    for i in range(n_objectives):
        mu, std = model[i].predict(x, return_std=True)
        mean.append(mu)
        sigma.append(std)

    # 2. Calculate the EHVI
    if n_objectives == 1:
        # Expected Improvement (EI) for single objective
        improvement = mean - pareto_front[0]  # pareto_front is just one value
        z = improvement / sigma
        ei = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
        ei = max(ei, 0)

    return ei


def ehvi_analytic(x, n_objectives, model: list[GaussianProcessRegressor], pareto_front, ref_point=None):
    
    if n_objectives != 2:
        raise ValueError("Analytic EHVI for other than two objectives is not supported.")
    
    x = x.reshape(1, -1)
    mean = []
    sigma = []

    for i in range(n_objectives):
        mu, std = model[i].predict(x, return_std=True)
        mean.append(mu)
        sigma.append(std)
    
    # Analytic EHVI for two objectives
    front = pareto_front
    mu1 = mean[0]
    mu2 = mean[1]
    sigma1 = sigma[0]
    sigma2 = sigma[1]
    r1 = ref_point[0]
    r2 = ref_point[1]

    # Standardize Pareto front and reference values
    f1_s = (front[:, 0] - mu1) / sigma1
    f2_s = (front[:, 1] - mu2) / sigma2
    r1_s = (r1 - mu1) / sigma1
    r2_s = (r2 - mu2) / sigma2

    ehvi = 0
    for i in range(len(front)):
        t1 = (r1_s - f1_s) * (r2_s - f2_s) * norm.cdf(f1_s) * norm.cdf(f2_s)
        t2 = sigma1 * (r2 - mu2) * norm.cdf(f2_s) * norm.pdf(f1_s)
        t3 = sigma2 * (r1 - mu1) * norm.pdf(f2_s) * norm.cdf(f1_s)
        t4 = sigma1 * sigma2 * norm.pdf(f1_s) * norm.pdf(f2_s)
        ehvi += t1 + t2 + t3 + t4
    ehvi = np.mean(ehvi)

    return ehvi


def ehvi_mc(x, n_objectives, model: list[GaussianProcessRegressor], pareto_front, ref_point=None):
    
    x = x.reshape(1, -1)
    mean = []
    sigma = []
    
    # Predict mean and standard deviation for each objective
    for i in range(n_objectives):
        mu, std = model[i].predict(x, return_std=True)
        mean.append(mu)
        sigma.append(std)
    
    n_samples = 100  # Number of samples
    cov = np.diag(np.array(sigma) ** 2)  # Create diagonal covariance matrix from sigmas
    samples = np.random.multivariate_normal(mean, cov, n_samples)  # (n_samples, n_objectives) 
    improvement = np.zeros(n_samples)

    for i in range(n_samples):
        sample = samples[i, :]
        # Calculate the hypervolume improvement for this sample
        dominated = False
        for j in range(pareto_front.shape[0]):
            if all(sample <= pareto_front[j]):
                dominated = True
                break
        if not dominated:  # the sample is not dominated by any point in Pareto front
            # Calculate the hypervolume improvement contributed by this sample
            hv_with_sample = 1
            hv_without_sample = 1

            # Hypervolume with the sample
            for k in range(n_objectives):
                hv_with_sample *= max(ref_point[k], sample[k]) - ref_point[k]

            # Hypervolume without the sample (Pareto front)
            for k in range(n_objectives):
                max_val_k = ref_point[k]
                for j in range(pareto_front.shape[0]):
                    max_val_k = max(max_val_k, pareto_front[j, k])
                hv_without_sample *= max_val_k - ref_point[k]
            improvement[i] = hv_with_sample - hv_without_sample

    ehvi = np.mean(improvement)

    return ehvi


def ehvi(x, n_objectives, model: list[GaussianProcessRegressor], pareto_front, ref_point=None):
    x = x.reshape(1, -1)
    mean = []
    sigma = []

    for i in range(n_objectives):
        mu, std = model[i].predict(x, return_std=True)
        mean.append(mu)
        sigma.append(std)

    # 1. Define the reference point
    if ref_point is None:
        raise ValueError("A reference point mus tbe defined before optimizing the EHVI.")

    # 2. Calculate the EHVI
    if n_objectives == 1:
        # Expected Improvement (EI) for single objective
        improvement = mean - pareto_front[0]  # pareto_front is just one value
        z = improvement / sigma
        ehvi = (improvement * norm.cdf(z) + sigma * norm.pdf(z))
        ehvi = max(ehvi, 0)

    # TODO: check analytical formula and its implementation
    elif n_objectives == 2:
        # Analytic EHVI for two objectives
        front = pareto_front
        mu1 = mean[0]
        mu2 = mean[1]
        sigma1 = sigma[0]
        sigma2 = sigma[1]
        r1 = ref_point[0]
        r2 = ref_point[1]

        # Standardize Pareto front and reference values
        f1_s = (front[:, 0] - mu1) / sigma1
        f2_s = (front[:, 1] - mu2) / sigma2
        r1_s = (r1 - mu1) / sigma1
        r2_s = (r2 - mu2) / sigma2

        ehvi = 0
        for i in range(len(front)):
            t1 = (r1_s - f1_s) * (r2_s - f2_s) * norm.cdf(f1_s) * norm.cdf(f2_s)
            t2 = sigma1 * (r2 - mu2) * norm.cdf(f2_s) * norm.pdf(f1_s)
            t3 = sigma2 * (r1 - mu1) * norm.pdf(f2_s) * norm.cdf(f1_s)
            t4 = sigma1 * sigma2 * norm.pdf(f1_s) * norm.pdf(f2_s)
            ehvi += t1 + t2 + t3 + t4
        ehvi = np.mean(ehvi)

    else:
        # For more than 2 objectives, use Monte Carlo approximation.
        # This is computationally more intensive.
        n_samples = 10000  # Number of samples
        samples = np.random.multivariate_normal(mean, cov, n_samples)  # (n_samples, n_objectives)
        improvement = np.zeros(n_samples)

        for i in range(n_samples):
            sample = samples[i, :]
            # Calculate the hypervolume improvement for this sample
            dominated = False
            for j in range(pareto_front.shape[0]):
                if all(sample <= pareto_front[j]):
                    dominated = True
                    break
            if not dominated:  # the sample is not dominated by any point in Pareto front
                # Calculate the hypervolume improvement contributed by this sample
                hv_with_sample = 1
                hv_without_sample = 1

                # Hypervolume with the sample
                for k in range(n_objectives):
                    hv_with_sample *= max(ref_point[k], sample[k]) - ref_point[k]

                # Hypervolume without the sample (Pareto front)
                for k in range(n_objectives):
                    max_val_k = ref_point[k]
                    for j in range(pareto_front.shape[0]):
                        max_val_k = max(max_val_k, pareto_front[j, k])
                    hv_without_sample *= max_val_k - ref_point[k]
                improvement[i] = hv_with_sample - hv_without_sample

        ehvi = np.mean(improvement)

    return ehvi