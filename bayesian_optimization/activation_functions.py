def expected_improvement(x: float, gpr: GaussianProcessRegressor, y_best):
    mu, sigma = gp.predict(x, return_std=True)
    sigma = np.maximum(sigma, 1e-8)  # avoid division by zero
    z = (mu - y_best) / sigma
    ei = (mu - y_best) * norm.cdf(z) + sigma * norm.pdf(z)
    return -ei
