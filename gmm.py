import numpy as np
import numpy.linalg as la
from k_means import kmeans_plusplus_initialization

EPSILON = 1e-8

# Helper function to get the log PDF of a point for the gaussian
def log_normal(x, mean, cov, d):
    try:
        cov_inv = la.inv(cov)
    except la.LinAlgError:
        cov_inv = la.pinv(cov)

    x_centered = x - mean

    log_const = d * np.log(2 * np.pi)
    log_cov = np.log(la.det(cov) + EPSILON)
    log_exponent = x_centered.T @ cov_inv @ x_centered

    log_pdf = -0.5 * (log_const + log_cov + log_exponent)
    return log_pdf

# GMM using Expectation Maximization
def GMM(X, num_classes, num_epochs=15):
    n, d = X.shape
    
    # Define Parameters
    priors = np.ones(num_classes) / num_classes                 # (c)
    means = kmeans_plusplus_initialization(X, num_classes)      # (c, d)
    covs = np.stack([np.eye(d) for _ in range(num_classes)])    # (c, d, d)
    posteriors = np.ones((n, num_classes)) / num_classes        # (n, c)

    # Loop through EM until convergence
    for epoch in range(num_epochs):
        # print("Epoch", epoch)
        
        # Step 1. Expectation Code
        for i, point in enumerate(X):    
            log_posteriors = np.zeros(num_classes)    
            for k in range(num_classes):
                log_prior_weight = np.log(priors[k] + EPSILON)
                log_likelihood = log_normal(point, means[k], covs[k], d)
                log_posteriors[k] = log_prior_weight + log_likelihood

            # Normalize log posteriors
            posteriors[i, :] = log_posteriors - np.log(np.sum(np.exp(log_posteriors)))

        # Step 2. Maximization Code     
        Ns = np.sum(posteriors, axis=0)
        priors = Ns / np.sum(Ns)
        
        for k in range(num_classes):
            means[k] = np.sum(posteriors[:, k, np.newaxis] * X, axis=0) / Ns[k]

            covs[k] = np.zeros((d, d))
            for i, point in enumerate(X):
                x_centered = point - means[k]
                covs[k] += posteriors[i, k] * np.outer(x_centered, x_centered)
            covs[k] = covs[k] / Ns[k]

    return posteriors, means, covs, priors
    
