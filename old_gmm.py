import numpy as np
import numpy.linalg as la
EPSILON = 1e-8

# Helper function to get the log PDF of a point for the gaussian
def log_normal(x, mean, cov, d):
    cov_det = np.abs(la.det(cov))
    try:
        cov_inv = la.inv(cov)
    except la.LinAlgError:
        cov_inv = la.pinv(cov)

    x_centered = x - mean

    log_const = d * np.log(2 * np.pi)
    log_cov = np.log(cov_det)
    log_exponent = x_centered.T @ cov_inv @ x_centered

    log_pdf = -0.5 * (log_const + log_cov + log_exponent)
    return log_pdf + EPSILON

def GMM(X, num_classes, num_epochs=15):
    n, d = X.shape
    
    # Define Parameters
    priors = np.ones(num_classes) / num_classes             # (c)
    means = kmeans_plusplus_initialization(X, num_classes)  # (c, d)
    covs = np.random.rand(num_classes, d, d) + EPSILON      # (c, d, d)
    posteriors = np.ones((n, num_classes)) / num_classes    # (n, c)
    
    # Loop through EM until convergence
    for epoch in range(num_epochs):
        print("Epoch", epoch)
        
        # Step 1. Expectation Code
        for i, point in enumerate(X):        
            for k in range(num_classes):
                prior_weight = priors[k]
                if prior_weight <= 0:
                    prior_weight = EPSILON
                    print("We got a runtime warning", priors[k])
                            
                posteriors[i, k] = np.log(prior_weight) + log_normal(point, means[k], covs[k], d)

            # Normalize posteriors
            posteriors[i] /= np.sum(posteriors[i])


        # Step 2. Maximization Code     
        Ns = np.sum(posteriors, axis=0)
        priors = Ns / np.sum(Ns)
        
        for k in range(num_classes):
            for i, point in enumerate(X):
                means[k] += posteriors[i, k] * point
            means[k] = means[k] / Ns[k]

            for i, point in enumerate(X):
                covs[k] += posteriors[i, k] * (point - means[k]) @ (point - means[k]).T
            covs[k] = covs[k] / Ns[k]

    return posteriors