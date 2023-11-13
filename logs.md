- Normal responsiblity factor in E-Step:

$γ(z_k) = p(z_k | x(i))
= \frac{π_k * N(x(i) | μ_k, Σ_k)}{\sum_{j=1}^K π_j * N(x(i) | μ_j, Σ_j)}$

- Log of responsibility factor in E-Step:

$log γ(z_k) = log π_k + log N(x(i) | μ_k, Σ_k) - log ∑_{j=1}^K exp(log π_j + log N(x(i) | μ_j, Σ_j))$

- Log of normal Gaussian:

$log N(x | μ, Σ) = -0.5 * (d * log(2π) + log det(Σ) + (x - μ)^T Σ^-1 (x - μ))$

- Possible log-sum-exp-trick
```python

# Helper function for undoing the log
def log_sum_exp_trick(log_values):
    a = np.max(log_values)
    return a + np.log(np.sum(np.exp(log_values - a)))

# Normalize log posteriors using log-sum-exp trick
log_sum = log_sum_exp_trick(log_posteriors)
posteriors[i, :] = np.exp(log_posteriors - log_sum)

def normal(x, mean, cov, d):
    cov_det = np.abs(la.det(cov))
    try:
        cov_inv = la.inv(cov)
    except la.LinAlgError:
        cov_inv = la.pinv(cov)
            
    N = (((2 * np.pi) ** d) * cov_det) ** 0.5

    x_centered = x - mean
    exponent = 0.5 * x_centered.T @ cov_inv @ x_centered

    return np.exp(-exponent) / N
```

