import numpy as np


# Tolerance for "exact node" branch to avoid catastrophic cancellation.
NODE_TOL = 1.5 * np.finfo(np.float64).eps

# Computes gamma_i = 1 / prod_{j != i} (x_i - x_j) for all i.
def compute_gamma(x, dtype=np.float64):
    x = np.asarray(x, dtype=dtype).ravel()
    n = x.size
    gamma = np.empty(n, dtype=dtype)
    for i in range(n):
        diff = x[i] - x
        diff[i] = 1.0  # skip j == i (product over j != i)
        gamma[i] = 1.0 / np.prod(diff)
    return gamma

# Precomputes gamma and y = f(x) for Barycentric Form 1.
def setup_barycentric1(x, f, dtype=np.float64):
    x = np.asarray(x, dtype=dtype).ravel()
    gamma = compute_gamma(x, dtype)
    y = np.asarray(f(x), dtype=dtype).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError("f(x) must return array of same length as x")
    return gamma, y

# Evaluates the interpolant at x_eval using Barycentric Form 1.
# p(x) = (sum_i gamma_i*y_i/(x-x_i)) / (sum_i gamma_i/(x-x_i)).
# Uses exact-node and near-node safeguards.
def barycentric1_eval(x_eval, x_nodes, gamma, y, dtype=np.float64):
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    x_nodes = np.asarray(x_nodes, dtype=dtype).ravel()
    gamma = np.asarray(gamma, dtype=dtype).ravel()
    y = np.asarray(y, dtype=dtype).ravel()
    n = x_nodes.size
    m = x_eval.size
    out = np.empty(m, dtype=dtype)
    scale = np.max(np.abs(x_nodes)) if n > 0 else dtype(1.0)
    tol = scale * NODE_TOL * max(n, 1)

    for k in range(m):
        xk = x_eval[k]
        d = xk - x_nodes
        j_near = np.argmin(np.abs(d))
        if np.abs(d[j_near]) <= tol:
            out[k] = y[j_near]
            continue
        denom = np.sum(gamma / d)
        num = np.sum(gamma * y / d)
        out[k] = num / denom
    return out
