import numpy as np

# Tolerance for knot span search and exact mesh hits.
NODE_TOL = 1.5 * np.finfo(np.float64).eps


def solve_full(A, rhs, dtype=np.float64):
    """Dense solve wrapper (kept for consistent call sites)."""
    return np.linalg.solve(np.asarray(A, dtype=dtype), np.asarray(rhs, dtype=dtype))
