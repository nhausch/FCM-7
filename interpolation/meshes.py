import numpy as np


# Uniform nodes on [a, b]: x_k = a + (b-a)*k/(n-1), k=0..n-1. Sorted ascending.
def uniform_mesh(a, b, n, dtype=np.float64):
    x = np.linspace(a, b, n, dtype=dtype)
    return np.sort(x)

# Chebyshev nodes of the first kind on [a, b].
# On [-1,1]: t_k = cos((2k-1)*pi/(2n)), k=1..n. Then affine map to [a,b].
# Returns sorted ascending.
def chebyshev_first_kind(a, b, n, dtype=np.float64):
    k = np.arange(1, n + 1, dtype=dtype)
    t = np.cos((2 * k - 1) * np.pi / (2 * n))
    x = (b - a) / 2 * t + (a + b) / 2
    return np.sort(x).astype(dtype)

# Chebyshev nodes of the second kind on [a, b] (n points).
# On [-1,1]: t_k = cos(k*pi/(n-1)), k=0..n-1 (n nodes). Then affine map to [a,b].
# Returns sorted ascending.
def chebyshev_second_kind(a, b, n, dtype=np.float64):
    if n == 1:
        return np.array([(a + b) / 2], dtype=dtype)
    k = np.arange(0, n, dtype=dtype)
    t = np.cos(k * np.pi / (n - 1))
    x = (b - a) / 2 * t + (a + b) / 2
    return np.sort(x).astype(dtype)


def build_mesh(mesh_type, a, b, n, dtype=np.float64):
    """Return 1D mesh for mesh_type: 'uniform', 'cheb1', or 'cheb2'."""
    if mesh_type == "uniform":
        return uniform_mesh(a, b, n, dtype)
    if mesh_type == "cheb1":
        return chebyshev_first_kind(a, b, n, dtype)
    if mesh_type == "cheb2":
        return chebyshev_second_kind(a, b, n, dtype)
    raise ValueError(f"Unknown mesh_type: {mesh_type}")
