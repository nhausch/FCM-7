import numpy as np

from meshes import build_mesh, chebyshev_second_kind

# Tolerance for confluent divided differences (repeated nodes).
NODE_TOL = 1.5 * np.finfo(np.float64).eps


# Builds the Newton divided-difference coefficients.
def divided_differences_newton(z_nodes, f, fp=None, dtype=np.float64):
    z_nodes = np.asarray(z_nodes, dtype=dtype).ravel()
    n = z_nodes.size
    Q = np.zeros((n, n), dtype=dtype)
    for i in range(n):
        Q[i, 0] = f(z_nodes[i])
    scale = np.max(np.abs(z_nodes)) if n > 0 else dtype(1.0)
    tol = NODE_TOL * max(scale, dtype(1.0)) * max(n, 1)

    for j in range(1, n):
        for i in range(j, n):
            denom = z_nodes[i] - z_nodes[i - j]

            # Use the derivative for repeated nodes.
            if abs(denom) <= tol:
                if fp is None:
                    raise ValueError(
                        "Derivative fp is required for repeated (Hermite) nodes."
                    )
                Q[i, j] = fp(z_nodes[i])
            else:
                Q[i, j] = (Q[i, j - 1] - Q[i - 1, j - 1]) / denom
    return np.array([Q[i, i] for i in range(n)], dtype=dtype)


# Evaluates a newton polynpmial using Horner's rule.
def newton_horner(x, z_nodes, coeffs, dtype=np.float64):
    z_nodes = np.asarray(z_nodes, dtype=dtype).ravel()
    coeffs = np.asarray(coeffs, dtype=dtype).ravel()
    n = coeffs.size
    if n == 0:
        return dtype(0.0)
    p = coeffs[n - 1]
    for k in range(n - 2, -1, -1):
        p = coeffs[k] + (x - z_nodes[k]) * p
    return p


def newton_horner_deriv1(x, z_nodes, coeffs, dtype=np.float64):
    """
    First derivative of the Newton interpolant at x (same z_nodes and coeffs as
    newton_horner). Uses the coupled Horner recurrence; no finite differencing.
    """
    z_nodes = np.asarray(z_nodes, dtype=dtype).ravel()
    coeffs = np.asarray(coeffs, dtype=dtype).ravel()
    n = coeffs.size
    if n == 0:
        return dtype(0.0)
    p = coeffs[n - 1]
    dp = dtype(0.0)
    for k in range(n - 2, -1, -1):
        dp = p + (x - z_nodes[k]) * dp
        p = coeffs[k] + (x - z_nodes[k]) * p
    return dp


# Computes the local nodes for a given interval and mesh type.
def _local_nodes(ai, bi, degree, local_nodes, dtype=np.float64):

    # The number of required points is one more than the desired degree.
    npts = degree + 1
    if local_nodes == "uniform":
        return np.linspace(ai, bi, npts, dtype=dtype)
    if local_nodes == "cheb2":
        return chebyshev_second_kind(ai, bi, npts, dtype=dtype)
    raise ValueError("local_nodes must be 'uniform' or 'cheb2'")

# Precomputes piecewise Newton interpolants on [a,b] by partitioning into
# the given number of subintervals. 
def setup_piecewise_newton(
    a,
    b,
    num_subintervals,
    f,
    degree=3,
    df=None,
    hermite=False,
    breakpoint_mesh="uniform",
    local_nodes="uniform",
    dtype=np.float64,
    breakpoints=None,
):
    """
    Precomputes piecewise Newton interpolants on [a,b] by partitioning into
    the given number of subintervals.

    If breakpoints is not None, it must be a strictly increasing 1D array of
    length num_subintervals + 1; a and b are ignored for mesh construction (use
    breakpoints[0] and breakpoints[-1] as the domain).

    If hermite is False:
        degree must be 1, 2, or 3. On each [ai, bi], interpolate f at degree+1
        nodes (endpoints plus interior points). Interior placement is uniform or
        Chebyshev-2 (local_nodes).

    If hermite is True:
        On each subinterval, the cubic Hermite polynomial matches f and f' at both
        endpoints (confluent Newton nodes [ai, ai, bi, bi]). df must be provided;
        degree is ignored.

    Returns
    -------
    breakpoints : ndarray, shape (M+1,)
    z_list : list of ndarray
        Newton nodes per subinterval.
    coeffs_list : list of ndarray
        Divided-difference coefficients per subinterval (same length as z_list[i]).
    """
    if num_subintervals < 1:
        raise ValueError("num_subintervals must be >= 1")
    if hermite:
        if df is None:
            raise ValueError("df is required when hermite=True")
    else:
        if degree not in (1, 2, 3):
            raise ValueError("degree must be 1, 2, or 3 when hermite=False")

    if breakpoints is not None:
        breakpoints = np.asarray(breakpoints, dtype=dtype).ravel()
        if breakpoints.size != num_subintervals + 1:
            raise ValueError(
                "breakpoints must have length num_subintervals + 1 "
                f"(got {breakpoints.size}, expected {num_subintervals + 1})"
            )
        if np.any(np.diff(breakpoints) <= 0):
            raise ValueError("breakpoints must be strictly increasing")
    else:
        breakpoints = build_mesh(breakpoint_mesh, a, b, num_subintervals + 1, dtype)
    M = num_subintervals
    z_list = []
    coeffs_list = []

    for i in range(M):
        ai = breakpoints[i]
        bi = breakpoints[i + 1]
        if hermite:
            z = np.array([ai, ai, bi, bi], dtype=dtype)
            c = divided_differences_newton(z, f, fp=df, dtype=dtype)
        else:
            z = _local_nodes(ai, bi, degree, local_nodes, dtype)
            c = divided_differences_newton(z, f, fp=None, dtype=dtype)
        z_list.append(z)
        coeffs_list.append(c)

    return breakpoints, z_list, coeffs_list

# Evaluates the piecewise Newton interpolant at x_eval.
# Points outside [breakpoints[0], breakpoints[-1]] use the first/last panel
# polynomial (extrapolation).
def piecewise_newton_eval(
    x_eval, breakpoints, z_list, coeffs_list, dtype=np.float64
):
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    breakpoints = np.asarray(breakpoints, dtype=dtype).ravel()
    m = x_eval.size
    out = np.empty(m, dtype=dtype)
    n_pieces = len(z_list)

    idx = np.searchsorted(breakpoints, x_eval, side="right") - 1
    idx = np.clip(idx, 0, n_pieces - 1)
    idx[x_eval < breakpoints[0]] = 0
    idx[x_eval > breakpoints[-1]] = n_pieces - 1

    for k in range(m):
        j = int(idx[k])
        out[k] = newton_horner(x_eval[k], z_list[j], coeffs_list[j], dtype=dtype)
    return out


def piecewise_newton_deriv1_eval(
    x_eval, breakpoints, z_list, coeffs_list, dtype=np.float64
):
    """
    First derivative of the piecewise Newton interpolant. Uses the same panel
    index as piecewise_newton_eval (searchsorted side='right'), so at an
    interior breakpoint the derivative is one-sided from the right-hand panel
    when the global interpolant is only C0.
    """
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    breakpoints = np.asarray(breakpoints, dtype=dtype).ravel()
    m = x_eval.size
    out = np.empty(m, dtype=dtype)
    n_pieces = len(z_list)

    idx = np.searchsorted(breakpoints, x_eval, side="right") - 1
    idx = np.clip(idx, 0, n_pieces - 1)
    idx[x_eval < breakpoints[0]] = 0
    idx[x_eval > breakpoints[-1]] = n_pieces - 1

    for k in range(m):
        j = int(idx[k])
        out[k] = newton_horner_deriv1(
            x_eval[k], z_list[j], coeffs_list[j], dtype=dtype
        )
    return out
