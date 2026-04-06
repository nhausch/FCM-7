import numpy as np

from spline_shared import NODE_TOL, solve_full


def _bspline_knot_vector(x, dtype):
    """Open clamped cubic knot vector: x0 and xn each have multiplicity p+1."""
    x = np.asarray(x, dtype=dtype).ravel()
    if x.size < 2:
        raise ValueError("Need at least two mesh points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Mesh x must be strictly increasing")
    p = 3
    inner = x[1:-1]
    return np.concatenate(([x[0]] * (p + 1), inner, [x[-1]] * (p + 1))).astype(dtype)


def _B_scalar(x, k, i, t, dtype):
    """Cox–de Boor B_{i,k}(x); 0/0 treated as 0 (SciPy-style)."""
    t = np.asarray(t, dtype=dtype).ravel()
    n_basis = t.size - k - 1
    if i < 0 or i >= n_basis or k < 0:
        return dtype(0.0)
    if k == 0:
        if i >= t.size - 1:
            return dtype(0.0)
        left, right = t[i], t[i + 1]
        if left <= x < right:
            return dtype(1.0)
        scale = max(abs(x), abs(right), 1.0)
        if abs(x - t[-1]) <= NODE_TOL * scale and abs(right - t[-1]) <= NODE_TOL * scale:
            if left <= x <= right:
                return dtype(1.0)
        return dtype(0.0)
    d1 = t[i + k] - t[i]
    d2 = t[i + k + 1] - t[i + 1]
    c1 = dtype(0.0) if d1 == 0 else (x - t[i]) / d1 * _B_scalar(x, k - 1, i, t, dtype)
    c2 = dtype(0.0) if d2 == 0 else (t[i + k + 1] - x) / d2 * _B_scalar(x, k - 1, i + 1, t, dtype)
    return c1 + c2


def _dB_scalar(x, k, i, t, dtype):
    """First derivative d/dx B_{i,k}(x)."""
    if k <= 0:
        return dtype(0.0)
    d1 = t[i + k] - t[i]
    d2 = t[i + k + 1] - t[i + 1]
    t1 = dtype(0.0) if d1 == 0 else dtype(k) / d1 * _B_scalar(x, k - 1, i, t, dtype)
    t2 = dtype(0.0) if d2 == 0 else dtype(k) / d2 * _B_scalar(x, k - 1, i + 1, t, dtype)
    return t1 - t2


def _d2B_scalar(x, k, i, t, dtype):
    """Second derivative d^2/dx^2 B_{i,k}(x)."""
    if k <= 1:
        return dtype(0.0)
    d1 = t[i + k] - t[i]
    d2 = t[i + k + 1] - t[i + 1]
    t1 = dtype(0.0) if d1 == 0 else dtype(k) / d1 * _dB_scalar(x, k - 1, i, t, dtype)
    t2 = dtype(0.0) if d2 == 0 else dtype(k) / d2 * _dB_scalar(x, k - 1, i + 1, t, dtype)
    return t1 - t2


def _bspline_basis_row(xq, t, p, n_basis, dtype):
    row = np.empty(n_basis, dtype=dtype)
    for j in range(n_basis):
        row[j] = _B_scalar(xq, p, j, t, dtype)
    return row


def _bspline_d1_row(xq, t, p, n_basis, dtype):
    row = np.empty(n_basis, dtype=dtype)
    for j in range(n_basis):
        row[j] = _dB_scalar(xq, p, j, t, dtype)
    return row


def _bspline_d2_row(xq, t, p, n_basis, dtype):
    row = np.empty(n_basis, dtype=dtype)
    for j in range(n_basis):
        row[j] = _d2B_scalar(xq, p, j, t, dtype)
    return row


def _build_spline2_system(x, y, bc_type, bc_values, p, dtype):
    """Linear system for B-spline coefficients (n+3 unknowns for cubic)."""
    t = _bspline_knot_vector(x, dtype)
    n_mesh = x.size
    n_basis = t.size - p - 1
    m = n_mesh + 2
    A = np.zeros((m, n_basis), dtype=dtype)
    rhs = np.zeros(m, dtype=dtype)

    for i in range(n_mesh):
        A[i, :] = _bspline_basis_row(x[i], t, p, n_basis, dtype)
        rhs[i] = y[i]

    if bc_type == "natural":
        A[n_mesh, :] = _bspline_d2_row(x[0], t, p, n_basis, dtype)
        A[n_mesh + 1, :] = _bspline_d2_row(x[-1], t, p, n_basis, dtype)
        rhs[n_mesh : n_mesh + 2] = 0.0
    elif bc_type == "clamped":
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("clamped requires bc_values=(s'(a), s'(b))")
        fp0, fpn = bc_values
        A[n_mesh, :] = _bspline_d1_row(x[0], t, p, n_basis, dtype)
        A[n_mesh + 1, :] = _bspline_d1_row(x[-1], t, p, n_basis, dtype)
        rhs[n_mesh] = fp0
        rhs[n_mesh + 1] = fpn
    elif bc_type == "curvature":
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("curvature requires bc_values=(s''(a), s''(b))")
        M0, Mn = bc_values
        A[n_mesh, :] = _bspline_d2_row(x[0], t, p, n_basis, dtype)
        A[n_mesh + 1, :] = _bspline_d2_row(x[-1], t, p, n_basis, dtype)
        rhs[n_mesh] = M0
        rhs[n_mesh + 1] = Mn
    elif bc_type == "not_a_knot":
        if n_mesh < 4:
            raise ValueError("not_a_knot needs at least four mesh points")
        h0 = x[1] - x[0]
        h1 = x[2] - x[1]
        d0 = _bspline_d2_row(x[0], t, p, n_basis, dtype)
        d1 = _bspline_d2_row(x[1], t, p, n_basis, dtype)
        d2 = _bspline_d2_row(x[2], t, p, n_basis, dtype)
        A[n_mesh, :] = d0 / h0 - d1 * (1.0 / h0 + 1.0 / h1) + d2 / h1
        rhs[n_mesh] = 0.0

        hm2 = x[-2] - x[-3]
        hm1 = x[-1] - x[-2]
        dm2 = _bspline_d2_row(x[-3], t, p, n_basis, dtype)
        dm1 = _bspline_d2_row(x[-2], t, p, n_basis, dtype)
        dn = _bspline_d2_row(x[-1], t, p, n_basis, dtype)
        A[n_mesh + 1, :] = hm1 * dm2 - (hm2 + hm1) * dm1 + hm2 * dn
        rhs[n_mesh + 1] = 0.0
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    return A, rhs, t


def setup_spline2(x, f, bc_type="natural", bc_values=None, dtype=np.float64):
    """
    Spline Code 2: interpolatory cubic spline, unknowns are cubic B-spline coefficients.
    """
    x = np.asarray(x, dtype=dtype).ravel()
    y = np.asarray(f(x), dtype=dtype).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError("f(x) must return array of same length as x")
    p = 3
    A, rhs, t = _build_spline2_system(x, y, bc_type, bc_values, p, dtype)
    c = solve_full(A, rhs, dtype)
    return {"x": x, "y": y, "t": t, "p": p, "c": c, "bc_type": bc_type}


def spline2_eval(x_eval, spline2, dtype=np.float64, extrapolate=False):
    """Evaluate s(x) = sum_j c_j B_{j,3}(x) at x_eval (NaN outside by default)."""
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    t = spline2["t"]
    c = spline2["c"]
    p = int(spline2["p"])
    x0, xn = spline2["x"][0], spline2["x"][-1]
    n_basis = c.size
    m = x_eval.size
    out = np.empty(m, dtype=dtype)

    for k in range(m):
        xv = x_eval[k]
        if not extrapolate and (xv < x0 or xv > xn):
            out[k] = np.nan
            continue
        s = dtype(0.0)
        for j in range(n_basis):
            s += c[j] * _B_scalar(xv, p, j, t, dtype)
        out[k] = s
    return out

