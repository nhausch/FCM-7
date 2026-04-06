import numpy as np

# Tolerance for knot span search and exact mesh hits.
NODE_TOL = 1.5 * np.finfo(np.float64).eps


def _build_moment_system(x, y, bc_type, bc_values, dtype):
    """
    Assemble (n+1)x(n+1) linear system A M = b for second derivatives M_i = s''(x_i).
    x must be strictly increasing, len n+1.
    """
    x = np.asarray(x, dtype=dtype).ravel()
    y = np.asarray(y, dtype=dtype).ravel()
    n = x.size - 1
    if n < 1:
        raise ValueError("Need at least two mesh points")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Mesh x must be strictly increasing")

    A = np.zeros((n + 1, n + 1), dtype=dtype)
    rhs = np.zeros(n + 1, dtype=dtype)

    # Interior rows i = 1 .. n-1
    for i in range(1, n):
        hi_1 = h[i - 1]
        hi = h[i]
        A[i, i - 1] = hi_1
        A[i, i] = 2.0 * (hi_1 + hi)
        A[i, i + 1] = hi
        rhs[i] = 6.0 * (
            (y[i + 1] - y[i]) / hi - (y[i] - y[i - 1]) / hi_1
        )

    if bc_type == "natural":
        A[0, 0] = 1.0
        A[n, n] = 1.0
        rhs[0] = 0.0
        rhs[n] = 0.0
    elif bc_type == "clamped":
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("clamped requires bc_values=(s'(a), s'(b))")
        fp0, fpn = bc_values
        h0 = h[0]
        hn_1 = h[n - 1]
        A[0, 0] = 2.0 * h0
        A[0, 1] = h0
        rhs[0] = 6.0 * ((y[1] - y[0]) / h0 - fp0)
        A[n, n - 1] = hn_1
        A[n, n] = 2.0 * hn_1
        rhs[n] = 6.0 * (fpn - (y[n] - y[n - 1]) / hn_1)
    elif bc_type == "curvature":
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("curvature requires bc_values=(s''(a), s''(b))")
        M0, Mn = bc_values
        A[0, 0] = 1.0
        rhs[0] = M0
        A[n, n] = 1.0
        rhs[n] = Mn
    elif bc_type == "not_a_knot":
        if n < 2:
            raise ValueError("not_a_knot needs at least three mesh points")
        h0, h1 = h[0], h[1]
        A[0, 0] = h1
        A[0, 1] = -(h0 + h1)
        A[0, 2] = h0
        rhs[0] = 0.0
        hm2, hm1 = h[n - 2], h[n - 1]
        A[n, n - 2] = hm1
        A[n, n - 1] = -(hm2 + hm1)
        A[n, n] = hm2
        rhs[n] = 0.0
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    return A, rhs


def _build_slope_system(x, y, bc_type, bc_values, dtype):
    """
    Assemble (n+1)x(n+1) system for slopes m_i = s'(x_i) from C^2 continuity
    at interior knots plus two boundary rows.
    """
    x = np.asarray(x, dtype=dtype).ravel()
    y = np.asarray(y, dtype=dtype).ravel()
    n = x.size - 1
    if n < 1:
        raise ValueError("Need at least two mesh points")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Mesh x must be strictly increasing")

    A = np.zeros((n + 1, n + 1), dtype=dtype)
    rhs = np.zeros(n + 1, dtype=dtype)

    # Interior: equate s'' from left and right at x_i (i = 1..n-1)
    for i in range(1, n):
        hi_1 = h[i - 1]
        hi = h[i]
        A[i, i - 1] = hi
        A[i, i] = 2.0 * (hi_1 + hi)
        A[i, i + 1] = hi_1
        rhs[i] = 3.0 * (
            hi * (y[i] - y[i - 1]) / hi_1 + hi_1 * (y[i + 1] - y[i]) / hi
        )

    if bc_type == "natural":
        h0 = h[0]
        hn_1 = h[n - 1]
        A[0, 0] = 2.0 * h0
        A[0, 1] = h0
        rhs[0] = 3.0 * (y[1] - y[0])
        A[n, n - 1] = hn_1
        A[n, n] = 2.0 * hn_1
        rhs[n] = 3.0 * (y[n] - y[n - 1])
    elif bc_type == "clamped":
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("clamped requires bc_values=(s'(a), s'(b))")
        fp0, fpn = bc_values
        A[0, 0] = 1.0
        rhs[0] = fp0
        A[n, n] = 1.0
        rhs[n] = fpn
    elif bc_type == "curvature":
        if bc_values is None or len(bc_values) != 2:
            raise ValueError("curvature requires bc_values=(s''(a), s''(b))")
        M0, Mn = bc_values
        h0 = h[0]
        hn_1 = h[n - 1]
        # s''(x0+) = 6/h0^2*(y1-y0) - (4*m0+2*m1)/h0 = M0
        A[0, 0] = 4.0 / h0
        A[0, 1] = 2.0 / h0
        rhs[0] = 6.0 * (y[1] - y[0]) / (h0 * h0) - M0
        A[n, n - 1] = 2.0 / hn_1
        A[n, n] = 4.0 / hn_1
        rhs[n] = 6.0 * (y[n] - y[n - 1]) / (hn_1 * hn_1) + Mn
    elif bc_type == "not_a_knot":
        if n < 2:
            raise ValueError("not_a_knot needs at least three mesh points")
        h0, h1 = h[0], h[1]
        A[0, 0] = h1
        A[0, 1] = -(h0 + h1)
        A[0, 2] = h0
        rhs[0] = 0.0
        hm2, hm1 = h[n - 2], h[n - 1]
        A[n, n - 2] = hm1
        A[n, n - 1] = -(hm2 + hm1)
        A[n, n] = hm2
        rhs[n] = 0.0
    else:
        raise ValueError(f"Unknown bc_type: {bc_type}")

    return A, rhs


def _solve_full(A, rhs, dtype):
    return np.linalg.solve(A, rhs)


def setup_spline1(x, f, bc_type="natural", bc_values=None, representation="moments", dtype=np.float64):
    """
    Spline Code 1: interpolatory cubic spline on mesh x.

    Parameters
    ----------
    x : array of strictly increasing knots (length n+1).
    f : callable; y = f(x) at mesh (same shape as x).
    bc_type : 'natural' | 'clamped' | 'curvature' | 'not_a_knot'
        Two boundary conditions are fixed by this choice:
        - natural: s''(a)=s''(b)=0
        - clamped: s'(a), s'(b) from bc_values=(g0, gn)
        - curvature: s''(a), s''(b) from bc_values=(M0, Mn)
        - not_a_knot: s''' continuous at x_1 and x_{n-1}
    bc_values : optional pair; see bc_type.
    representation : 'moments' (s''_i unknowns) or 'slopes' (s'_i unknowns).

    Returns
    -------
    dict with keys: x, y, h, representation, bc_type, moments (if moments), slopes (if slopes)
    """
    x = np.asarray(x, dtype=dtype).ravel()
    y = np.asarray(f(x), dtype=dtype).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError("f(x) must return array of same length as x")
    if representation not in ("moments", "slopes"):
        raise ValueError("representation must be 'moments' or 'slopes'")
    if representation == "slopes" and bc_type == "not_a_knot":
        raise ValueError("not_a_knot is only supported with representation='moments'")

    if representation == "moments":
        A, rhs = _build_moment_system(x, y, bc_type, bc_values, dtype)
        M = _solve_full(A, rhs, dtype)
        return {
            "x": x,
            "y": y,
            "h": np.diff(x),
            "representation": "moments",
            "bc_type": bc_type,
            "moments": M,
        }
    A, rhs = _build_slope_system(x, y, bc_type, bc_values, dtype)
    m = _solve_full(A, rhs, dtype)
    return {
        "x": x,
        "y": y,
        "h": np.diff(x),
        "representation": "slopes",
        "bc_type": bc_type,
        "slopes": m,
    }


def spline1_eval(x_eval, spline1, dtype=np.float64):
    """Evaluate Spline Code 1 precomputed spline at x_eval."""
    x_eval = np.asarray(x_eval, dtype=dtype).ravel()
    x = spline1["x"]
    y = spline1["y"]
    h = spline1["h"]
    n = x.size - 1
    m = x_eval.size
    out = np.empty(m, dtype=dtype)
    scale = np.max(np.abs(x)) if x.size > 0 else dtype(1.0)
    tol = NODE_TOL * max(scale, dtype(1.0)) * max(x.size, 1)

    if spline1["representation"] == "moments":
        M = spline1["moments"]
        for k in range(m):
            xv = x_eval[k]
            j = int(np.searchsorted(x, xv, side="right") - 1)
            j = np.clip(j, 0, n - 1)
            if abs(xv - x[j]) <= tol:
                out[k] = y[j]
                continue
            if abs(xv - x[j + 1]) <= tol:
                out[k] = y[j + 1]
                continue
            t = xv - x[j]
            hj = h[j]
            Mi, Mi1 = M[j], M[j + 1]
            yi, yi1 = y[j], y[j + 1]
            out[k] = (
                yi
                + t
                * (
                    (yi1 - yi) / hj
                    - hj * (2.0 * Mi + Mi1) / 6.0
                )
                + (t * t) * Mi / 2.0
                + (t * t * t) * (Mi1 - Mi) / (6.0 * hj)
            )
    else:
        msl = spline1["slopes"]
        for k in range(m):
            xv = x_eval[k]
            j = int(np.searchsorted(x, xv, side="right") - 1)
            j = np.clip(j, 0, n - 1)
            if abs(xv - x[j]) <= tol:
                out[k] = y[j]
                continue
            if abs(xv - x[j + 1]) <= tol:
                out[k] = y[j + 1]
                continue
            t = xv - x[j]
            hj = h[j]
            t_rel = t / hj
            h00 = (1.0 + 2.0 * t_rel) * (1.0 - t_rel) ** 2
            h10 = t * (1.0 - t_rel) ** 2
            h01 = t_rel**2 * (3.0 - 2.0 * t_rel)
            h11 = t_rel**2 * (t - hj)
            out[k] = h00 * y[j] + h10 * msl[j] + h01 * y[j + 1] + h11 * msl[j + 1]
    return out


# --- Spline Code 2: cubic B-spline basis (Cox–de Boor) ---


def _bspline_knot_vector(x, dtype):
    """Open clamped cubic knot vector: x0 and xn each have multiplicity p+1."""
    x = np.asarray(x, dtype=dtype).ravel()
    if x.size < 2:
        raise ValueError("Need at least two mesh points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Mesh x must be strictly increasing")
    p = 3
    inner = x[1:-1]
    return np.concatenate(
        ([x[0]] * (p + 1), inner, [x[-1]] * (p + 1))
    ).astype(dtype)


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
    """Dense row A[j] = B_{j,p}(xq), j = 0..n_basis-1."""
    row = np.empty(n_basis, dtype=dtype)
    for j in range(n_basis):
        row[j] = _B_scalar(xq, p, j, t, dtype)
    return row


def _bspline_d2_row(xq, t, p, n_basis, dtype):
    """Dense row A[j] = B''_{j,p}(xq)."""
    row = np.empty(n_basis, dtype=dtype)
    for j in range(n_basis):
        row[j] = _d2B_scalar(xq, p, j, t, dtype)
    return row


def _bspline_d1_row(xq, t, p, n_basis, dtype):
    """Dense row A[j] = B'_{j,p}(xq)."""
    row = np.empty(n_basis, dtype=dtype)
    for j in range(n_basis):
        row[j] = _dB_scalar(xq, p, j, t, dtype)
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
    Spline Code 2: same interpolatory cubic spline, unknowns are B-spline coefficients.

    Knot vector: [x0]*4, x1..x_{n-1}, [xn]*4 (cubic, open clamped). The system has
    n+3 coefficients matching n+1 interpolation values plus two boundary rows.

    bc_type : 'natural' | 'clamped' | 'curvature' | 'not_a_knot'
        not_a_knot: third-derivative continuity at x_1 and x_{n-1} (linear constraints
        on B'' coefficients), matching the standard not-a-knot cubic spline.
    """
    x = np.asarray(x, dtype=dtype).ravel()
    y = np.asarray(f(x), dtype=dtype).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError("f(x) must return array of same length as x")
    p = 3
    A, rhs, t = _build_spline2_system(x, y, bc_type, bc_values, p, dtype)
    c = np.linalg.solve(A, rhs)
    return {"x": x, "y": y, "t": t, "p": p, "c": c, "bc_type": bc_type}


def spline2_eval(x_eval, spline2, dtype=np.float64, extrapolate=False):
    """Evaluate s(x) = sum_j c_j B_{j,3}(x) at x_eval (inside [x[0], x[-1]] by default)."""
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
        if not extrapolate:
            if xv < x0 or xv > xn:
                out[k] = np.nan
                continue
        s = dtype(0.0)
        for j in range(n_basis):
            s += c[j] * _B_scalar(xv, p, j, t, dtype)
        out[k] = s
    return out
