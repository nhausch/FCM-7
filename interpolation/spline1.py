import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
_rp = str(_root)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

from utils import NODE_TOL, solve_full


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


def setup_spline1(
    x,
    f,
    bc_type="natural",
    bc_values=None,
    representation="moments",
    dtype=np.float64,
):
    """
    Spline Code 1: interpolatory cubic spline on mesh x.

    bc_type: 'natural' | 'clamped' | 'curvature' | 'not_a_knot'
    representation: 'moments' (s''_i unknowns) or 'slopes' (s'_i unknowns).
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
        M = solve_full(A, rhs, dtype)
        return {
            "x": x,
            "y": y,
            "h": np.diff(x),
            "representation": "moments",
            "bc_type": bc_type,
            "moments": M,
        }
    A, rhs = _build_slope_system(x, y, bc_type, bc_values, dtype)
    m = solve_full(A, rhs, dtype)
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
                + t * ((yi1 - yi) / hj - hj * (2.0 * Mi + Mi1) / 6.0)
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

