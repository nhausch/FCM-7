"""
Named test functions for interpolation validation.

Vectorized in x (numpy arrays or scalars). Optional derivative df for Hermite / clamped splines.
"""

import numpy as np


def _as_array(x):
    return np.asarray(x, dtype=np.float64)


def make_linear():
    """f(x) = 2x - 1. Natural spline reproduces exactly (g'' = 0)."""

    def f(x):
        x = _as_array(x)
        return 2.0 * x - 1.0

    def df(x):
        x = _as_array(x)
        return np.full_like(x, 2.0, dtype=np.float64)

    def d2f(x):
        x = _as_array(x)
        return np.zeros_like(x, dtype=np.float64)

    return {
        "f": f,
        "df": df,
        "d2f": d2f,
        "description": "linear 2x-1",
        "recommended_bc": "natural",
    }


def make_quadratic():
    """f(x) = x^2 - x. Natural spline does not match globally (f'' != 0 at ends in general domain)."""

    def f(x):
        x = _as_array(x)
        return x * x - x

    def df(x):
        x = _as_array(x)
        return 2.0 * x - 1.0

    def d2f(x):
        x = _as_array(x)
        return np.full_like(x, 2.0, dtype=np.float64)

    return {
        "f": f,
        "df": df,
        "d2f": d2f,
        "description": "quadratic x^2-x",
        "recommended_bc": "clamped",
    }


def make_cubic_poly():
    """f(x) = x^3 - x. Use with clamped BC and bc_values from df at endpoints."""

    def f(x):
        x = _as_array(x)
        return x * x * x - x

    def df(x):
        x = _as_array(x)
        return 3.0 * x * x - 1.0

    def d2f(x):
        x = _as_array(x)
        return 6.0 * x

    return {
        "f": f,
        "df": df,
        "d2f": d2f,
        "description": "cubic x^3-x",
        "recommended_bc": "clamped",
    }


def make_random_cubic_clamped(seed=0):
    """Random cubic on [a,b]; use clamped BC with true derivatives at a, b."""
    rng = np.random.default_rng(seed)
    c0, c1, c2, c3 = rng.standard_normal(4)

    def f(x):
        x = _as_array(x)
        return ((c3 * x + c2) * x + c1) * x + c0

    def df(x):
        x = _as_array(x)
        return (3.0 * c3 * x + 2.0 * c2) * x + c1

    def d2f(x):
        x = _as_array(x)
        return 6.0 * c3 * x + 2.0 * c2

    return {
        "f": f,
        "df": df,
        "d2f": d2f,
        "description": f"random cubic (seed={seed})",
        "recommended_bc": "clamped",
        "seed": seed,
    }


def make_sin():
    def f(x):
        return np.sin(_as_array(x))

    def df(x):
        return np.cos(_as_array(x))

    return {
        "f": f,
        "df": df,
        "description": "sin(x)",
        "recommended_bc": "clamped",
    }


def make_exp():
    def f(x):
        return np.exp(_as_array(x))

    def df(x):
        return np.exp(_as_array(x))

    return {
        "f": f,
        "df": df,
        "description": "exp(x)",
        "recommended_bc": "clamped",
    }


def make_rational():
    """Runge-type 1/(1+x^2)."""

    def f(x):
        x = _as_array(x)
        return 1.0 / (1.0 + x * x)

    def df(x):
        x = _as_array(x)
        u = 1.0 + x * x
        return -2.0 * x / (u * u)

    return {
        "f": f,
        "df": df,
        "description": "1/(1+x^2)",
        "recommended_bc": "clamped",
    }


def make_piecewise_cubic(breakpoints, seed=0):
    """
    C0 piecewise cubic on the given strictly increasing breakpoints.
    On [a_i, b_i], p(x) = c3*(x-a_i)^3 + c2*(x-a_i)^2 + c1*(x-a_i) + c0 with random c1,c2,c3
    and c0 chosen so values match at interior knots.
    """
    b = np.asarray(breakpoints, dtype=np.float64).ravel()
    if b.size < 2 or np.any(np.diff(b) <= 0):
        raise ValueError("breakpoints must be strictly increasing with length >= 2")
    rng = np.random.default_rng(seed)
    M = b.size - 1
    coeffs = []
    end_value = None
    for i in range(M):
        ai, bi = b[i], b[i + 1]
        h = bi - ai
        c1, c2, c3 = rng.standard_normal(3)
        if i == 0:
            c0 = float(rng.standard_normal())
        else:
            c0 = end_value
        end_value = ((c3 * h + c2) * h + c1) * h + c0
        coeffs.append(np.array([c3, c2, c1, c0], dtype=np.float64))

    def f(x):
        x = _as_array(x)
        flat = x.ravel()
        out = np.empty_like(flat, dtype=np.float64)
        for k in range(flat.size):
            xv = flat[k]
            j = int(np.searchsorted(b, xv, side="right") - 1)
            j = np.clip(j, 0, M - 1)
            aj = b[j]
            c3, c2, c1, c0 = coeffs[j]
            t = xv - aj
            out[k] = ((c3 * t + c2) * t + c1) * t + c0
        return out.reshape(x.shape)

    def df(x):
        x = _as_array(x)
        flat = x.ravel()
        out = np.empty_like(flat, dtype=np.float64)
        for k in range(flat.size):
            xv = flat[k]
            j = int(np.searchsorted(b, xv, side="right") - 1)
            j = np.clip(j, 0, M - 1)
            aj = b[j]
            c3, c2, c1, _ = coeffs[j]
            t = xv - aj
            out[k] = (3.0 * c3 * t + 2.0 * c2) * t + c1
        return out.reshape(x.shape)

    return {
        "f": f,
        "df": df,
        "description": f"piecewise cubic C0 (seed={seed}, {M} pieces)",
        "recommended_bc": "clamped",
        "piecewise_breakpoints": b,
        "seed": seed,
    }


def make_low_degree_poly_for_barycentric(degree, seed=1):
    """
    Random polynomial of given degree (0 <= degree <= 20).
    Barycentric interpolation on n nodes is exact for degree <= n-1.
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    rng = np.random.default_rng(seed)
    coeffs = rng.standard_normal(degree + 1)

    def f(x):
        return np.polyval(coeffs, _as_array(x))

    def df(x):
        return np.polyval(np.polyder(coeffs), _as_array(x))

    return {
        "f": f,
        "df": df,
        "description": f"random poly degree {degree} (seed={seed})",
        "recommended_bc": "clamped",
        "poly_degree": degree,
        "poly_coeffs": coeffs,
        "seed": seed,
    }


_REGISTRY = {
    "linear": make_linear,
    "quadratic": make_quadratic,
    "cubic": make_cubic_poly,
    "random_cubic": make_random_cubic_clamped,
    "sin": make_sin,
    "exp": make_exp,
    "rational": make_rational,
}


def list_function_names():
    return sorted(_REGISTRY.keys())


def get_function_spec(name, seed=0, poly_degree=5, piecewise_breakpoints=None):
    """
    Build a function spec by name.

    piecewise_breakpoints: required for name 'piecewise_cubic' (ndarray).
    poly_degree: used for name 'poly' (random polynomial of that degree).
    """
    if name == "piecewise_cubic":
        if piecewise_breakpoints is None:
            raise ValueError("piecewise_cubic requires piecewise_breakpoints")
        return make_piecewise_cubic(piecewise_breakpoints, seed=seed)
    if name == "poly":
        return make_low_degree_poly_for_barycentric(poly_degree, seed=seed)
    if name == "random_cubic":
        return make_random_cubic_clamped(seed=seed)
    maker = _REGISTRY.get(name)
    if maker is None:
        raise ValueError(
            f"Unknown function {name!r}. Choose from {list_function_names()} or poly, piecewise_cubic."
        )
    return maker()


def spline_bc_values(bc_type, spec, a, b):
    """Return bc_values tuple for spline setup, or None."""
    if bc_type == "natural" or bc_type == "not_a_knot":
        return None
    if bc_type == "clamped":
        df = spec.get("df")
        if df is None:
            raise ValueError("clamped BC requires df in function spec")
        return (float(df(a)), float(df(b)))
    if bc_type == "curvature":
        d2f = spec.get("d2f")
        if d2f is None:
            raise ValueError("curvature BC requires d2f (e.g. cubic / poly specs)")
        return (float(d2f(a)), float(d2f(b)))
    raise ValueError(f"Unknown bc_type: {bc_type}")
