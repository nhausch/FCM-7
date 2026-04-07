"""
Pytest mirror of key interpolation validation checks.

Run from FCM-7:  python3 -m pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_INTERP = Path(__file__).resolve().parents[1] / "interpolation"
if str(_INTERP) not in sys.path:
    sys.path.insert(0, str(_INTERP))

from barycentric_form1 import barycentric1_eval, setup_barycentric1
from meshes import build_mesh
from piecewise_newton import divided_differences_newton, newton_horner, piecewise_newton_eval, setup_piecewise_newton
from spline1 import setup_spline1, spline1_eval
from spline2 import setup_spline2, spline2_eval
from test_functions import get_function_spec, spline_bc_values


RTOL = 1e-10
ATOL = 1e-12
SPLINE12_TOL = 50 * np.finfo(np.float64).eps


def test_divided_differences_hermite_cubic():
    def f(t):
        return t**3 - t

    def fp(t):
        return 3 * t * t - 1

    a, b = -0.3, 0.7
    z = np.array([a, a, b, b], dtype=np.float64)
    c = divided_differences_newton(z, f, fp=fp, dtype=np.float64)
    for t in np.linspace(a, b, 20):
        p = newton_horner(t, z, c, dtype=np.float64)
        assert abs(p - f(t)) < ATOL


def test_spline1_agrees_spline2_clamped_cubic():
    a, b = -1.0, 1.0
    n = 11
    x = build_mesh("uniform", a, b, n)
    spec = get_function_spec("cubic")
    f = spec["f"]
    bc_values = spline_bc_values("clamped", spec, a, b)
    sp1 = setup_spline1(x, f, bc_type="clamped", bc_values=bc_values)
    sp2 = setup_spline2(x, f, bc_type="clamped", bc_values=bc_values)
    xx = np.linspace(a, b, 400)
    y1 = spline1_eval(xx, sp1)
    y2 = spline2_eval(xx, sp2, extrapolate=False)
    assert np.max(np.abs(y1 - y2)) < SPLINE12_TOL


def test_natural_spline_exact_linear():
    a, b = 0.25, 1.75
    n = 15
    x = build_mesh("cheb2", a, b, n)
    spec = get_function_spec("linear")
    f = spec["f"]
    sp1 = setup_spline1(x, f, bc_type="natural", bc_values=None)
    sp2 = setup_spline2(x, f, bc_type="natural", bc_values=None)
    xx = np.linspace(a, b, 300)
    truth = f(xx)
    assert np.max(np.abs(spline1_eval(xx, sp1) - truth)) < 1e-9
    assert np.max(np.abs(spline2_eval(xx, sp2, extrapolate=False) - truth)) < 1e-9


def test_clamped_spline_matches_cubic_truth():
    a, b = -1.0, 1.0
    n = 12
    x = build_mesh("uniform", a, b, n)
    spec = get_function_spec("cubic")
    f = spec["f"]
    bc_values = spline_bc_values("clamped", spec, a, b)
    sp1 = setup_spline1(x, f, bc_type="clamped", bc_values=bc_values)
    xx = np.linspace(a, b, 500)
    err = spline1_eval(xx, sp1) - f(xx)
    assert np.max(np.abs(err)) < 1e-10


def test_barycentric_exact_low_degree_poly():
    a, b = -0.5, 0.5
    n = 10
    x = build_mesh("uniform", a, b, n)
    spec = get_function_spec("poly", seed=2, poly_degree=n - 2)
    f = spec["f"]
    gamma, y = setup_barycentric1(x, f)
    xx = np.linspace(a, b, 200)
    p = barycentric1_eval(xx, x, gamma, y)
    assert np.max(np.abs(p - f(xx))) < 1e-8


def test_piecewise_cubic_exact_on_panels():
    a, b = 0.0, 2.0
    n = 5
    x = build_mesh("uniform", a, b, n)
    spec = get_function_spec("piecewise_cubic", seed=42, piecewise_breakpoints=x)
    f = spec["f"]
    breakpoints, z_list, coeffs_list = setup_piecewise_newton(
        a,
        b,
        n - 1,
        f,
        degree=3,
        breakpoint_mesh="uniform",
        local_nodes="uniform",
    )
    xx = np.linspace(a, b, 300)
    pn = piecewise_newton_eval(xx, breakpoints, z_list, coeffs_list)
    assert np.max(np.abs(pn - f(xx))) < 1e-9


@pytest.mark.parametrize("mesh_type", ["uniform", "cheb1", "cheb2"])
def test_spline12_random_cubic_clamped(mesh_type):
    a, b = -1.0, 1.0
    n = 13
    x = build_mesh(mesh_type, a, b, n)
    spec = get_function_spec("random_cubic", seed=7)
    f = spec["f"]
    bc_values = spline_bc_values("clamped", spec, a, b)
    sp1 = setup_spline1(x, f, bc_type="clamped", bc_values=bc_values)
    sp2 = setup_spline2(x, f, bc_type="clamped", bc_values=bc_values)
    xx = np.linspace(float(x[0]), float(x[-1]), 250)
    assert np.max(np.abs(spline1_eval(xx, sp1) - spline2_eval(xx, sp2, extrapolate=False))) < SPLINE12_TOL * 100
