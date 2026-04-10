"""
Reference checks for Task 2 data: spline2 vs SciPy natural cubic; deg-1 vs numpy.interp.

Requires scipy for spline test (skipped if missing).
Run: python3 -m pytest tests/test_task2_reference.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INTERP = _ROOT / "interpolation"
for _p in (_ROOT, _INTERP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from piecewise_newton import piecewise_newton_eval, setup_piecewise_newton
from spline2 import setup_spline2, spline2_deriv1_eval, spline2_eval
from task_2_spline import T_NODES, Y_NODES


def test_spline2_matches_scipy_natural_cubic_interior():
    pytest.importorskip("scipy")
    from scipy.interpolate import CubicSpline

    t_nodes = T_NODES
    y_nodes = Y_NODES
    t_first, t_last = float(t_nodes[0]), float(t_nodes[-1])
    t_eval = np.linspace(t_first, t_last, 200, dtype=np.float64)

    y_arr = y_nodes.copy()

    def _f_mesh(_z: np.ndarray) -> np.ndarray:
        return y_arr

    sp = setup_spline2(t_nodes, _f_mesh, bc_type="natural")
    y_s2 = spline2_eval(t_eval, sp, extrapolate=True)
    yp_s2 = spline2_deriv1_eval(t_eval, sp, extrapolate=True)

    cs = CubicSpline(t_nodes, y_nodes, bc_type="natural", extrapolate=True)
    y_ref = cs(t_eval)
    yp_ref = cs(t_eval, nu=1)

    assert np.max(np.abs(y_s2 - y_ref)) < 1e-10
    assert np.max(np.abs(yp_s2 - yp_ref)) < 1e-10


def test_piecewise_newton_degree1_matches_numpy_interp():
    t_nodes = T_NODES
    y_nodes = Y_NODES
    a, b = float(t_nodes[0]), float(t_nodes[-1])
    num_sub = int(t_nodes.size - 1)

    def f_pl(t: float) -> float:
        return float(np.interp(t, t_nodes, y_nodes))

    breakpoints, z_list, coeffs_list = setup_piecewise_newton(
        a,
        b,
        num_sub,
        f_pl,
        degree=1,
        breakpoint_mesh="uniform",
        local_nodes="uniform",
        breakpoints=t_nodes,
    )

    rng = np.random.default_rng(0)
    xx = rng.uniform(t_nodes[0], t_nodes[-1], size=50).astype(np.float64)
    pn = piecewise_newton_eval(xx, breakpoints, z_list, coeffs_list)
    truth = np.interp(xx, t_nodes, y_nodes)
    assert np.max(np.abs(pn - truth)) < 1e-12
