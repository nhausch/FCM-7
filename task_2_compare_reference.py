"""
Compare Task 2 interpolants to reference implementations.

- spline2 (natural B-spline) vs scipy.interpolate.CubicSpline(..., bc_type="natural")
- piecewise Newton degree-1 vs numpy.interp (same piecewise-linear target)
- diagnostic: default piecewise cubic vs SciPy natural cubic (not expected to match)

By default, error metrics use only abscissas in [t_0, t_last] (data interval).
Use --no-interior-only to include extrapolation region in the reported max/RMS.

Requires: pip install -r requirements.txt

Run from project root:

    python3 task_2_compare_reference.py
    python3 task_2_compare_reference.py --t-max 40 --no-interior-only
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None  # type: ignore[misc, assignment]

_ROOT = Path(__file__).resolve().parent
_INTERP = _ROOT / "interpolation"
if str(_INTERP) not in sys.path:
    sys.path.insert(0, str(_INTERP))

from piecewise_newton import (
    piecewise_newton_deriv1_eval,
    piecewise_newton_eval,
    setup_piecewise_newton,
)
from spline2 import setup_spline2, spline2_deriv1_eval, spline2_eval

from task_2_spline import T_NODES, Y_NODES, build_t_grid


def _make_f_pl(t_nodes: np.ndarray, y_nodes: np.ndarray) -> Callable[[float], float]:
    def f(t: float) -> float:
        return float(np.interp(t, t_nodes, y_nodes))

    return f


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))


def _mask_for_metrics(
    t_eval: np.ndarray, t_first: float, t_last: float, interior_only: bool
) -> np.ndarray:
    if interior_only:
        return (t_eval >= t_first) & (t_eval <= t_last)
    return np.ones(t_eval.shape, dtype=bool)


def main() -> None:
    if CubicSpline is None:
        print(
            "scipy is required for this script. Install with:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)

    p = argparse.ArgumentParser(
        description="Compare Task 2 spline/piecewise code to SciPy and NumPy references."
    )
    p.add_argument("--t-start", type=float, default=0.5, help="Grid start (default 0.5).")
    p.add_argument("--t-max", type=float, default=20.0, help="Grid end (default 20).")
    p.add_argument("--dt", type=float, default=0.5, help="Grid step (default 0.5).")
    p.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="If set, use t_start + k*dt for k = 0..n_rows-1.",
    )
    p.add_argument(
        "--interior-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict error metrics to [t_0, t_last] (default: true).",
    )
    p.add_argument(
        "--skip-diagnostic",
        action="store_true",
        help="Do not print piecewise cubic vs SciPy natural cubic distances.",
    )
    args = p.parse_args()

    t_nodes = T_NODES
    y_nodes = Y_NODES
    t_first = float(t_nodes[0])
    t_last = float(t_nodes[-1])
    a, b = t_first, t_last
    num_sub = int(t_nodes.size - 1)
    f_pl = _make_f_pl(t_nodes, y_nodes)

    if args.n_rows is not None:
        if args.n_rows < 1:
            raise SystemExit("n-rows must be >= 1")
        t_eval = args.t_start + np.arange(args.n_rows, dtype=np.float64) * args.dt
    else:
        t_eval = build_t_grid(args.t_start, args.t_max, args.dt)

    interior_only = bool(args.interior_only)
    m = _mask_for_metrics(t_eval, t_first, t_last, interior_only)
    if not np.any(m):
        raise SystemExit("no evaluation points in the selected interval; adjust grid or flags.")

    # SciPy natural cubic (reference); extrapolate=True to match typical task scripts
    cs = CubicSpline(t_nodes, y_nodes, bc_type="natural", extrapolate=True)
    y_scipy = cs(t_eval)
    yp_scipy = cs(t_eval, nu=1)

    y_arr = y_nodes.copy()

    def _f_mesh(_z: np.ndarray) -> np.ndarray:
        return y_arr

    sp = setup_spline2(t_nodes, _f_mesh, bc_type="natural")
    y_s2 = spline2_eval(t_eval, sp, extrapolate=True)
    yp_s2 = spline2_deriv1_eval(t_eval, sp, extrapolate=True)

    dy = np.abs(y_s2 - y_scipy)
    dyp = np.abs(yp_s2 - yp_scipy)
    print("A) spline2 vs SciPy CubicSpline (natural), extrapolate=True")
    print(f"    points in metric: {int(np.count_nonzero(m))} / {t_eval.size}")
    print(f"    max |y - y_ref|   = {np.max(dy[m]):.6e}    RMS = {_rms((y_s2 - y_scipy)[m]):.6e}")
    print(f"    max |y'-y'_ref|   = {np.max(dyp[m]):.6e}    RMS = {_rms((yp_s2 - yp_scipy)[m]):.6e}")

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
    y_pn1 = piecewise_newton_eval(t_eval, breakpoints, z_list, coeffs_list)
    y_np = np.interp(t_eval, t_nodes, y_nodes)
    d1 = np.abs(y_pn1 - y_np)
    print("\nB) piecewise Newton degree=1 vs numpy.interp (same PL data)")
    print(f"    max |y - y_ref|   = {np.max(d1[m]):.6e}    RMS = {_rms((y_pn1 - y_np)[m]):.6e}")

    if not args.skip_diagnostic:
        br3, z3, c3 = setup_piecewise_newton(
            a,
            b,
            num_sub,
            f_pl,
            degree=3,
            breakpoint_mesh="uniform",
            local_nodes="uniform",
            breakpoints=t_nodes,
        )
        y_pn3 = piecewise_newton_eval(t_eval, br3, z3, c3)
        yp_pn3 = piecewise_newton_deriv1_eval(t_eval, br3, z3, c3)
        print("\nC) diagnostic: piecewise Newton degree=3 vs SciPy natural cubic")
        print("    (different constructions; large gaps are expected.)")
        dc = np.abs(y_pn3 - y_scipy)
        dcp = np.abs(yp_pn3 - yp_scipy)
        print(f"    max |y - y_ref|   = {np.max(dc[m]):.6e}    RMS = {_rms((y_pn3 - y_scipy)[m]):.6e}")
        print(f"    max |y'-y'_ref|   = {np.max(dcp[m]):.6e}    RMS = {_rms((yp_pn3 - yp_scipy)[m]):.6e}")


if __name__ == "__main__":
    main()
