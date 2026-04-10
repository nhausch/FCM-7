"""
Tabulate y(t), f(t), D(t) from discrete (t_i, y_i) via a natural cubic spline.

Uses the B-spline interpolant from spline2 (natural boundary conditions). For any
evaluation t beyond the last data abscissa, B-spline extrapolation is used (optional
via --no-extrapolate to emit NaN outside the data interval instead).

Relationships (continuous model): D(t) = exp(-t y(t)), f(t) = y(t) + t y'(t).

Run from the FCM-7 project root:

    python3 task_2.py
    python3 task_2.py --t-max 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_INTERP = _ROOT / "interpolation"
if str(_INTERP) not in sys.path:
    sys.path.insert(0, str(_INTERP))

from spline2 import setup_spline2, spline2_deriv1_eval, spline2_eval

# Problem data (nonuniform mesh).
T_NODES = np.array([0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
Y_NODES = np.array(
    [0.0552, 0.06, 0.0682, 0.0801, 0.0843, 0.0931, 0.0912, 0.0857],
    dtype=np.float64,
)


def build_t_grid(t_start: float, t_max: float, dt: float) -> np.ndarray:
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t_max < t_start:
        raise ValueError("t_max must be >= t_start")
    n = int(round((t_max - t_start) / dt)) + 1
    if n < 1:
        raise ValueError("empty grid")
    return np.linspace(t_start, t_start + (n - 1) * dt, n, dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="Tabulate spline-based y, f, D from discrete data.")
    p.add_argument(
        "--t-start",
        type=float,
        default=0.5,
        help="First tabulation abscissa (default: 0.5).",
    )
    p.add_argument(
        "--t-max",
        type=float,
        default=20.0,
        help="Last tabulation abscissa (default: 20, i.e. 40 rows with dt=0.5).",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Tabulation step (default: 0.5).",
    )
    p.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="If set, override t-max: use exactly n rows t_start, t_start+dt, ...",
    )
    p.add_argument(
        "--extrapolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extrapolate spline past data (default: true). Use --no-extrapolate for NaN outside data.",
    )
    args = p.parse_args()

    t_nodes = T_NODES
    y_nodes = Y_NODES

    if args.n_rows is not None:
        if args.n_rows < 1:
            raise SystemExit("n-rows must be >= 1")
        t_eval = args.t_start + np.arange(args.n_rows, dtype=np.float64) * args.dt
    else:
        t_eval = build_t_grid(args.t_start, args.t_max, args.dt)

    t_last = float(t_nodes[-1])
    extrapolate = bool(args.extrapolate)
    if np.max(t_eval) > t_last and not extrapolate:
        raise SystemExit(
            f"t grid extends past last node {t_last}; use --extrapolate or shorten the grid."
        )
    if np.max(t_eval) > t_last and extrapolate:
        print(
            f"note: some t exceed the last data node ({t_last}); "
            "values use B-spline extrapolation.\n",
            file=sys.stderr,
        )

    y_arr = y_nodes.copy()

    def _f_mesh(_z: np.ndarray) -> np.ndarray:
        return y_arr

    sp = setup_spline2(t_nodes, _f_mesh, bc_type="natural")

    y_hat = spline2_eval(t_eval, sp, extrapolate=extrapolate)
    yp_hat = spline2_deriv1_eval(t_eval, sp, extrapolate=extrapolate)
    f_hat = y_hat + t_eval * yp_hat
    d_hat = np.exp(-t_eval * y_hat)

    print(f"{'t':>10} {'y_hat':>14} {'y_prime':>14} {'f_hat':>14} {'D_hat':>14}")
    for i in range(t_eval.size):
        print(
            f"{t_eval[i]:10.4f} {y_hat[i]:14.6f} {yp_hat[i]:14.6f} "
            f"{f_hat[i]:14.6f} {d_hat[i]:14.6e}"
        )


if __name__ == "__main__":
    main()
