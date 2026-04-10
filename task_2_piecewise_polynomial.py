"""
Tabulate y(t), f(t), D(t) from discrete (t_i, y_i) via a piecewise Newton interpolant.

Breaks are the data abscissas (nonuniform mesh). The target function on [t_0, t_{n-1}]
is the piecewise-linear curve through the samples; each panel uses setup_piecewise_newton
on [t_i, t_{i+1}] (optional cubic Hermite with slopes from that PL curve). Evaluation
beyond the last node uses the first/last panel polynomial unless --no-extrapolate,
which sets NaN outside [t_0, t_{n-1}].

Relationships (continuous model): D(t) = exp(-t y(t)), f(t) = y(t) + t y'(t).

Run from the FCM-7 project root:

    python3 task_2_piecewise_polynomial.py
    python3 task_2_piecewise_polynomial.py --t-max 40
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_INTERP = _ROOT / "interpolation"
if str(_INTERP) not in sys.path:
    sys.path.insert(0, str(_INTERP))

from piecewise_newton import (
    piecewise_newton_deriv1_eval,
    piecewise_newton_eval,
    setup_piecewise_newton,
)

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


def _make_f_and_df(
    t_nodes: np.ndarray, y_nodes: np.ndarray
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Piecewise-linear values and slopes (per segment) for Hermite / high-degree panels."""

    def f(t: float) -> float:
        return float(np.interp(t, t_nodes, y_nodes))

    def df_pl(t: float) -> float:
        i = int(np.searchsorted(t_nodes, t, side="right") - 1)
        i = max(0, min(i, t_nodes.size - 2))
        return float((y_nodes[i + 1] - y_nodes[i]) / (t_nodes[i + 1] - t_nodes[i]))

    return f, df_pl


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tabulate piecewise-Newton y, f, D from discrete data."
    )
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
        help="Extrapolate past data with endpoint panels (default: true). "
        "Use --no-extrapolate for NaN outside [t_0, t_last].",
    )
    p.add_argument(
        "--degree",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help="Local polynomial degree on each segment when not --hermite (default: 3).",
    )
    p.add_argument(
        "--local-nodes",
        choices=("uniform", "cheb2"),
        default="uniform",
        help="Interior node layout on each subinterval (default: uniform).",
    )
    p.add_argument(
        "--hermite",
        action="store_true",
        help="Cubic Hermite on each segment matching PL f and f' at endpoints.",
    )
    args = p.parse_args()

    t_nodes = T_NODES
    y_nodes = Y_NODES
    num_sub = int(t_nodes.size - 1)
    a, b = float(t_nodes[0]), float(t_nodes[-1])
    f, df_pl = _make_f_and_df(t_nodes, y_nodes)

    if args.n_rows is not None:
        if args.n_rows < 1:
            raise SystemExit("n-rows must be >= 1")
        t_eval = args.t_start + np.arange(args.n_rows, dtype=np.float64) * args.dt
    else:
        t_eval = build_t_grid(args.t_start, args.t_max, args.dt)

    t_last = float(t_nodes[-1])
    t_first = float(t_nodes[0])
    extrapolate = bool(args.extrapolate)
    if np.max(t_eval) > t_last and not extrapolate:
        raise SystemExit(
            f"t grid extends past last node {t_last}; use --extrapolate or shorten the grid."
        )
    if np.max(t_eval) > t_last and extrapolate:
        print(
            f"note: some t exceed the last data node ({t_last}); "
            "values use the last-panel polynomial.\n",
            file=sys.stderr,
        )

    breakpoints, z_list, coeffs_list = setup_piecewise_newton(
        a,
        b,
        num_sub,
        f,
        degree=args.degree,
        df=df_pl if args.hermite else None,
        hermite=bool(args.hermite),
        breakpoint_mesh="uniform",
        local_nodes=args.local_nodes,
        breakpoints=t_nodes,
    )

    y_hat = piecewise_newton_eval(t_eval, breakpoints, z_list, coeffs_list)
    yp_hat = piecewise_newton_deriv1_eval(t_eval, breakpoints, z_list, coeffs_list)
    if not extrapolate:
        outside = (t_eval < t_first) | (t_eval > t_last)
        y_hat = y_hat.copy()
        yp_hat = yp_hat.copy()
        y_hat[outside] = np.nan
        yp_hat[outside] = np.nan

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
