"""
Interpolation validation CLI.

Run from the FCM-7 directory:

  python main.py --mesh uniform --n 17 --function cubic --bc clamped --dense 2000

Adds interpolation/ on sys.path so local imports resolve (same as sibling modules).

Complexity (order of growth, informal): spline setup solves a dense (n+1)x(n+1) or
least-squares-style system — O(n^3) with numpy.linalg.solve; spline1_eval is O(1) per
point; spline2_eval (local support: span search + four cubic bases) is O(log n) per
point in mesh size n; spline2_eval_full sums all bases with recursive Cox-de Boor —
O(n) per point; barycentric1_eval is O(n) per point; piecewise Newton is
O(panels * degree) per point.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_INTERP = _ROOT / "interpolation"
if str(_INTERP) not in sys.path:
    sys.path.insert(0, str(_INTERP))

from barycentric_form1 import barycentric1_eval, setup_barycentric1
from meshes import build_mesh
from piecewise_newton import piecewise_newton_eval, setup_piecewise_newton
from spline1 import setup_spline1, spline1_eval
from spline2 import setup_spline2, spline2_eval
from test_functions import get_function_spec, list_function_names, spline_bc_values


def _check_n_bc(n_nodes: int, bc_type: str) -> None:
    if n_nodes < 2:
        raise SystemExit("n_nodes must be >= 2")
    if bc_type == "not_a_knot" and n_nodes < 4:
        raise SystemExit("not_a_knot requires n_nodes >= 4 (spline2 needs >= 4 knots)")


def _errors(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    diff = pred - truth
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff * diff)))
    return max_abs, rms


def run_validation(
    a: float,
    b: float,
    n_nodes: int,
    mesh_type: str,
    bc_type: str,
    function_name: str,
    seed: int,
    poly_degree: int,
    local_nodes: str,
    hermite: bool,
    dense_n: int,
    time_repeat: int,
    compare_all: bool,
) -> None:
    _check_n_bc(n_nodes, bc_type)
    x = build_mesh(mesh_type, a, b, n_nodes)
    num_subintervals = n_nodes - 1
    if function_name == "piecewise_cubic":
        spec = get_function_spec(
            "piecewise_cubic",
            seed=seed,
            piecewise_breakpoints=x,
        )
    else:
        spec = get_function_spec(
            function_name,
            seed=seed,
            poly_degree=poly_degree,
        )
    f = spec["f"]
    bc_values = spline_bc_values(bc_type, spec, a, b)

    df = spec.get("df") if hermite else None
    if hermite and df is None:
        raise SystemExit("--hermite requires a function spec with df (most named functions have it)")

    t_setup0 = time.perf_counter()
    sp1 = setup_spline1(x, f, bc_type=bc_type, bc_values=bc_values)
    sp2 = setup_spline2(x, f, bc_type=bc_type, bc_values=bc_values)
    gamma, y_nodes = setup_barycentric1(x, f)
    breakpoints, z_list, coeffs_list = setup_piecewise_newton(
        a,
        b,
        num_subintervals,
        f,
        degree=3,
        df=df,
        hermite=hermite,
        breakpoint_mesh=mesh_type,
        local_nodes=local_nodes,
    )
    t_setup = time.perf_counter() - t_setup0

    if not np.allclose(breakpoints, x, rtol=0, atol=1e-14):
        raise SystemExit("internal: breakpoints != spline nodes")

    # B-spline spline2_eval(..., extrapolate=False) is only defined on [x[0], x[-1]].
    # Chebyshev-1 nodes omit endpoints, so linspace(a,b) would lie outside the spline support.
    xa, xb = float(x[0]), float(x[-1])
    xx = np.linspace(xa, xb, dense_n, dtype=np.float64)
    truth = np.asarray(f(xx), dtype=np.float64).ravel()

    def timed_eval(label: str, fn):
        t0 = time.perf_counter()
        for _ in range(max(time_repeat, 1)):
            out = fn()
        elapsed = time.perf_counter() - t0
        return out, elapsed / max(time_repeat, 1)

    s1, t_e1 = timed_eval(
        "spline1",
        lambda: spline1_eval(xx, sp1),
    )
    s2, t_e2 = timed_eval(
        "spline2",
        lambda: spline2_eval(xx, sp2, extrapolate=False),
    )
    bar, t_eb = timed_eval(
        "barycentric1",
        lambda: barycentric1_eval(xx, x, gamma, y_nodes),
    )
    pn, t_ep = timed_eval(
        "piecewise_newton",
        lambda: piecewise_newton_eval(xx, breakpoints, z_list, coeffs_list),
    )

    nan2 = np.isnan(s2)
    if np.any(nan2):
        raise SystemExit("spline2_eval produced NaN inside [a,b]; check mesh")

    print(f"function: {spec.get('description', function_name)}")
    print(f"mesh: {mesh_type}, n_nodes={n_nodes}, domain=[{a}, {b}], bc={bc_type}")
    if xa > a + 1e-14 or xb < b - 1e-14:
        print(
            f"note: evaluation grid [{xa:.6g}, {xb:.6g}] (knot span); "
            f"mesh nodes do not reach full [a,b]"
        )
    print(
        f"piecewise: hermite={hermite}, local_nodes={local_nodes}, "
        f"breakpoints aligned with spline knots"
    )
    print(f"dense grid points: {dense_n}")
    print()

    max_s12, rms_s12 = _errors(s1, s2)
    print(f"spline1 vs spline2: max_abs={max_s12:.3e} rms={rms_s12:.3e}")

    if compare_all:
        for name, pred, t_ev in [
            ("spline1", s1, t_e1),
            ("spline2", s2, t_e2),
            ("barycentric1", bar, t_eb),
            ("piecewise_newton", pn, t_ep),
        ]:
            mx, rm = _errors(pred, truth)
            print(f"{name} vs truth: max_abs={mx:.3e} rms={rm:.3e}  (eval ~{t_ev*1e3:.3f} ms/round)")

    print()
    print(
        f"setup wall time (all methods once): {t_setup*1e3:.3f} ms  "
        f"(repeat={time_repeat} for eval timing)"
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Validate interpolation methods on a shared mesh.")
    p.add_argument("--a", type=float, default=-1.0, help="left endpoint")
    p.add_argument("--b", type=float, default=1.0, help="right endpoint")
    p.add_argument(
        "--n",
        type=int,
        default=17,
        dest="n_nodes",
        help="number of spline / barycentric nodes (breakpoints = same)",
    )
    p.add_argument(
        "--mesh",
        type=str,
        default="uniform",
        choices=["uniform", "cheb1", "cheb2"],
        help="mesh type for spline knots, barycentric nodes, and piecewise breakpoints",
    )
    p.add_argument(
        "--bc",
        type=str,
        default="clamped",
        choices=["natural", "clamped", "curvature", "not_a_knot"],
        help="spline boundary conditions",
    )
    p.add_argument(
        "--function",
        type=str,
        default="cubic",
        help=(
            "test function: "
            + ", ".join(list_function_names())
            + ", poly, piecewise_cubic, random_cubic"
        ),
    )
    p.add_argument("--seed", type=int, default=0, help="seed for random_cubic, poly, piecewise_cubic")
    p.add_argument(
        "--poly-degree",
        type=int,
        default=5,
        dest="poly_degree",
        help="degree for --function poly",
    )
    p.add_argument(
        "--local-nodes",
        type=str,
        default="uniform",
        choices=["uniform", "cheb2"],
        dest="local_nodes",
        help="panel node placement for piecewise Newton (endpoints plus interior)",
    )
    p.add_argument(
        "--hermite",
        action="store_true",
        help="use cubic Hermite piecewise Newton (f,f' at subinterval ends)",
    )
    p.add_argument("--dense", type=int, default=2000, help="dense evaluation grid size")
    p.add_argument(
        "--time-repeat",
        type=int,
        default=1,
        help="repeat eval timing loop this many times (average reported)",
    )
    p.add_argument(
        "--compare-all",
        action="store_true",
        default=True,
        help="print errors vs analytic truth for each method (default: on)",
    )
    p.add_argument(
        "--no-compare-all",
        action="store_false",
        dest="compare_all",
        help="only print spline1 vs spline2 agreement",
    )
    args = p.parse_args(argv)

    if args.a >= args.b:
        raise SystemExit("require a < b")

    run_validation(
        a=args.a,
        b=args.b,
        n_nodes=args.n_nodes,
        mesh_type=args.mesh,
        bc_type=args.bc,
        function_name=args.function,
        seed=args.seed,
        poly_degree=args.poly_degree,
        local_nodes=args.local_nodes,
        hermite=args.hermite,
        dense_n=args.dense,
        time_repeat=args.time_repeat,
        compare_all=args.compare_all,
    )


if __name__ == "__main__":
    main()
