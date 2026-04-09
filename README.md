# FCM-7 Interpolation Validation CLI

This repository contains a small CLI (`main.py`) to compare multiple interpolation methods on a shared mesh:

- `spline1` (cubic interpolatory spline in second-derivative form)
- `spline2` (cubic interpolatory spline in B-spline coefficient form)
- `barycentric1`
- `piecewise_newton` (optionally Hermite)

## Quick start

Run from the `FCM-7` directory:

```bash
python3 main.py --mesh uniform --n 17 --function cubic --bc clamped --dense 2000
```

## Test command set

These all run *all* methods (`spline1`, `spline2`, `barycentric1`, `piecewise_newton`) on the same nodes and print errors + timing.

```bash
# Baseline (smooth function, moderate n)
python3 main.py --mesh uniform --n 17 --function cubic --bc clamped --dense 2000 --time-repeat 5

# Mesh sweep (same function/BC)
python3 main.py --mesh uniform --n 33 --function cubic --bc clamped --dense 4000 --time-repeat 3
python3 main.py --mesh cheb1   --n 33 --function cubic --bc clamped --dense 4000 --time-repeat 3
python3 main.py --mesh cheb2   --n 33 --function cubic --bc clamped --dense 4000 --time-repeat 3

# Boundary condition sweep (same mesh/function)
python3 main.py --mesh uniform --n 33 --function cubic --bc natural    --dense 4000 --time-repeat 3
python3 main.py --mesh uniform --n 33 --function cubic --bc clamped    --dense 4000 --time-repeat 3
python3 main.py --mesh uniform --n 33 --function cubic --bc curvature  --dense 4000 --time-repeat 3
python3 main.py --mesh uniform --n 33 --function cubic --bc not_a_knot --dense 4000 --time-repeat 3

# Polynomial exactness-style checks (poly degree < n is the “easy” regime)
python3 main.py --mesh uniform --n 17 --function poly --poly-degree 5 --bc clamped --dense 2000 --time-repeat 3
python3 main.py --mesh cheb2   --n 17 --function poly --poly-degree 9 --bc clamped --dense 2000 --time-repeat 3

# Random polynomial (repeatable via seed)
python3 main.py --mesh uniform --n 25 --function random_cubic --seed 0 --bc clamped --dense 3000 --time-repeat 3
python3 main.py --mesh uniform --n 25 --function random_cubic --seed 1 --bc clamped --dense 3000 --time-repeat 3

# Nonsmooth-ish target (piecewise cubic); good for seeing overshoot behavior
python3 main.py --mesh uniform --n 65 --function piecewise_cubic --seed 0 --bc clamped --dense 8000 --time-repeat 2
python3 main.py --mesh cheb2   --n 65 --function piecewise_cubic --seed 0 --bc clamped --dense 8000 --time-repeat 2

# Piecewise Newton variants (local nodes + Hermite vs non-Hermite)
python3 main.py --mesh uniform --n 33 --function cubic --bc clamped --local-nodes uniform --dense 4000 --time-repeat 3
python3 main.py --mesh uniform --n 33 --function cubic --bc clamped --local-nodes cheb2   --dense 4000 --time-repeat 3
python3 main.py --mesh uniform --n 33 --function cubic --bc clamped --hermite --dense 4000 --time-repeat 3

# Scaling / timing (bump n and dense; use --no-compare-all for less output)
python3 main.py --mesh uniform --n 129 --function cubic --bc clamped --dense 20000 --time-repeat 2 --no-compare-all
```

## Complexity notes (order of growth, informal)

Spline setup solves a dense \((n+1)\times(n+1)\) or least-squares-style system — \(O(n^3)\) with `numpy.linalg.solve`;
`spline1_eval` is \(O(1)\) per point; `spline2_eval` uses cubic B-spline local support (4 basis functions) and:
\(O(\log n)\) per point for unsorted `x_eval` (binary span search) but \(O(1)\) amortized per point when `x_eval` is
nondecreasing (span-walk fast path); `spline2_eval_full` sums all bases with recursive Cox-de Boor — \(O(n)\) per point;
`barycentric1_eval` is \(O(n)\) per point; piecewise Newton is \(O(\text{panels} \cdot \text{degree})\) per point.

