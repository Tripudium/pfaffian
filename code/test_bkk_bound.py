#!/usr/bin/env python3
"""
Numerical test of the BKK bound md(V) = O(w^n) from Proposition 4.11 of the
Pfaffian tubular-neighbourhood paper.

We take n = 2 and a single-hidden-layer sigmoid network of varying width w,
    f(x) = c_0 + sum_{k=1}^w d_k * sigma(a_k . x + b_k),
with rational (small-integer) weights a_k in Z^2 bounded by L in sup-norm.
For the regular Gauss-map value v = e_2 = (0, 1), the Gauss-map preimage on
V = {f = 0} is
    { x in R^2 : f(x) = 0,  df/dx_1 (x) = 0 }.
Proposition 4.11 bounds its cardinality by C(n, L) * w^n = C(2, L) * w^2.

This script counts those points by:
  (i)  extracting the zero-level contours of F = f using matplotlib, which
       returns polyline approximations of the curve V on a dense grid;
  (ii) evaluating G = df/dx_1 along each contour and counting sign changes.
Each sign change of G on a contour of F corresponds to one solution of the
Gauss-map system (generically transverse). We aggregate over trials and
widths, then fit a power law log(count) ~ slope * log(w) + const.

Usage:
    python3 test_bkk_bound.py
    python3 test_bkk_bound.py --widths 4 8 16 32 --trials 5 --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import expit as sigmoid


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


def sample_network(w: int, seed: int, L: int = 3, mode: str = "random"):
    """Generate a single-layer sigmoid net for the scaling test.

    Three modes are available:

    - "random": integer weights in [-L, L]^2, random biases, alternating d_k.
      Typical but does not saturate the Omega(w^n) lower bound.

    - "grid": a structured construction approximating the lower-bound
      configuration of Remark 4.12. Half the sigmoids have a_k = (1, 0)
      with biases evenly spaced across the x_1-range, and half have
      a_k = (0, 1) with biases across x_2. With alternating d_k signs the
      zero set V = {f=0} develops ~ (w/2)^2 = O(w^2) small components,
      each contributing at least 2 Gauss-map points (top/bottom tangents).
      This is what the theoretical lower bound predicts.

    - "gridrot": the grid construction with an extra rotation applied to
      tilt the two families so they are no longer axis-aligned. Useful to
      check generic position.
    """
    rng = np.random.default_rng(seed)
    if mode == "random":
        A = rng.integers(-L, L + 1, size=(w, 2))
        mask = np.all(A == 0, axis=1)
        while mask.any():
            A[mask] = rng.integers(-L, L + 1, size=(mask.sum(), 2))
            mask = np.all(A == 0, axis=1)
        b = rng.uniform(-2.0, 2.0, size=w)
        signs = np.where(np.arange(w) % 2 == 0, 1.0, -1.0)
        d = signs * rng.uniform(0.8, 1.2, size=w)
        c0 = 0.0
        return A.astype(float), b, d, c0

    # Structured "grid" construction, approximating the lower bound.
    #
    # We build two independent square-wave-like functions
    #   f_1(x_1) = sum_{j=1}^m (-1)^{j+1} sigma(L(x_1 - p_j))
    # and similarly f_2(x_2), each oscillating between approximately 0 and 1
    # as x crosses the transition points p_j. Setting f = f_1 + f_2 + c_0 with
    # c_0 = -1 gives f = 0 on an XOR-like checkerboard with ~m^2 cells;
    # boundaries of these cells contribute ~2m^2 Gauss-map points.
    m1 = w // 2
    m2 = w - m1
    # Transition points spaced evenly, with spacing >> 1/L so the square waves
    # are well-resolved. Transitions have width ~ 1/L; we require spacing of
    # at least 4/L between adjacent transitions, so x_max scales linearly in m.
    spacing = 4.0 / L
    x_max = max(3.0, 0.5 * max(m1, m2) * spacing)
    p = np.linspace(-x_max, x_max, m1) if m1 > 1 else np.array([0.0])
    q = np.linspace(-x_max, x_max, m2) if m2 > 1 else np.array([0.0])
    # Weight vectors: (L, 0) for the x_1-family, (0, L) for the x_2-family.
    A1 = np.tile(np.array([[L, 0.0]]), (m1, 1))
    A2 = np.tile(np.array([[0.0, L]]), (m2, 1))
    A = np.vstack([A1, A2])
    # sigma(L(x_i - p_j)) = sigma(L x_i - L p_j)  =>  b_j = -L p_j.
    b = np.concatenate([-L * p, -L * q])
    signs1 = np.where(np.arange(m1) % 2 == 0, 1.0, -1.0)
    signs2 = np.where(np.arange(m2) % 2 == 0, 1.0, -1.0)
    d = np.concatenate([signs1, signs2])
    # c_0 sets the target level; -1 gives XOR / checkerboard zero set.
    c0 = -1.0
    if mode == "gridrot":
        theta = 0.3
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        A = A @ R.T
    return A, b, d, c0


def eval_f(X: np.ndarray, Y: np.ndarray, A, b, d, c0) -> np.ndarray:
    pts = np.stack([X, Y], axis=-1)  # (H, W, 2)
    z = pts @ A.T + b                # (H, W, w)
    return c0 + np.sum(d * sigmoid(z), axis=-1)


def eval_df_dx1(X: np.ndarray, Y: np.ndarray, A, b, d) -> np.ndarray:
    pts = np.stack([X, Y], axis=-1)
    z = pts @ A.T + b
    return np.sum(d * sigmoid_prime(z) * A[:, 0], axis=-1)


def count_gauss_points(A, b, d, c0, range_lim: float = 6.0,
                       grid_n: int = 801) -> int:
    """Count Gauss-map preimages by contour + sign-change tracing."""
    xs = np.linspace(-range_lim, range_lim, grid_n)
    ys = np.linspace(-range_lim, range_lim, grid_n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    F = eval_f(X, Y, A, b, d, c0)
    G = eval_df_dx1(X, Y, A, b, d)

    # Interpolator for G at arbitrary (x, y).
    G_interp = RegularGridInterpolator(
        (ys, xs), G, bounds_error=False, fill_value=0.0
    )

    # Extract 0-level contours of F using matplotlib (returns polylines).
    # We use a throwaway figure to avoid state leakage.
    fig_tmp = plt.figure()
    try:
        cs = plt.contour(X, Y, F, levels=[0.0])
        segments = []
        # Modern matplotlib API (>=3.8): cs.allsegs[0] is a list of (N,2) arrays.
        if hasattr(cs, "allsegs"):
            segments = cs.allsegs[0]
        else:
            for coll in getattr(cs, "collections", []):
                for path in coll.get_paths():
                    segments.append(path.vertices)
    finally:
        plt.close(fig_tmp)

    count = 0
    for seg in segments:
        if len(seg) < 2:
            continue
        # Evaluate G along this contour polyline.
        pts = np.column_stack([seg[:, 1], seg[:, 0]])  # (y, x) for interp
        g_vals = G_interp(pts)
        # Count sign changes; ignore exact zeros to avoid double-counting.
        nonzero = g_vals[np.abs(g_vals) > 1e-14]
        if len(nonzero) < 2:
            continue
        signs = np.sign(nonzero)
        sc = int(np.sum(np.diff(signs) != 0))
        # If the contour is closed, add the wrap-around crossing.
        is_closed = np.allclose(seg[0], seg[-1], atol=1e-8)
        if is_closed and signs[0] != signs[-1]:
            sc += 1
        count += sc
    return count


def run_experiment(widths, trials: int, L: int, range_lim: float,
                   grid_n: int, mode: str, auto_range: bool = True):
    results: dict[int, list[int]] = {}
    for w in widths:
        counts = []
        for seed in range(trials):
            A, b, d, c0 = sample_network(w, seed, L=L, mode=mode)
            # Auto-adjust the plot range to bracket the transition region
            # (relevant for the grid construction, where transitions spread as w grows).
            if auto_range:
                bias_reach = np.max(np.abs(b)) / L + 2.0 / L + 1.0
                rl = max(range_lim, float(bias_reach))
            else:
                rl = range_lim
            n = count_gauss_points(A, b, d, c0, range_lim=rl, grid_n=grid_n)
            counts.append(n)
        results[w] = counts
        print(f"w = {w:3d}  counts = {counts}  "
              f"median = {int(np.median(counts))}  max = {max(counts)}")
    return results


def fit_power(widths, counts):
    """Return slope, intercept for log(counts) = slope * log(w) + intercept."""
    counts = np.asarray(counts, dtype=float)
    ws = np.asarray(widths, dtype=float)
    mask = counts > 0
    if mask.sum() < 2:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(np.log(ws[mask]), np.log(counts[mask]), 1)
    return float(slope), float(intercept)


def make_plot(results, outpath: Path, L: int, mode: str = ""):
    widths = sorted(results.keys())
    max_counts = [max(results[w]) for w in widths]
    med_counts = [int(np.median(results[w])) for w in widths]

    slope_max, _ = fit_power(widths, max_counts)
    slope_med, _ = fit_power(widths, med_counts)

    plt.figure(figsize=(6.2, 4.5))
    ws_arr = np.array(widths, dtype=float)
    plt.loglog(ws_arr, max_counts, "o-", label=f"max over trials (slope {slope_max:.2f})")
    plt.loglog(ws_arr, med_counts, "s-", label=f"median (slope {slope_med:.2f})")
    # Reference slope-2 line anchored at the median at the smallest width.
    ref_c = med_counts[0] / (ws_arr[0] ** 2)
    plt.loglog(ws_arr, ref_c * ws_arr ** 2, "k--", alpha=0.5,
               label="reference slope 2 (theory)")
    plt.xlabel("hidden-layer width  $w$")
    plt.ylabel("# Gauss-map points of $V = \\{f=0\\}$")
    suffix = f", mode={mode}" if mode else ""
    plt.title(f"BKK scaling test ($n=2$, $L={L}${suffix})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"wrote plot: {outpath}")
    return slope_max, slope_med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--widths", type=int, nargs="+",
                   default=[4, 8, 16, 24, 32, 48])
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--L", type=int, default=3,
                   help="sup-norm bound on integer weight entries")
    p.add_argument("--range-lim", type=float, default=6.0)
    p.add_argument("--grid-n", type=int, default=801)
    p.add_argument("--mode", type=str, default="grid",
                   choices=["random", "grid", "gridrot"],
                   help="network-sampling mode; 'grid' is the structured "
                        "construction that saturates the Omega(w^n) bound")
    p.add_argument("--plot", type=str, default="bkk_scaling.png")
    args = p.parse_args()

    print(f"widths = {args.widths}, trials = {args.trials}, "
          f"L = {args.L}, grid = {args.grid_n}, "
          f"mode = {args.mode}")
    results = run_experiment(
        widths=args.widths, trials=args.trials, L=args.L,
        range_lim=args.range_lim, grid_n=args.grid_n,
        mode=args.mode,
    )
    outpath = Path(__file__).parent / args.plot
    slope_max, slope_med = make_plot(results, outpath, L=args.L, mode=args.mode)
    print(f"\nFitted slopes:  max = {slope_max:.3f}   median = {slope_med:.3f}")
    print(f"Theoretical upper bound: slope <= 2 (Proposition 4.11).")
    print(f"Lower-bound construction: slope = 2 (Remark 4.12, Theta(w^n)).")


if __name__ == "__main__":
    main()
