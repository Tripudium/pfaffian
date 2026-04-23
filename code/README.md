# Numerical test of the BKK scaling bound

This directory contains a small Python script that numerically verifies the
$O(w^n)$ upper bound on the Gauss-map degree from **Proposition 4.11** (the
single-layer sigmoid-classifier case) of the paper, and the matching
$\Omega(w^n)$ lower bound of **Remark 4.12**, for $n = 2$.

## What is tested

For a single-hidden-layer sigmoid network
$$
  f(x) \;=\; c_0 + \sum_{k=1}^w d_k\,\sigma(a_k\cdot x + b_k),
  \qquad x\in\R^2,
$$
with rational (small-integer) weight vectors $a_k\in\Z^2$ of sup-norm at most
$L$, we count the number of *Gauss-map preimages* of $e_2 = (0,1)$ on the
zero set $V = \{f=0\}$:
$$
  \#\,\bigl\{x\in\R^2 \;:\; f(x) = 0,\ \partial f/\partial x_1\,(x) = 0\bigr\}.
$$
Proposition 4.11 bounds this by $C(n,L)\cdot w^n = O(w^2)$; Remark 4.12 gives a
matching lower bound $\Omega(w^2)$.

## How counting works

The script uses a 2-D grid and extracts the zero-level contours of $f$ via
`matplotlib.contour`, then evaluates $\partial f/\partial x_1$ along each
contour and counts sign changes (each sign change corresponds to one Gauss-map
preimage). This is accurate as long as the grid resolves the zero-set
topology; see caveats below.

## Running

Dependencies: `numpy`, `scipy`, `matplotlib` (no `scikit-image` needed).

```bash
# Structured construction that saturates the Omega(w^n) lower bound
python3 test_bkk_bound.py --mode grid \
    --widths 4 6 8 12 16 24 32 48 64 --trials 1 \
    --plot bkk_scaling_grid.png

# Random small-integer weights (does not saturate the bound)
python3 test_bkk_bound.py --mode random \
    --widths 8 16 32 64 128 --trials 20 \
    --plot bkk_scaling_random.png
```

## Results

| width $w$ | grid (worst-case) | random (typical, max over 20 trials) |
|-----------|-------------------|--------------------------------------|
| 4         | 2                 | —                                    |
| 8         | 8                 | 5                                    |
| 16        | 40                | 9                                    |
| 32        | 160               | 7                                    |
| 64        | 736               | 9                                    |
| 128       | —                 | 7                                    |

Log-log fit for the **grid mode** over widths $\{4,\ldots,64\}$: fitted slope
$\approx 2.10$, consistent with the theoretical $\Theta(w^2)$.

The **random mode** stays bounded by a small constant over the tested range of
$w$; this is expected, since random rational weights generically produce a
simple zero set with $O(1)$ Gauss-map points. The $\Omega(w^n)$ rate is
achieved only by careful constructions (see the `grid` mode, which puts the
weight vectors in two orthogonal clusters and alternates output signs to
produce a checkerboard-like zero set).

## Construction details (grid mode)

The `grid` construction takes $m = w/2$ sigmoids with $a_k = (L,0)$ and
alternating $d_k = \pm 1$, with biases $b_j = -L p_j$ for transition points
$p_j$ spaced at distance $4/L$ across $[-x_{\max}, x_{\max}]$; mirror for the
other $m$ sigmoids with $a_k = (0,L)$. This makes
$f_1(x_1) = \sum (-1)^{j+1}\sigma(L(x_1-p_j))$ approximately a square wave in
$\{0,1\}$ with $\sim m$ plateaus, and likewise $f_2(x_2)$. With $c_0=-1$, the
zero set $\{f_1+f_2=1\}$ is an XOR / checkerboard pattern with
$\Theta(m^2) = \Theta(w^2)$ cells, each contributing 2 Gauss-map points.
The transition spacing is scaled with $w$ so the square-wave approximation
remains valid as $w$ grows.

## Caveats

- The contour + sign-change counter is accurate when the zero set is
  resolved by the grid; for very large $w$ the checkerboard cells shrink
  and the script under-counts (this is why the fit uses $w\le 64$ only).
- Near-saturation points where $f$ barely crosses zero are sensitive to
  the grid resolution; the `--grid-n` flag controls this (default 801).
- The bound $C(n,L)\cdot w^n$ has a constant that grows as $(2L)^n$; for
  $L=3$ and $n=2$ the theoretical prefactor is $\le 2\cdot 2! \cdot 36 = 144$.
  Our grid-mode data at $w=64$ gives $736/64^2 \approx 0.18$, well below
  this prefactor, so the bound is not saturated as a constant but the
  exponent is.
