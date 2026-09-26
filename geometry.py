"""
geometry.py — Exact evaluation of the covering radius

    R_D(C) = max_{x in D} min_i ||x - c_i||,   D = closed unit disk.

Correctness argument (full proof in REPORT.md, Theorem A):
g(x) = min_i ||x - c_i|| is 1-Lipschitz and piecewise smooth.  Its maximum over D
is attained at one of:
  (a) a Voronoi vertex of C lying in D  (= circumcenter of a Delaunay triangle),
  (b) a point of the boundary circle where two nearest centers tie:
        (c_i - c_j) . (cos th, sin th) = 0,
  (c) a point of the boundary circle maximizing a single center's distance:
        (cos th, sin th) = -c_i/||c_i||   (the "antipode" of c_i).
Every candidate is a point of D, so max over candidates <= R_D(C); and the true
argmax lies in the candidate set, so equality holds.  Degenerate / non-Delaunay
circumcenters are harmless: we evaluate g at them anyway, which can only
underestimate (never overestimate) R_D(C).
"""

import numpy as np
from scipy.spatial import Delaunay


def _boundary_candidates(C):
    """Tie points and antipode points on the unit circle (as 2D points).

    Tie between centers i,j on the circle: ||u - c_i|| = ||u - c_j|| with ||u||=1
      <=>  (c_i - c_j).u = (||c_i||^2 - ||c_j||^2)/2 =: beta,  ||u|| = 1.
    Solution: u = (beta/||D||^2) D ± sqrt(1 - beta^2/||D||^2) * perp(D)/||D||.
    """
    cands = []
    n = len(C)
    # antipodes
    d = np.hypot(C[:, 0], C[:, 1])
    nz = d > 1e-14
    if nz.any():
        cands.append(-C[nz] / d[nz, None])
    # pairwise tie points on the circle
    if n > 1:
        iu = np.triu_indices(n, 1)
        Dxy = C[iu[0]] - C[iu[1]]                      # (m,2) = c_i - c_j
        L2 = (Dxy ** 2).sum(1)
        beta = ((C[iu[0]] ** 2).sum(1) - (C[iu[1]] ** 2).sum(1)) / 2.0
        ok = L2 > 1e-14
        Dxy, L2, beta = Dxy[ok], L2[ok], beta[ok]
        q = beta / L2                                  # (m,)
        disc = 1.0 - q * beta                          # 1 - beta^2/L2
        ok2 = disc >= 0.0
        Dxy, L2, q, disc = Dxy[ok2], L2[ok2], q[ok2], disc[ok2]
        base = q[:, None] * Dxy                        # (m,2)
        perp = np.stack([-Dxy[:, 1], Dxy[:, 0]], 1) / np.sqrt(L2)[:, None]
        t = np.sqrt(disc)[:, None]
        cands.append(base + t * perp)
        cands.append(base - t * perp)
    return np.vstack(cands) if cands else np.zeros((0, 2))


def _g_values(P, C):
    """g(p) = min_i ||p - c_i|| for rows of P (fast matmul form)."""
    if len(P) == 0:
        return np.zeros(0)
    d2 = ((P * P).sum(1)[:, None] + (C * C).sum(1)[None, :]
          - 2.0 * (P @ C.T))
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def _voronoi_vertices_in_D(C):
    """Circumcenters of all Delaunay simplices, filtered to the closed disk."""
    try:
        dl = Delaunay(C)
    except Exception:
        return np.zeros((0, 2))
    tri = dl.simplices
    A, B, Cc = C[tri[:, 0]], C[tri[:, 1]], C[tri[:, 2]]
    d = 2.0 * (A[:, 0] * (B[:, 1] - Cc[:, 1])
               + B[:, 0] * (Cc[:, 1] - A[:, 1])
               + Cc[:, 0] * (A[:, 1] - B[:, 1]))
    m = np.abs(d) > 1e-13
    if not m.any():
        return np.zeros((0, 2))
    A, B, Cc, d = A[m], B[m], Cc[m], d[m]
    a2, b2, c2 = (A ** 2).sum(1), (B ** 2).sum(1), (Cc ** 2).sum(1)
    ux = (a2 * (B[:, 1] - Cc[:, 1]) + b2 * (Cc[:, 1] - A[:, 1])
          + c2 * (A[:, 1] - B[:, 1])) / d
    uy = (a2 * (Cc[:, 0] - B[:, 0]) + b2 * (A[:, 0] - Cc[:, 0])
          + c2 * (B[:, 0] - A[:, 0])) / d
    V = np.stack([ux, uy], axis=1)
    inside = (V ** 2).sum(1) <= 1.0 + 1e-9
    return V[inside]


def covering_radius(C, check_grid=False):
    """Exact R_D(C) up to float round-off (~1e-12)."""
    C = np.asarray(C, float)
    R = 0.0
    V = _voronoi_vertices_in_D(C)
    if len(V):
        R = max(R, float(_g_values(V, C).max()))
    P = _boundary_candidates(C)
    if len(P):
        R = max(R, float(_g_values(P, C).max()))
    if check_grid:  # paranoia cross-check, not used in the final numbers
        th = np.linspace(0, 2 * np.pi, 400001)[:-1]
        B = np.stack([np.cos(th), np.sin(th)], 1)
        gB = _g_values(B, C).max()
        assert gB <= R + 1e-6, f"grid found larger value {gB} > {R}"
    return R


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(30):
        n = rng.integers(3, 15)
        C = rng.uniform(-1.1, 1.1, size=(n, 2))
        R = covering_radius(C, check_grid=True)
        worst = max(worst, R)
    print(f"30 randomized cross-checks passed (grid <= exact + 1e-6). max R seen = {worst:.6f}")
