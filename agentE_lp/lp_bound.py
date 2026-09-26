"""
lp_bound.py — Continuous radial LP lower bound for r_D(100).

RELAXATION (soundness in comments): fix radius r.  A disk of radius r with
center at distance d from the origin covers
  * area A(d) inside the unit disk D        (non-increasing in d), and
  * arc B(d, rho) on the circle of radius rho centered at the origin.
If 100 disks cover D, writing d_i for their center distances:
  (i)  sum_i A(d_i) >= pi                    (area)
  (ii) for EVERY rho in (0,1]:  sum_i B(d_i, rho) >= 2*pi   (each circle covered)
Both (i) and (ii) depend only on the multiset {d_i}.  We relax:
  - d_i are pooled into M intervals I_j, x_j := #{i: d_i in I_j} (real x_j >= 0),
  - each constraint uses Binf_{j,k} = inf{B(d,rho): d in I_j, rho in K_k},
  - the objective uses Asup_j = sup{A(d): d in I_j} = A(left endpoint).
Then every real configuration maps to a feasible LP point whose objective
upper-bounds its true area; hence LP(r) >= area of every covering configuration,
and LP(r) < pi  =>  covering with radius r is impossible  =>  r_D(100) > r.

KEY LEMMA (corner lemma): on any rectangle [d1,d2]x[rho1,rho2],
  inf B = 0                                    if the rectangle meets the dead
                                               zone {d >= rho+r} or {rho >= d+r},
  inf B = min(B at the four corners)           otherwise.
Proof facts: B has no interior critical point (dB/dd = 0 at d = sqrt(rho^2-r^2),
dB/drho = 0 at rho = sqrt(d^2-r^2); simultaneously impossible for r > 0), so the
inf lies on the boundary; along each edge B is unimodal (single peak at
sqrt(other^2 - r^2)) or a 2*pi plateau, so edge minima are at corners; and B = 0
on the whole dead zone, which the override captures exactly.
"""
import json
import numpy as np
from scipy.optimize import linprog

PI = float(np.pi)


def A(d, r):
    """Area inside D of a disk radius r centered at distance d (vectorized)."""
    d = np.asarray(d, float)
    out = np.empty_like(d)
    inner = d <= 1 - r
    outer = d >= 1 + r
    mid = ~inner & ~outer
    out[inner] = PI * r * r
    out[outer] = 0.0
    dm = d[mid]
    a1 = r * r * np.arccos(np.clip((dm * dm + r * r - 1) / (2 * dm * r), -1, 1))
    a2 = np.arccos(np.clip((dm * dm + 1 - r * r) / (2 * dm), -1, 1))
    a3 = 0.5 * np.sqrt(np.clip((-dm + r + 1) * (dm + r - 1) * (dm - r + 1) * (dm + r + 1), 0, None))
    out[mid] = a1 + a2 - a3
    return out


def B(d, rho, r):
    """Arc on the circle of radius rho covered by the disk (vectorized, outer)."""
    d = np.asarray(d, float)[:, None]          # (m,1)
    rho = np.asarray(rho, float)[None, :]      # (1,k)
    out = np.zeros(np.broadcast(d, rho).shape)
    full = (d + rho) <= r
    dead = (np.abs(d - rho) >= r) & ~full
    with np.errstate(divide="ignore", invalid="ignore"):
        gam = (d * d + rho * rho - r * r) / (2 * d * rho)
    alive = ~full & ~dead & (d > 0) & (rho > 0)
    out[full] = 2 * PI
    out[alive] = 2 * np.arccos(np.clip(gam[alive], -1, 1))
    return out


def Binf_matrix(d_edges, rho_edges, r):
    """Binf_{j,k} = inf of B over I_j x K_k  (corner lemma + dead override)."""
    dl = d_edges[:-1]
    dr = d_edges[1:]
    rl = rho_edges[:-1]
    rr = rho_edges[1:]
    c = np.minimum(np.minimum(
        np.minimum(B(dl, rl, r), B(dl, rr, r)),
        np.minimum(B(dr, rl, r), B(dr, rr, r))), 2 * PI)
    # dead-zone override: rectangle meets {d >= rho + r} or {rho >= d + r}
    dead = (dr[:, None] - rl[None, :] >= r) | (rr[None, :] - dl[:, None] >= r)
    return np.where(dead, 0.0, c)


def make_grids(r, M=360, K=360):
    # d-axis: [0, 1+r], denser near 1-r (regime change) and near rho-ridges
    d_edges = np.linspace(0.0, 1 + r, M + 1)
    # rho-axis: (0, 1], geometric near 0, dense near 1
    lo, hi = 1e-4, 1.0
    rho_edges = np.concatenate([
        np.geomspace(lo, 0.35, K // 3 + 1),
        np.linspace(0.35, 0.90, K // 3 + 1)[1:],
        np.linspace(0.90, hi, K - 2 * (K // 3) + 1)[1:],
    ])
    return d_edges, rho_edges


def lp_value(r, d_edges, rho_edges):
    Asup = A(d_edges[:-1], r)                       # left endpoints (A decreasing)
    Binf = Binf_matrix(d_edges, rho_edges, r)       # (M, K)
    M = len(Asup)
    res = linprog(
        c=-Asup,
        A_ub=-Binf.T, b_ub=-2 * PI * np.ones(Binf.shape[1]),
        A_eq=np.ones((1, M)), b_eq=[100.0],
        bounds=[(0.0, 100.0)] * M,
        method="highs",
    )
    if not res.success:
        return float("-inf")     # even the relaxation cannot cover -> excluded
    return float(-res.fun)


def cross_check_corner(rng, r, trials=30):
    """Dense-sampling validation of the corner lemma on alive rectangles."""
    worst = 0.0
    for _ in range(trials):
        d1, d2 = np.sort(rng.uniform(0.02, 1 + r, 2))
        rho1, rho2 = np.sort(rng.uniform(0.05, 1.0, 2))
        if (d2 - rho1 >= r) or (rho2 - d1 >= r):
            continue
        dd = np.linspace(d1, d2, 61)
        rr = np.linspace(rho1, rho2, 61)
        samp = B(dd, rr, r)
        corners = min(B(np.array([d1]), np.array([rho1]), r)[0, 0],
                      B(np.array([d1]), np.array([rho2]), r)[0, 0],
                      B(np.array([d2]), np.array([rho1]), r)[0, 0],
                      B(np.array([d2]), np.array([rho2]), r)[0, 0])
        corners = min(corners, 2 * PI)
        worst = max(worst, abs(min(samp.min(), 2 * PI) - corners))
    return worst


def certify(r_lo=0.100, r_hi=0.115, M=360, K=360):
    curve = []
    for r in np.arange(r_lo, r_hi + 1e-9, 0.001):
        v = lp_value(r, *make_grids(r, M, K))
        curve.append((round(float(r), 4), v))
        print(f"  LP({r:.4f}) = {v:.6f}   {'< pi -> excluded' if v < PI else '>= pi -> possible'}",
              flush=True)
    # bisection on the crossing
    a, b = r_lo, r_hi
    assert lp_value(a, *make_grids(a, M, K)) < PI, "sanity: LP(0.100) should exclude"
    assert lp_value(b, *make_grids(b, M, K)) > PI, "sanity: LP(0.115) should be feasible"
    for _ in range(36):
        m = 0.5 * (a + b)
        v = lp_value(m, *make_grids(m, M, K))
        if v < PI:
            a = m
        else:
            b = m
    return a, b, curve


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    w = cross_check_corner(rng, 0.108)
    print(f"corner-lemma dense-sampling check: worst |inf_sample - corner| = {w:.2e}",
          flush=True)
    assert w < 1e-7

    a, b, curve = certify()
    thr = a  # LP(a) < pi <= LP(b), interval length ~1e-9 after 36 bisections
    safe = float(np.floor(a * 1e6) / 1e6) - 5e-6
    print(f"\ncrossing in ({a:.9f}, {b:.9f})")
    print(f"CERTIFIED: r_D(100) > {safe:.6f}")
    json.dump(dict(threshold_exact=float(a), threshold_reported=safe,
                   curve=curve,
                   note="continuous radial LP; corner lemma verified by dense sampling"),
              open("agentE_lp/lp_certificate_self.json", "w"), indent=1)
