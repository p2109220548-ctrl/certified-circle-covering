"""
lloyd_cover.py — Voronoi-cell 1-center iteration for min-max covering.

For each center i: compute its clipped Voronoi cell V_i ∩ D (Sutherland–Hodgman
clipping of a fine disk polygon against the 99 bisector half-planes), compute
the minimum enclosing circle of the cell, and jump the center toward that
1-center.  Every candidate move is accepted ONLY if the exact evaluator
(geometry.covering_radius) confirms strict improvement, so the process is
monotone non-increasing in R_D(C) regardless of polygon approximation error.
"""
import json
import time
import numpy as np
from geometry import covering_radius

N_DISK = 1440


def disk_polygon():
    th = np.linspace(0, 2 * np.pi, N_DISK, endpoint=False)
    return np.stack([np.cos(th), np.sin(th)], 1)


DISK = disk_polygon()


def clip_halfplane(poly, a, b, c):
    """keep {x : a*x0 + b*x1 <= c} (Sutherland–Hodgman)."""
    if len(poly) == 0:
        return poly
    f = a * poly[:, 0] + b * poly[:, 1] - c
    res = []
    n = len(poly)
    for k in range(n):
        fk, f2 = f[k], f[(k + 1) % n]
        if fk <= 0:
            res.append(poly[k])
        if (fk < 0 < f2) or (f2 < 0 < fk):
            t = fk / (fk - f2)
            res.append(poly[k] + t * (poly[(k + 1) % n] - poly[k]))
    return np.array(res) if res else np.zeros((0, 2))


def cell_polygon(C, i):
    """V_i ∩ D as a polygon."""
    poly = DISK
    ci = C[i]
    for j in range(len(C)):
        if j == i:
            continue
        d = C[j] - ci
        # keep {x : ||x-ci||^2 <= ||x-cj||^2} = {2(cj-ci).x <= ||cj||^2-||ci||^2}
        c = d @ d + 2 * (ci @ d)          # ||cj||^2 - ||ci||^2 = ||cj-ci||^2 + 2ci.(cj-ci)
        poly = clip_halfplane(poly, 2 * d[0], 2 * d[1], c)
        if len(poly) == 0:
            return poly
    return poly


def mec(pts, rng):
    """Minimum enclosing circle of a point set (Welzl, float)."""
    P = pts[np.random.permutation(len(pts))]
    c, r = None, -1.0
    for i in range(len(P)):
        p = P[i]
        if c is None or np.hypot(*(p - c)) > r + 1e-14:
            c, r = p.copy(), 0.0            # restart circle at p; inner loops grow it
            for j in range(i):
                q = P[j]
                if np.hypot(*(q - c)) > r + 1e-14:
                    c = (p + q) / 2
                    r = float(np.hypot(*(q - c)))
                    for k in range(j):
                        s = P[k]
                        if np.hypot(*(s - c)) > r + 1e-14:
                            c, r = _circum(p, q, s)
    return c, r


def _circum(a, b, c):
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-14:
        return (a + b + c) / 3, 1e9
    a2, b2, c2 = a @ a, b @ b, c @ c
    u = np.array([(a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
                  (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d])
    return u, float(np.hypot(*(u - a)))


def lloyd_polish(C, R, sweeps=8, lams=(1.0, 0.5, 0.2), seed=3, log=True):
    rng = np.random.default_rng(seed)
    C = C.copy()
    t0 = time.time()
    for sw in range(sweeps):
        improved = False
        for i in rng.permutation(len(C)):
            cell = cell_polygon(C, i)
            if len(cell) < 3:
                continue
            c_new, _ = mec(cell, rng)
            if not np.all(np.isfinite(c_new)):
                continue
            for lam in lams:
                C[i] = C[i] + lam * (c_new - C[i])
                R2 = covering_radius(C)
                if R2 < R - 1e-12:
                    R = R2
                    improved = True
                    break
                C[i] = C[i] - lam * (c_new - C[i])   # revert
        if log:
            print(f"  lloyd sweep {sw}: R = {R:.9f}   [{time.time()-t0:.0f}s]",
                  flush=True)
        if not improved:
            break
    return C, R


if __name__ == "__main__":
    data = json.load(open("centers_best.json"))
    C = np.array(data["centers"], float)
    R = covering_radius(C)
    print(f"start R = {R:.9f}", flush=True)
    C2, R2 = lloyd_polish(C, R)
    print(f"lloyd result: R = {R:.9f} -> {R2:.9f}")
    if R2 < R - 1e-9:
        import shutil
        shutil.copy("centers_best.json", "centers_best_backup.json")
        out = dict(n=len(C2), radius=float(R2), topology=data.get("topology"),
                   centers=[[float(a), float(b)] for a, b in C2])
        json.dump(out, open("centers_best.json", "w"), indent=1)
        print("centers_best.json UPDATED")
    else:
        print("no improvement kept")
