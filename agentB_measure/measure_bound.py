# -*- coding: utf-8 -*-
"""
Rigorous LOWER bound for r_D(100) via radial measure duality.

Method: if mu is a Borel probability measure on the unit disk D such that every
disk of radius r (center anywhere in the plane) has mu-mass <= 1/100, then
100 disks of radius r cannot cover D.  Hence r_D(100) >= r.

Family: mu = sum_k w_k * U(circle of radius t_k), 0<=t_k<=1, w_k>=0, sum w_k = 1.
Mass of B(c,r) with d = ||c||:  F(d) = sum_k w_k phi_k(d),  phi in [0,1] = fraction
of the circle of radius t_k lying inside B(c,r).

phi_k regimes (exact):
  d = 0            : 1 if t_k <= r else 0
  d + t_k <= r     : 1          (disk contains the circle)
  |d - t_k| >= r   : 0          (circle outside disk, or disk strictly inside circle)
  otherwise        : alpha/pi,  cos(alpha) = (t_k^2 + d^2 - r^2)/(2 t_k d)

Monotonicity (verified via alpha'(d) = -(d^2+r^2-t^2)/(d*sqrt((r^2-(d-t)^2)((d+t)^2-r^2)))):
  * t <= r: phi non-increasing on [0, inf) (equals 1 on [0, r-t], then decreases).
  * t >  r: phi = 0 on [0, t-r], strictly increasing on [t-r, sqrt(t^2-r^2)],
            strictly decreasing on [sqrt(t^2-r^2), t+r], 0 beyond.  UNIMODAL with
            peak at p = sqrt(t^2 - r^2)  (NOT at d = t; alpha'(d)=0 iff d^2=t^2-r^2).
=> For any interval [a,b] whose endpoints (and the peak p when t>r) are grid points,
   sup_{[a,b]} phi_k = max(phi_k(a), phi_k(b), [peak in [a,b]] phi_k(p)).
   With every tangency point |t_k +/- r| and every peak p_k inserted into the grid,
   endpoint values alone give the EXACT sup of each phi_k on each grid subinterval,
   hence  sup_{[0,D]} F <= max_j sum_k w_k max(Phi[j,k], Phi[j+1,k])   (rigorous).
"""
import numpy as np
from scipy.optimize import linprog, minimize
import json, time, sys, os

PI = np.pi
OUT = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)

# ---------------------------------------------------------------- phi (exact)
def phi(t, d, r):
    """Coverage fraction of circle radius t (center O) by disk radius r center at distance d."""
    d = abs(float(d))
    if t <= 0.0:
        return 1.0 if d <= r else 0.0
    if d + t <= r:
        return 1.0
    if abs(d - t) >= r:
        return 0.0
    g = (t * t + d * d - r * r) / (2.0 * t * d)
    g = min(1.0, max(-1.0, g))
    return np.arccos(g) / PI

def phi_mat(t_vec, d_arr, r):
    """Matrix Phi[i,k] = phi_k(d_i), vectorized."""
    t_vec = np.asarray(t_vec, float)
    d_arr = np.asarray(d_arr, float)
    T = t_vec[None, :]
    D = d_arr[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        g = (T * T + D * D - r * r) / (2.0 * T * D)
        alpha = np.arccos(np.clip(g, -1.0, 1.0))
    partial = (D + T > r) & (np.abs(D - T) < r) & (T > 0)
    out = np.where(D + T <= r, 1.0, np.where(partial, alpha / PI, 0.0))
    zc = t_vec <= 0
    if zc.any():
        out[:, zc] = (d_arr <= r).astype(float)[:, None]
    return out

def critical_points(t_vec, r, dom):
    """All critical locations: 0, tangencies |t_k - r|, t_k + r, peaks sqrt(t_k^2-r^2), domain end."""
    pts = [0.0, dom]
    for t in t_vec:
        if t <= 0:
            pts.append(min(r, dom))
            continue
        pts.append(min(t, dom))
        pts.append(min(abs(t - r), dom))
        pts.append(min(t + r, dom))
        if t > r:
            p = np.sqrt(t * t - r * r)
            if p < dom:
                pts.append(p)
    return np.unique(np.clip(pts, 0.0, dom))

def rigorous_sup(t_vec, w, r, h0=2e-3, cap=None, min_len=5e-10, max_rounds=300,
                 max_points=None):
    """Rigorous upper bound of sup_{d in [0,dom]} F(d).

    Grid = criticals (all tangencies |t_k-r|, t_k+r, all peaks sqrt(t_k^2-r^2), 0, dom)
    + uniform(h0).  Since every critical point of every phi_k is a grid point, each
    phi_k is MONOTONE on each grid subinterval [a,b].  Hence for any x in [a,b]:
        F(x) <= F(a) + sum_k w_k max(0, phi_k(b)-phi_k(a))     (one-sided from a)
        F(x) <= F(b) + sum_k w_k max(0, phi_k(a)-phi_k(b))     (one-sided from b)
    bound_j = min of the two  >= sup_{[a,b]} F, with conservatism <= sum_k w_k|phi_k(b)-phi_k(a)|
    = O(h) GLOBALLY (the two-sided min removes the sqrt-type slack that a plain
    max-endpoint bound suffers at tangency points).
    Refinement is by need: intervals with bound > cap (default: max-1e-12) are bisected
    while longer than min_len, with a hard point cap.  Rigorous at ANY refinement level.
    """
    t_vec = np.asarray(t_vec, float); w = np.asarray(w, float)
    K = len(t_vec)
    if max_points is None:
        max_points = max(150_000, 2_500_000 // max(K, 1))
    dom = float(np.max(t_vec)) + r
    pts = list(critical_points(t_vec, r, dom))
    pts += list(np.arange(0.0, dom, h0))
    pts = np.unique(np.clip(np.array(pts), 0.0, dom))
    for _ in range(max_rounds):
        Phi = phi_mat(t_vec, pts, r)
        Fa = Phi[:-1] @ w
        Fb = Phi[1:] @ w
        D = Phi[1:] - Phi[:-1]
        bounds = np.minimum(Fa + np.maximum(D, 0.0) @ w,
                            Fb + np.maximum(-D, 0.0) @ w)
        ub = bounds.max()
        thr = (ub - 1e-12) if cap is None else cap
        jx = np.where(bounds > thr)[0]
        lens = pts[jx + 1] - pts[jx]
        sel = jx[lens > min_len]
        if sel.size == 0 or pts.size + 2 * sel.size > max_points:
            break
        mid = (pts[sel] + pts[sel + 1]) / 2.0
        pts = np.unique(np.concatenate([pts, mid]))
    Phi = phi_mat(t_vec, pts, r)
    Fa = Phi[:-1] @ w
    Fb = Phi[1:] @ w
    D = Phi[1:] - Phi[:-1]
    ub = float(np.minimum(Fa + np.maximum(D, 0.0) @ w,
                          Fb + np.maximum(-D, 0.0) @ w).max()) + 5e-14
    return ub, dict(n_grid=int(pts.size), dom=dom, h0=h0)

# ---------------------------------------------------------------- LP feasibility
def lp_grid(t_vec, r, n_uniform=3000):
    dom = float(np.max(t_vec)) + r
    pts = list(critical_points(t_vec, r, dom))
    pts += list(np.linspace(0.0, dom, n_uniform))
    return np.unique(np.clip(np.array(pts), 0.0, dom))

def lp_feasible(t_vec, r, target, grid=None, n_uniform=3000, time_limit=4.0):
    """Is there w >= 0, sum w = 1 with F(d_j) <= target on the LP grid?

    Returns (True, w) feasible; (False, None) infeasible-or-unknown (time limit is
    treated as infeasible: conservative for the r-search, never overstates r).
    """
    if grid is None:
        grid = lp_grid(t_vec, r, n_uniform=n_uniform)
    Phi = phi_mat(t_vec, grid, r)
    K = len(t_vec)
    res = linprog(c=np.zeros(K), A_ub=Phi, b_ub=np.full(len(grid), target),
                  A_eq=np.ones((1, K)), b_eq=[1.0], bounds=[(0, None)] * K,
                  method='highs', options=dict(time_limit=time_limit, presolve=True))
    if res.status != 0:
        return False, None
    w = res.x / res.x.sum()
    return True, w

def best_r_for_t(t_vec, r_lo=0.09, r_hi=0.13, iters=26, n_uniform=3000, w_ret=False,
                 budget=35.0):
    """Bisect largest r with LP-feasible discretized system. Returns (r, w|None).

    Wall-clock budget: if exceeded mid-bisection, returns current best lo (safe).
    """
    t0 = time.time()
    target = 1.0 / 100.0
    lo, hi = r_lo, r_hi          # lo: feasible (assumed), hi: infeasible (assumed)
    ok, _ = lp_feasible(t_vec, lo, target)
    if not ok:
        return (0.0, None) if w_ret else 0.0
    ok_hi, _ = lp_feasible(t_vec, hi, target)
    if ok_hi:
        return (hi, None) if w_ret else hi
    best_w = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok, w = lp_feasible(t_vec, mid, target, n_uniform=n_uniform)
        if ok:
            lo, best_w = mid, w
        else:
            hi = mid
        if hi - lo < 2e-7 or time.time() - t0 > budget:
            break
    return (lo, best_w) if w_ret else lo

# ---------------------------------------------------------------- outer search
def patterns(K):
    u = (np.arange(K) + 0.5) / K
    cands = [
        u.copy(),
        np.sqrt(u),                      # denser near 1
        u ** 0.7,
        u ** 1.4,                        # denser near 0
        np.linspace(0.02, 0.995, K),
        np.linspace(0.25, 1.0, K),
        np.linspace(0.45, 1.0, K),
    ]
    if K >= 12:                          # boundary-heavy + one atom at origin
        cands.append(np.concatenate([[0.0], 0.6 + 0.4 * (np.arange(K - 1) + 0.5) / (K - 1)]))
        cands.append(np.concatenate([[0.0], np.linspace(0.85, 1.0, K - 1)]))
    return [np.clip(np.sort(t), 0.0, 1.0) for t in cands]

def nm_objective(x, K, n_uniform=2000, iters=18):
    t = np.clip(np.sort(np.asarray(x, float)), 0.0, 1.0)
    r = best_r_for_t(t, iters=iters, n_uniform=n_uniform)
    return -r

def main():
    log("=== radial measure lower bound for r_D(100) ===")
    target = 1.0 / 100.0
    DEADLINE = T0 + 22 * 60.0   # global wall-clock budget (minutes)

    # ---- stage 1: pattern sweep, K = 10..20 (+ seed from previous runs)
    best = None   # (r, t, w)
    seed_path = os.path.join(OUT, "seed_t.json")
    if os.path.exists(seed_path):
        with open(seed_path) as f:
            t_seed = np.array(json.load(f), float)
        r_s, w_s = best_r_for_t(t_seed, iters=24, n_uniform=3000, w_ret=True)
        best = (r_s, t_seed.copy(), w_s)
        log("seed pattern: r=%.8f (K=%d)" % (r_s, len(t_seed)))
    for K in range(10, 21):
        if best is not None and time.time() > DEADLINE - 420:
            log("deadline guard: stop sweep at K=%d" % K)
            break
        for t in patterns(K):
            r, w = best_r_for_t(t, iters=24, n_uniform=3000, w_ret=True)
            if best is None or r > best[0]:
                best = (r, t.copy(), w)
        log("K=%2d done (%.0fs), best so far r=%.8f  (t=%s)"
            % (K, time.time() - T0, best[0], np.round(best[1], 4)))
    r0, t0, w0 = best
    log("stage-1 best: r=%.8f, K=%d, t=%s" % (r0, len(t0), np.round(t0, 5)))

    # ---- stage 2: Nelder-Mead refinement of t (top structure), restart from best
    K = len(t0)
    x0 = t0.copy()
    if time.time() < DEADLINE - 420:
        res = minimize(nm_objective, x0, args=(K,), method='Nelder-Mead',
                       options=dict(maxfev=150, xatol=1e-4, fatol=1e-6, adaptive=True))
        t_nm = np.clip(np.sort(res.x), 0.0, 1.0)
        r_nm, w_nm = best_r_for_t(t_nm, iters=26, n_uniform=4000, w_ret=True)
        log("Nelder-Mead: r=%.8f (was %.8f), t=%s" % (r_nm, r0, np.round(t_nm, 5)))
        if r_nm > r0:
            r0, t0, w0 = r_nm, t_nm, w_nm

        # also NM from sqrt-pattern start if it was competitive
        t2 = np.sort(np.sqrt((np.arange(K) + 0.5) / K))
        r2 = best_r_for_t(t2, iters=24)
        if r2 > 0 and r2 > r0 - 0.001 and time.time() < DEADLINE - 420:
            res2 = minimize(nm_objective, t2, args=(K,), method='Nelder-Mead',
                            options=dict(maxfev=120, xatol=1e-4, fatol=1e-6, adaptive=True))
            t_nm2 = np.clip(np.sort(res2.x), 0.0, 1.0)
            r_nm2, w_nm2 = best_r_for_t(t_nm2, iters=26, n_uniform=4000, w_ret=True)
            log("Nelder-Mead#2: r=%.8f, t=%s" % (r_nm2, np.round(t_nm2, 5)))
            if r_nm2 > r0:
                r0, t0, w0 = r_nm2, t_nm2, w_nm2

    log("optimization done: r_LP = %.8f, K = %d" % (r0, len(t0)))
    log("t = %s" % np.array2string(np.asarray(t0), precision=6, max_line_width=200))
    log("w = %s" % np.array2string(np.asarray(w0), precision=6, max_line_width=200))

    # ---- stage 3: rigorous certification with FIXED w
    # phi_k(d; r) is non-decreasing in r for every d, so sup F(t, w, r) is
    # non-decreasing in r: bisect r with w frozen.  This decouples certification
    # from LP re-saturation (re-solving the LP at each r would push sup back to
    # exactly 1/100 at grid points and mask the true feasibility margin).
    t_vec = np.asarray(t0, float)
    ok, w_fix = lp_feasible(t_vec, r0, target, n_uniform=5000)
    if not ok:
        r0 -= 2e-6
        ok, w_fix = lp_feasible(t_vec, r0, target, n_uniform=5000)
        assert ok, "LP infeasible at slightly reduced r"
    hi = r0
    sup_hi, _ = rigorous_sup(t_vec, w_fix, hi, cap=target)
    lo = hi - 3e-3
    for _ in range(60):                       # ensure lo passes
        sup_lo, _ = rigorous_sup(t_vec, w_fix, lo, cap=target)
        if sup_lo <= target:
            break
        lo -= 3e-3
    else:
        raise RuntimeError("no certifiable r found even after large shrink")
    for _ in range(40):                       # bisect [lo, hi] on fixed w
        mid = 0.5 * (lo + hi)
        sup_m, info = rigorous_sup(t_vec, w_fix, mid, cap=target)
        if sup_m <= target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    r_cert = lo
    sup_ub, info = rigorous_sup(t_vec, w_fix, r_cert, cap=target)
    w0 = w_fix
    log("CERTIFIED: r=%.9f  sup_upper=%.12f (<= 1/100)  n_grid=%d"
        % (r_cert, sup_ub, info['n_grid']))

    # final re-verify with finer base grid, SAME protocol (refine everything above target)
    sup_ub2, info2 = rigorous_sup(t_vec, w0, r_cert, h0=5e-4, cap=target,
                                  min_len=2e-10, max_rounds=400)
    log("re-verify h0=5e-4 cap=target: sup_upper=%.12f  n_grid=%d" % (sup_ub2, info2['n_grid']))
    if sup_ub2 > target:                      # shrink r a touch, w unchanged
        for _ in range(50):
            r_cert -= 1e-6
            sup_ub2, info2 = rigorous_sup(t_vec, w0, r_cert, h0=5e-4, cap=target,
                                          min_len=2e-10, max_rounds=400)
            if sup_ub2 <= target:
                break
        log("final shrink -> r=%.9f sup_upper=%.12f" % (r_cert, sup_ub2))

    cert = dict(
        r=r_cert,
        t=[float(x) for x in t_vec],
        w=[float(x) for x in w0],
        sup_upper=float(sup_ub2),
        grid=dict(h0_final=5e-4, n_grid=info2['n_grid'], domain_end=info2['dom'],
                  min_len=1e-11, method="unimodal-exact-sup per interval + adaptive refinement"),
        target=1.0 / 100.0,
        comparison=dict(area_bound=0.1, oler_claim=0.10397, radial_claim=0.10643),
    )
    with open(os.path.join(OUT, "measure_certificate.json"), "w") as f:
        json.dump(cert, f, indent=2)
    log("certificate saved: r_cert = %.9f  sup_upper = %.12f" % (r_cert, sup_ub2))
    log("comparison: area 0.1 | Oler claim 0.10397 | radial claim 0.10643 | certified %.6f"
        % r_cert)

if __name__ == "__main__":
    main()
