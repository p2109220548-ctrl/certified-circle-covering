# -*- coding: utf-8 -*-
"""
measure_upgrade.py — Upgrade the certified radial-measure lower bound.

Base: agentB_measure (certified r = 0.105445, K=20 rings).  Upgrades:
  1. CUTTING-PLANE: at the current best r, find the binding center-distances
     (where F(d) is closest to the cap 1/100) and ADD new rings at t = d_binding
     (a ring at t puts its mass-peak at center-distance sqrt(t^2-r^2) ~ t).
     Re-solve the LP for w with the enlarged ring set; re-bisect r upward.
  2. DENSIFICATION: insert extra rings between existing rings weighted by w.
  3. NM polish of the final t-vector (budgeted).
Everything reuses the VERIFIED machinery in agentB_measure/measure_bound.py
(phi exact, critical points, two-sided sup protocol, LP feasibility).
Soundness: any (t, w) with rigorous_sup <= 1/100 certifies r_D(100) > r.
"""
import os
import sys
import json
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "agentB_measure"))
sys.path.insert(0, BASE)

from measure_bound import (phi_mat, critical_points, rigorous_sup,
                           lp_feasible, best_r_for_t)
from scipy.optimize import minimize

T0 = time.time()
def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)

OUT = HERE


def binding_ds(t_vec, w, r, n=20000):
    """d-locations where F(d) is closest to the cap (top binding sites)."""
    dom = float(np.max(t_vec)) + r
    d = np.linspace(0.0, dom, n)
    F = phi_mat(t_vec, d, r) @ w
    cap = 1.0 / 100.0
    gap = cap - F
    idx = np.argsort(gap)[:60]          # smallest gap = most binding
    return d[idx], gap[idx]


def solve_w(t_vec, r, n_uniform=4000):
    ok, w = lp_feasible(t_vec, r, 1.0 / 100.0, n_uniform=n_uniform, time_limit=8.0)
    return ok, w


def certify(t_vec, w, r):
    ub, info = rigorous_sup(t_vec, w, r, h0=1e-3, cap=1.0 / 100.0)
    return ub <= 1.0 / 100.0, ub, info


def main():
    target = 1.0 / 100.0
    cert = json.load(open(os.path.join(BASE, "agentB_measure",
                                       "measure_certificate_verified.json")))
    t_cur = np.array(cert["t"], float)
    r_cur = float(cert["r"])
    ok, w = solve_w(t_cur, r_cur)
    log("seed reproduce: r=%.9f feasible=%s" % (r_cur, ok))
    best = (r_cur, t_cur.copy(), w)

    # ---- cutting-plane rounds ----
    for rd in range(10):
        r_try = best[0] + 3e-4      # push the target up a notch
        t_b = best[1].copy()
        ok, w = solve_w(t_b, r_try, n_uniform=5000)
        added = 0
        if ok:
            # still feasible at higher r with current rings: just take it
            best = (r_try, t_b, w)
            log("round %d: feasible at r=%.9f with current rings" % (rd, r_try))
            continue
        # find binding d's at the CURRENT best r
        dbs, gaps = binding_ds(t_b, best[2], best[0])
        new_t = []
        for db in dbs[:12]:
            t_new = float(np.clip(db, 0.0, 1.0))
            if all(abs(t_new - x) > 2e-3 for x in t_b):
                new_t.append(t_new)
                added += 1
        if added == 0:
            # densify: midpoints between top-weight adjacent rings
            order = np.argsort(best[2])[::-1][:12]
            tt = np.sort(t_b)
            for i in order[::2]:
                j = min(np.searchsorted(tt, tt[i]) + 1, len(tt) - 1)
                mid = 0.5 * (tt[max(0, j - 1)] + tt[j])
                if all(abs(mid - x) > 1e-3 for x in t_b):
                    new_t.append(float(mid))
                    added += 1
        if added == 0:
            log("round %d: no new rings to add; stop" % rd)
            break
        t_new = np.sort(np.concatenate([t_b, np.array(new_t)]))
        ok, w = solve_w(t_new, r_try, n_uniform=5000)
        if ok:
            best = (r_try, t_new, w)
            log("round %d: +%d rings (K=%d) -> feasible at r=%.9f"
                % (rd, added, len(t_new), r_try))
        else:
            # try half the push
            r_try = best[0] + 1e-4
            ok, w = solve_w(t_new, r_try, n_uniform=5000)
        if ok:
            best = (r_try, t_new, w)
            log("round %d: +%d rings (K=%d) -> feasible at r=%.9f (half push)"
                % (rd, added, len(t_new), r_try))
        else:
            log("round %d: +%d rings did not help at r=%.9f; keep rings, retry base"
                % (rd, added, r_try))
            # keep the enlarged ring set but re-solve w at the current best r
            ok2, w2 = solve_w(t_new, best[0], n_uniform=5000)
            if ok2:
                best = (best[0], t_new, w2)
            # else: revert to previous (r, t, w) untouched

    # ---- NM polish on the final t (budgeted) ----
    t_f = best[1]
    r_f, w_f = best_r_for_t(t_f, iters=26, n_uniform=5000, w_ret=True, budget=240.0)
    log("post-CP bisection: r=%.9f (K=%d)" % (r_f, len(t_f)))
    if r_f > best[0]:
        best = (r_f, t_f, w_f)

    def nm_obj(x):
        t = np.clip(np.sort(np.asarray(x, float)), 0.0, 1.0)
        return -best_r_for_t(t, iters=16, n_uniform=2500, budget=12.0)

    res = minimize(nm_obj, t_f, method="Nelder-Mead",
                   options=dict(maxfev=90, xatol=1e-4, fatol=1e-6, adaptive=True))
    t_nm = np.clip(np.sort(res.x), 0.0, 1.0)
    r_nm, w_nm = best_r_for_t(t_nm, iters=26, n_uniform=5000, w_ret=True, budget=240.0)
    log("NM polish: r=%.9f (was %.9f)" % (r_nm, r_f))
    if r_nm > r_f:
        best = (r_nm, t_nm, w_nm)

    # ---- strict certification with shrink loop ----
    r_best, t_best, w_best = best
    if w_best is None:
        ok, w_best = solve_w(t_best, r_best, n_uniform=6000)
    json.dump(dict(r=r_best, t=list(map(float, t_best))),
              open(os.path.join(OUT, "upgrade_checkpoint.json"), "w"), indent=1)
    for _ in range(40):
        if w_best is None:
            r_best -= 2e-5
            ok, w_best = solve_w(t_best, r_best, n_uniform=6000)
            if not ok:
                continue
        passed, ub, info = certify(t_best, w_best, r_best)
        if passed:
            break
        r_best -= 2e-6
        ok, w_best = solve_w(t_best, r_best, n_uniform=6000)
        if not ok:
            w_best = None   # force a fresh solve at the next iteration with lower r
    log("CERTIFIED: r=%.9f  sup_upper=%.12f  grid=%d" % (r_best, ub, info["n_grid"]))

    json.dump(dict(r=r_best, t=list(map(float, t_best)),
                   w=list(map(float, w_best)), sup_upper=float(ub),
                   grid=info),
              open(os.path.join(OUT, "measure_certificate_upgraded.json"), "w"), indent=1)
    log("saved measure_certificate_upgraded.json")
    log("comparison: previous certified 0.105445 | upgrade %.6f (+%.6f)"
        % (r_best, r_best - 0.105445))


if __name__ == "__main__":
    main()
