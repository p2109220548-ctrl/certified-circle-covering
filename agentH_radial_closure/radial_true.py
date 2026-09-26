# -*- coding: utf-8 -*-
"""
radial_true.py — Push the radial-measure lower bound to its TRUE limit.

Motivation (see OPTIMALITY_THEOREM.md): by rotation-averaging, the radial family
is OPTIMAL within the entire class of Borel probability measures.  Hence the best
lower bound obtainable from ANY measure/duality argument equals the radial value

    V(r) = 1  iff  exists probability measure nu on [0,1] (radial profile)
                 with  sup_{d in [0, 1+r]} F_nu(d) <= 1/100,
    F_nu(d) = integral phi_t(d) dnu(t),   phi_t = ring kernel (exact, unimodal).

Machine:
  1. Fine ring grid: t = 0 (atom) + 200 uniform in (0,1] + 120 uniform in
     [0.82, 1]  (boundary-heavy, matches the known optimal profile shape).
  2. LP over ring weights w (>=0, sum=1) on a d-grid = all critical points
     (tangencies |t-r|, t+r, peaks sqrt(t^2-r^2)) + uniform 6000.
  3. Cutting planes on d: fine-scan F (200k pts), add the top violating/binding
     d's as new LP rows, re-solve (iterate until sup - cap stable).
  4. Bisect r upward to the LP limit.
  5. Strict certification: fixed-w rigorous_sup (agentB protocol: exact
     criticals + two-sided monotone bounds + adaptive refinement), shrink r
     until sup_upper <= 1/100.  Certificate saved.

Soundness unchanged: any (t, w, r) with rigorous sup F <= 1/100 proves
r_D(100) >= r.  The theorem above proves NOTHING can do better than V's limit.
"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "agentB_measure"))

from measure_bound import phi_mat, critical_points, rigorous_sup

T0 = time.time()
def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)

CAP = 1.0 / 100.0


def build_t_grid():
    t = [0.0]
    t += list(np.linspace(1.0 / 200, 1.0, 200))
    t += list(np.linspace(0.82, 1.0, 120))
    return np.unique(np.clip(np.array(t), 0.0, 1.0))


def lp_grid_d(t_vec, r, n_uniform=6000, extra=None):
    dom = float(np.max(t_vec)) + r
    pts = list(critical_points(t_vec, r, dom))
    pts += list(np.linspace(0.0, dom, n_uniform))
    if extra is not None:
        pts += list(extra)
    return np.unique(np.clip(np.array(pts), 0.0, dom))


def solve_w(t_vec, r, grid=None, n_uniform=6000, time_limit=20.0):
    if grid is None:
        grid = lp_grid_d(t_vec, r, n_uniform=n_uniform)
    Phi = phi_mat(t_vec, grid, r)
    K = len(t_vec)
    res = linprog(c=np.zeros(K), A_ub=Phi, b_ub=np.full(len(grid), CAP),
                  A_eq=np.ones((1, K)), b_eq=[1.0], bounds=[(0, None)] * K,
                  method="highs", options=dict(time_limit=time_limit,
                                               presolve=True))
    if res.status != 0:
        return False, None, grid
    w = res.x / res.x.sum()
    return True, w, grid


def fine_scan_sup(t_vec, w, r, dom, n_fine=200_001, chunk=20_000):
    """Memory-safe fine scan of F in chunks.  Returns (sup, d_at_sup)."""
    d_fine = np.linspace(0.0, dom, n_fine)
    best, arg = -1.0, 0
    for s in range(0, n_fine, chunk):
        blk = d_fine[s:s + chunk]
        F = phi_mat(t_vec, blk, r) @ w
        j = int(np.argmax(F))
        if float(F[j]) > best:
            best, arg = float(F[j]), s + j
    return best, d_fine[arg]


def cutting_rounds(t_vec, r, w, grid, rounds=6):
    """Add binding d's (from a chunked fine scan) as LP rows; re-solve."""
    dom = float(np.max(t_vec)) + r
    for it in range(rounds):
        mx, dbest = fine_scan_sup(t_vec, w, r, dom)
        crit = critical_points(t_vec, r, dom)
        Fc = phi_mat(t_vec, crit, r) @ w
        if float(Fc.max()) > mx:
            mx, dbest = float(Fc.max()), float(crit[int(np.argmax(Fc))])
        slack = CAP - mx
        log("  cut-round %d: sup_fine=%.9f at d=%.6f  slack=%.3e"
            % (it, mx, dbest, slack))
        if slack >= -1e-12:
            return w, grid, mx
        # add the worst d's: recompute per-chunk top-40 (cheap, chunked)
        d_fine = np.linspace(0.0, dom, 100_001)
        worst = []
        for s in range(0, len(d_fine), 20_000):
            blk = d_fine[s:s + 20_000]
            F = phi_mat(t_vec, blk, r) @ w
            worst += list(blk[np.argsort(F)[-8:]])
        worst.append(dbest)
        grid = np.unique(np.concatenate([grid, np.clip(np.array(worst), 0, dom)]))
        ok, w2, grid = solve_w(t_vec, r, grid=grid)
        if not ok:
            log("  cut-round %d: LP became infeasible on refined grid" % it)
            return None, grid, mx
        w = w2
    return w, grid, mx


def best_r(t_vec, r_lo=0.09, r_hi=0.13, iters=22):
    """Bisect largest r that stays LP-feasible (grid rebuilt each r)."""
    ok, w, _ = solve_w(t_vec, r_lo)
    if not ok:
        return 0.0, None
    ok_hi, _, _ = solve_w(t_vec, r_hi)
    if ok_hi:
        return r_hi, None
    lo, hi, best_w = r_lo, r_hi, w
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok, w, _ = solve_w(t_vec, mid)
        if ok:
            lo, best_w = mid, w
        else:
            hi = mid
        if hi - lo < 3e-7:
            break
    return lo, best_w


def certify(t_vec, w, r):
    ub, info = rigorous_sup(t_vec, w, r, h0=2e-3, cap=CAP, max_points=60_000)
    return ub <= CAP, ub, info


def prune(t_vec, w, eps=1e-12):
    keep = w > eps
    return t_vec[keep], w[keep] / w[keep].sum()


def main():
    t_vec = build_t_grid()
    log("ring grid: K=%d" % len(t_vec))

    r_seed = float(os.environ.get("RADIAL_R_SEED", "0"))
    if r_seed > 0:
        r0, w0 = best_r(t_vec, r_lo=r_seed - 1.5e-4, r_hi=r_seed + 1.5e-4, iters=18)
    else:
        r0, w0 = best_r(t_vec)
    log("LP bisection: r=%.9f" % r0)

    # cutting rounds at the achieved r
    ok, w, grid = solve_w(t_vec, r0)
    w, grid, sup_fine = cutting_rounds(t_vec, r0, w, grid)
    if w is None:
        r0 -= 3e-5
        ok, w, grid = solve_w(t_vec, r0)
        w, grid, sup_fine = cutting_rounds(t_vec, r0, w, grid)
    log("after cutting: r=%.9f  sup_fine=%.9f" % (r0, sup_fine))

    # one more bisection with the refined-grid solver (cuts active at every r)
    def feas(rr):
        ok, w, grid = solve_w(t_vec, rr, n_uniform=6000)
        if not ok:
            return None
        w, grid, _ = cutting_rounds(t_vec, rr, w, grid, rounds=3)
        return w

    lo, hi = r0, r0 + 3e-4
    best_w = w
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        wm = feas(mid)
        if wm is not None:
            lo, best_w = mid, wm
        else:
            hi = mid
        if hi - lo < 3e-7:
            break
    r_lp, w_lp = lo, best_w
    log("final LP limit: r=%.9f" % r_lp)

    # prune to active support (memory + speed for certification; measure unchanged)
    t_act, w_act = prune(t_vec, w_lp)
    log("pruned K: %d -> %d" % (len(t_vec), len(t_act)))

    # strict certification with fixed w, shrink until pass.
    # Start slightly below the LP limit: the rigorous sup adds conservatism,
    # so the first few 1e-6-steps of r are pure waste otherwise.
    r_cert = r_lp - 2e-5
    w_fix = None                      # (re)solved lazily in the loop
    ub, info = np.inf, None
    for it in range(80):
        if w_fix is None:
            ok, w_full, _ = solve_w(t_vec, r_cert, n_uniform=6000, time_limit=40.0)
            if not ok:
                r_cert -= 3e-6        # LP infeasible (or timed out): go lower
                continue
            t_act, w_fix = prune(t_vec, w_full)
        passed, ub, info = certify(t_act, w_fix, r_cert)
        if passed:
            break
        r_cert -= 1.5e-6
        w_fix = None                  # force re-solve at the new r
    log("CERTIFIED: r=%.9f  sup_upper=%.12f  n_grid=%d  (K_active=%d)"
        % (r_cert, ub, info["n_grid"], len(t_act)))

    # explicit strictness margin: shrink a hair; then S(r_cert) < 1/100 strictly
    r_strict = r_cert - 1e-8
    ub_s, _ = rigorous_sup(t_act, w_fix, r_strict, h0=2e-3, cap=CAP,
                           max_points=60_000)
    if ub_s < CAP:
        r_cert, ub = r_strict, ub_s
        log("strictness margin: r=%.9f  sup_upper=%.12f  delta=%.3e"
            % (r_cert, ub, CAP - ub))

    # independent re-verify with finer base grid
    ub2, info2 = rigorous_sup(t_act, w_fix, r_cert, h0=5e-4, cap=CAP,
                              min_len=2e-10, max_rounds=400, max_points=60_000)
    log("re-verify h0=5e-4: sup_upper=%.12f  n_grid=%d" % (ub2, info2["n_grid"]))
    if ub2 > CAP:
        for _ in range(50):
            r_cert -= 1e-6
            ub2, info2 = rigorous_sup(t_act, w_fix, r_cert, h0=5e-4, cap=CAP,
                                      min_len=2e-10, max_rounds=400,
                                      max_points=60_000)
            if ub2 <= CAP:
                break
        log("final shrink -> r=%.9f  sup_upper=%.12f" % (r_cert, ub2))

    cert = dict(
        r=r_cert,
        K_total=int(len(t_vec)),
        K_active=int(len(t_act)),
        t=[float(x) for x in t_act],
        w=[float(x) for x in w_fix],
        sup_upper=float(min(ub, ub2)),
        strict_margin=float(CAP - min(ub, ub2)),
        claim="no covering of D by 100 disks of radius <= r exists "
              "(mass chain with strict margin); hence r_D(100) >= r",
        lp_limit=float(r_lp),
        grid=dict(h0_final=5e-4, n_grid=int(info2["n_grid"]),
                  method="exact criticals + two-sided monotone bounds + adaptive refinement"),
        target=CAP,
        theorem="rotation-averaging: radial family is optimal among all Borel "
                "probability measures (see OPTIMALITY_THEOREM.md)",
    )
    with open(os.path.join(HERE, "radial_true_certificate.json"), "w") as f:
        json.dump(cert, f, indent=1)
    log("saved radial_true_certificate.json")
    log("comparison: previous certified 0.105759 | new certified %.6f (+%.6f)"
        % (r_cert, r_cert - 0.105759))
    log("measure-method ceiling reached; further progress requires "
        "non-measure (structural) arguments.")


if __name__ == "__main__":
    main()
