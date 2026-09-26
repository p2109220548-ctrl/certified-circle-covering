# -*- coding: utf-8 -*-
"""
certify_frozen.py — Sound frozen-w certification + repair of the fall-through bug.

AUDIT FINDING (2026-09-26): measure_upgrade.py's shrink loop falls through after
40 failed iterations and still logs CERTIFIED; the saved flagship certificate
agentF_upgrade/measure_certificate_upgraded.json has sup_upper = 0.010000091 >
1/100, i.e. its rigorous protocol NEVER passed.  Root cause: re-solving the LP
at each lowered r re-saturates the constraints (sup pinned just above cap) —
exactly the effect agentB's stage-3 comment warned about.

Correct protocol (this file):
  1. Obtain (t, w) ONCE from the LP at r_lp; PRUNE to active support; FREEZE w.
  2. Shrink r only (sup is non-decreasing in r for fixed measure => bisection
     valid); NEVER re-solve the LP during certification.
  3. Certificate is saved ONLY if rigorous sup_upper <= 1/100 (asserted).

Modes:
  repair : re-certify the agentF 88-ring certificate with frozen w (fast).
  new    : LP phase for the K=320 machine (seeded), checkpoint, then frozen
           certification with slope-extrapolated target.
"""
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "agentB_measure"))

from measure_bound import phi_mat, critical_points, rigorous_sup  # noqa: E402
from radial_true import (build_t_grid, solve_w, cutting_rounds,  # noqa: E402
                         fine_scan_sup, prune)

T0 = time.time()
def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)

CAP = 1.0 / 100.0


def certify_pass(t_vec, w, r, h0=2e-3, max_points=60_000):
    ub, info = rigorous_sup(t_vec, w, r, h0=h0, cap=CAP,
                            max_points=max_points)
    return ub, info


def frozen_shrink(t_vec, w, r_start, step, max_iter=60, h0=2e-3,
                  max_points=60_000, label=""):
    """Freeze w; decrease r by `step` until rigorous ub <= CAP.  No LP re-solve.
    Returns (r_cert, ub, info) with ub <= CAP guaranteed (raises otherwise)."""
    r = r_start
    for it in range(max_iter):
        ub, info = certify_pass(t_vec, w, r, h0=h0, max_points=max_points)
        log("  %s it=%d r=%.9f ub=%.12f %s" % (label, it, r, ub,
                                               "PASS" if ub <= CAP else ""))
        if ub <= CAP:
            return r, ub, info
        r -= step
    raise RuntimeError("frozen certification failed to pass within %d steps"
                       % max_iter)


def repair_old():
    """Re-certify the agentF flagship with the frozen-w protocol."""
    p = os.path.join(BASE, "agentF_upgrade", "measure_certificate_upgraded.json")
    c = json.load(open(p))
    t = np.array(c["t"], float)
    w = np.array(c["w"], float)
    r_bad = float(c["r"])
    log("repair: loaded agentF cert r=%.9f sup_upper=%.12f (>CAP: %s)"
        % (r_bad, c["sup_upper"], c["sup_upper"] > CAP))
    t_a, w_a = prune(t, w, eps=1e-12)
    log("repair: pruned K=%d -> %d" % (len(t), len(t_a)))
    r_cert, ub, info = frozen_shrink(t_a, w_a, r_bad, step=2e-6, max_iter=60,
                                     h0=1e-3, max_points=120_000,
                                     label="repair")
    out = dict(r=r_cert, t=[float(x) for x in t_a], w=[float(x) for x in w_a],
               sup_upper=float(ub), strict_margin=float(CAP - ub),
               repaired_from=r_bad,
               note="repaired frozen-w certification; original certificate "
                    "had sup_upper=0.010000091 > 1/100 (fall-through bug in "
                    "measure_upgrade.py shrink loop)",
               claim="no covering of D by 100 disks of radius <= r exists",
               grid=dict(h0=1e-3, n_grid=int(info["n_grid"])))
    with open(os.path.join(HERE, "agentF_repaired_certificate.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("repair DONE: r=%.9f (was %.9f, delta=%.2e)  ub=%.12f"
        % (r_cert, r_bad, r_bad - r_cert, ub))


def new_machine():
    """K=320 LP phase (seeded) -> checkpoint -> frozen certification."""
    ckpt_path = os.path.join(HERE, "lp_checkpoint.json")
    if os.path.exists(ckpt_path):
        ck = json.load(open(ckpt_path))
        t_act = np.array(ck["t"], float)
        w_act = np.array(ck["w"], float)
        r_lp = float(ck["r_lp"])
        log("new: loaded checkpoint r_lp=%.9f K_active=%d"
            % (r_lp, len(t_act)))
    else:
        t_vec = build_t_grid()
        r_seed = float(os.environ.get("RADIAL_R_SEED", "0.1065"))
        # coarse bracket bisection (no cuts) just to land near the limit
        lo, hi = r_seed - 1.5e-4, r_seed + 1.5e-4
        ok, w, grid = solve_w(t_vec, lo)
        assert ok, "seed lo infeasible"
        best_w, best_grid = w, grid
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            ok, w, grid = solve_w(t_vec, mid)
            if ok:
                lo, best_w, best_grid = mid, w, grid
            else:
                hi = mid
            if hi - lo < 5e-7:
                break
        # cutting rounds at the bracket edge (tighten the d-grid there)
        r0 = lo
        ok, w, grid = solve_w(t_vec, r0)
        w, grid, _ = cutting_rounds(t_vec, r0, w, grid, rounds=4)
        if w is None:
            r0 -= 1e-5
            ok, w, grid = solve_w(t_vec, r0)
            w, grid, _ = cutting_rounds(t_vec, r0, w, grid, rounds=4)
        r_lp = r0
        log("new: LP phase done r_lp=%.9f" % r_lp)
        t_act, w_act = prune(t_vec, w, eps=1e-12)
        json.dump(dict(r_lp=r_lp, t=[float(x) for x in t_act],
                       w=[float(x) for x in w_act]),
                  open(ckpt_path, "w"), indent=1)
        log("new: checkpoint saved (K_active=%d)" % len(t_act))

    # frozen certification with slope extrapolation
    r_a = r_lp - 1.5e-5
    ub_a, _ = certify_pass(t_act, w_act, r_a, h0=2e-3)
    log("new: probe A r=%.9f ub=%.12f" % (r_a, ub_a))
    if ub_a <= CAP:
        r_cert, ub, info = r_a, ub_a, None
    else:
        r_b = r_lp - 6e-5
        ub_b, _ = certify_pass(t_act, w_act, r_b, h0=2e-3)
        log("new: probe B r=%.9f ub=%.12f" % (r_b, ub_b))
        slope = (ub_a - ub_b) / (r_a - r_b)     # >= 0 expected
        if slope <= 1e-9:
            r_target = r_b - 1e-5
        else:
            r_target = r_b - (ub_b - CAP) / slope - 2e-7
        log("new: slope=%.6f  target r=%.9f" % (slope, r_target))
        r_cert, ub, info = frozen_shrink(t_act, w_act, r_target, step=1e-6,
                                         max_iter=40, h0=1e-3,
                                         max_points=150_000, label="new")
    # finer independent re-verify at the certified point
    ub2, info2 = certify_pass(t_act, w_act, r_cert, h0=5e-4, max_points=150_000)
    log("new: re-verify h0=5e-4 ub=%.12f" % (ub2,))
    while ub2 > CAP:
        r_cert -= 2e-6
        ub2, info2 = certify_pass(t_act, w_act, r_cert, h0=5e-4,
                                  max_points=150_000)
        log("new: re-verify shrink r=%.9f ub=%.12f" % (r_cert, ub2))
    out = dict(r=r_cert, t=[float(x) for x in t_act],
               w=[float(x) for x in w_act],
               sup_upper=float(ub2), strict_margin=float(CAP - ub2),
               lp_limit=float(r_lp),
               claim="no covering of D by 100 disks of radius <= r exists; "
                     "hence r_D(100) >= r",
               theorem="rotation-averaging optimality (OPTIMALITY_THEOREM.md)",
               grid=dict(h0_final=5e-4, n_grid=int(info2["n_grid"])))
    with open(os.path.join(HERE, "radial_true_certificate.json"), "w") as f:
        json.dump(out, f, indent=1)
    log("new DONE: r=%.9f  ub=%.12f  (LP limit %.9f, frozen-shrink cost %.2e)"
        % (r_cert, ub2, r_lp, r_lp - r_cert))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("repair", "both"):
        repair_old()
    if mode in ("new", "both"):
        new_machine()
