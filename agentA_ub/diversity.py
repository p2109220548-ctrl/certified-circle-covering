"""
diversity.py — Diversity moves on top of the ILS best.

Move (a): delete the 2 centers with the smallest Voronoi-cell "usefulness"
(smallest max distance from center to its cell polygon), add 2 random
boundary-ring centers, then re-polish (lloyd + coordinate descent).
Keeps improving if a move helps; repeats until budget.

Move (b): build the runner-up topology [1,6,12,20,28,33] from scratch with a
short Nelder-Mead run, then polish — cheap exploration of another cell family.

Usage: python diversity.py <budget_s> <ckpt_out>
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from geometry import covering_radius          # noqa: E402
from lloyd_cover import lloyd_polish, cell_polygon  # noqa: E402
from construct import coordinate_descent, build, unpack_x  # noqa: E402


def load_best():
    """Best across all chain checkpoints and the shared best."""
    cands = [os.path.join(PARENT, "centers_best.json")]
    cands += sorted(glob.glob(os.path.join(HERE, "ckpt_A*.json")))
    best = None
    for p in cands:
        with open(p) as f:
            d = json.load(f)
        R = covering_radius(np.array(d["centers"], float))
        if best is None or R < best[0]:
            best = (R, np.array(d["centers"], float), p)
    return best


def save_checkpoint(C, path):
    Rc = covering_radius(C)
    out = dict(n=int(len(C)), radius=float(Rc), topology="agentA",
               centers=[[float(a), float(b)] for a, b in C])
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return Rc


def usefulness(C, i):
    cell = cell_polygon(C, i)
    if len(cell) < 3:
        return -1.0                      # empty/dead cell: least useful
    return float(np.max(np.hypot(cell[:, 0] - C[i, 0],
                                 cell[:, 1] - C[i, 1])))


def move_a(C, R, rng):
    use = [usefulness(C, i) for i in range(len(C))]
    order = np.argsort(use)
    drop = [int(order[0]), int(order[1])]
    keep = [i for i in range(len(C)) if i not in drop]
    C2 = C[keep].copy()
    ring = np.hypot(C[:, 0], C[:, 1])
    r_typ = float(np.quantile(ring, 0.85))
    for _ in range(2):
        th = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(r_typ, min(1.02, r_typ + 0.15))
        C2 = np.vstack([C2, [r * np.cos(th), r * np.sin(th)]])
    R2 = covering_radius(C2)
    C2, R2 = lloyd_polish(C2, R2, sweeps=5, lams=(1.0, 0.5, 0.2),
                          seed=int(rng.integers(1, 10**9)), log=False)
    C2, R2 = coordinate_descent(C2, R2, steps=(2e-4, 3e-5), passes=1)
    return C2, R2


def topology_attempt(m, rng):
    """Build ring topology m from scratch, short NM + polish."""
    from construct import init_x, objective

    def obj(x):
        return objective(x, m)

    x0 = init_x(m, rng, randomize=True)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options=dict(maxiter=1200, xatol=1e-7, fatol=1e-10,
                                adaptive=True))
    rho, psi = unpack_x(m, res.x)
    C = build(m, rho, psi)
    R = covering_radius(C)
    C, R = lloyd_polish(C, R, sweeps=3, lams=(1.0, 0.5, 0.2),
                        seed=int(rng.integers(1, 10**9)), log=False)
    return C, R


def main():
    budget_s = float(sys.argv[1])
    ckpt_out = os.path.join(HERE, sys.argv[2])
    rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 77)
    R_best, C_best, src = load_best()
    print(f"[diversity] start from {os.path.basename(src)}, "
          f"R = {R_best:.10f}", flush=True)
    t0 = time.time()
    att = 0

    # move (a): delete-2 add-2 ring, repeat
    while time.time() - t0 < budget_s * 0.7:
        att += 1
        C2, R2 = move_a(C_best, R_best, rng)
        tag = "  <-- new best" if R2 < R_best - 1e-10 else ""
        print(f"[diversity] move_a #{att}: R = {R2:.10f}{tag}"
              f"   [{time.time()-t0:.0f}s]", flush=True)
        if R2 < R_best - 1e-10:
            R_best, C_best = R2, C2.copy()
            Rc = save_checkpoint(C_best, ckpt_out)
            print(f"[diversity] saved, recomputed R = {Rc:.10f}", flush=True)

    # move (b): runner-up topology, if time remains
    while time.time() - t0 < budget_s:
        att += 1
        C2, R2 = topology_attempt([1, 6, 12, 20, 28, 33], rng)
        tag = "  <-- new best" if R2 < R_best - 1e-10 else ""
        print(f"[diversity] topo_b #{att}: R = {R2:.10f}{tag}"
              f"   [{time.time()-t0:.0f}s]", flush=True)
        if R2 < R_best - 1e-10:
            R_best, C_best = R2, C2.copy()
            Rc = save_checkpoint(C_best, ckpt_out)
            print(f"[diversity] saved, recomputed R = {Rc:.10f}", flush=True)
        else:
            break                      # one attempt is enough if it stalls

    print(f"[diversity] FINAL best R = {R_best:.10f} "
          f"({att} attempts, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
