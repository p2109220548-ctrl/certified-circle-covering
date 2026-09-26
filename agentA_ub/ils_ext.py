"""
ils_ext.py — Extended iterated local search for the n=100 disk covering problem.

Chain: kick (N(0,kick) on all 200 coords) -> lloyd_polish -> coordinate descent.
Keeps a per-chain checkpoint (agentA_ub/<ckpt>) updated after every improvement,
with the saved radius recomputed via the exact evaluator.

Usage: python ils_ext.py <seed> <kick_order_csv> <reps_per_kick> <budget_s> <ckpt> [start_json]
"""
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from geometry import covering_radius          # noqa: E402
from lloyd_cover import lloyd_polish          # noqa: E402
from construct import coordinate_descent      # noqa: E402


def save_checkpoint(C, path):
    """Save with radius recomputed by the exact evaluator (sanity)."""
    Rc = covering_radius(C)
    out = dict(n=int(len(C)), radius=float(Rc), topology="agentA",
               centers=[[float(a), float(b)] for a, b in C])
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return Rc


def main():
    seed = int(sys.argv[1])
    kicks = [float(x) for x in sys.argv[2].split(",")]
    reps = int(sys.argv[3])
    budget_s = float(sys.argv[4])
    ckpt = os.path.join(HERE, sys.argv[5])

    rng = np.random.default_rng(seed)
    start_file = sys.argv[6] if len(sys.argv) > 6 else \
        os.path.join(PARENT, "centers_best.json")
    with open(start_file) as f:
        data = json.load(f)
    C_best = np.array(data["centers"], float)
    R_best = covering_radius(C_best)
    print(f"[seed {seed}] start ({os.path.basename(start_file)}) "
          f"R = {R_best:.10f}", flush=True)
    t0 = time.time()
    n_rounds = 0

    for kick in kicks:
        for rep in range(reps):
            if time.time() - t0 > budget_s:
                print(f"[seed {seed}] budget reached, stopping "
                      f"({n_rounds} rounds done)", flush=True)
                break
            C = C_best + rng.normal(0, kick, size=C_best.shape)
            R = covering_radius(C)
            C, R = lloyd_polish(C, R, sweeps=5, lams=(1.0, 0.5, 0.2),
                                seed=int(rng.integers(1, 10**9)), log=False)
            C, R = coordinate_descent(C, R, steps=(2e-4, 3e-5), passes=1)
            n_rounds += 1
            tag = "  <-- new best" if R < R_best - 1e-10 else ""
            print(f"[seed {seed}] kick={kick} rep={rep}: R = {R:.10f}{tag}"
                  f"   [{time.time()-t0:.0f}s]", flush=True)
            if R < R_best - 1e-10:
                R_best, C_best = R, C.copy()
                Rc = save_checkpoint(C_best, ckpt)
                print(f"[seed {seed}] checkpoint saved, recomputed R = "
                      f"{Rc:.10f}   [{time.time()-t0:.0f}s]", flush=True)
        else:
            continue
        break

    print(f"[seed {seed}] FINAL best R = {R_best:.10f} "
          f"({n_rounds} ILS rounds, {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
