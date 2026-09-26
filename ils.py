"""
ils.py — Iterated local search on top of the best configuration:
random kick -> Lloyd polish -> fine coordinate descent, keep the best.
"""
import json
import shutil
import time
import numpy as np
from geometry import covering_radius
from lloyd_cover import lloyd_polish
from construct import coordinate_descent


def main(iters=6, kick=0.004, seed=11):
    rng = np.random.default_rng(seed)
    data = json.load(open("centers_best.json"))
    C_best = np.array(data["centers"], float)
    R_best = covering_radius(C_best)
    print(f"start R = {R_best:.9f}", flush=True)
    t0 = time.time()

    for it in range(iters):
        C = C_best + rng.normal(0, kick, size=C_best.shape)
        R = covering_radius(C)
        C, R = lloyd_polish(C, R, sweeps=4, log=False)
        C, R = coordinate_descent(C, R, steps=(2e-4, 3e-5), passes=1)
        tag = "  <-- new best" if R < R_best - 1e-9 else ""
        print(f"ILS {it}: R = {R:.9f}{tag}   [{time.time()-t0:.0f}s]", flush=True)
        if R < R_best - 1e-9:
            R_best, C_best = R, C.copy()
            out = dict(n=len(C_best), radius=float(R_best),
                       topology=data.get("topology"),
                       centers=[[float(a), float(b)] for a, b in C_best])
            json.dump(out, open("centers_best.json", "w"), indent=1)
            print("  centers_best.json updated", flush=True)

    print(f"\nfinal best R = {R_best:.9f}")


if __name__ == "__main__":
    main()
