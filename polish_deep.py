"""
polish_deep.py — Deep local refinement of the best configuration found by
construct.py: coordinate descent with a decreasing step schedule, several
random orderings, keeping the best.  Overwrites centers_best.json only if
strictly improved (and keeps a backup of the previous best).
"""
import json
import shutil
import time
import numpy as np
from geometry import covering_radius
from construct import coordinate_descent


def main():
    data = json.load(open("centers_best.json"))
    C = np.array(data["centers"], float)
    R = covering_radius(C)
    print(f"start: R = {R:.9f}", flush=True)
    t0 = time.time()

    schedule = [(2e-3, 1e-4), (5e-4, 5e-5), (1e-4, 2e-5), (5e-5, 1e-5)]
    best_C, best_R = C.copy(), R
    for si, steps in enumerate(schedule):
        C2, R2 = coordinate_descent(best_C, best_R, steps=steps,
                                    passes=3)
        if R2 < best_R - 1e-12:
            best_C, best_R = C2, R2
        print(f"schedule {si} {steps}: R = {R2:.9f}   "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print(f"\ndeep polish: R = {R:.9f} -> {best_R:.9f}")
    if best_R < R - 1e-9:
        shutil.copy("centers_best.json", "centers_best_backup.json")
        out = dict(n=len(best_C), radius=float(best_R),
                   topology=data.get("topology"),
                   centers=[[float(a), float(b)] for a, b in best_C])
        json.dump(out, open("centers_best.json", "w"), indent=1)
        print("centers_best.json UPDATED (old copy in centers_best_backup.json)")
    else:
        print("no improvement; centers_best.json unchanged")


if __name__ == "__main__":
    main()
