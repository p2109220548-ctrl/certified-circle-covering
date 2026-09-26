"""
construct.py — Search for an explicit 100-center configuration covering the unit
disk with small radius R_D(C).

Stage 1: circularized-hexagonal ring families (few parameters) optimized by
         Nelder-Mead on the EXACT covering radius.
Stage 2: coordinate descent on all 200 coordinates (exact objective).
Output:  centers_best.json  (best configuration found)
"""
import json
import time
import numpy as np
from scipy.optimize import minimize
from geometry import covering_radius

TOPOLOGIES = [
    [1, 6, 12, 18, 24, 39],
    [1, 6, 12, 18, 27, 36],
    [1, 6, 12, 20, 28, 33],
    [1, 7, 13, 19, 27, 33],
    [1, 8, 14, 20, 26, 31],
    [0, 7, 13, 19, 28, 33],
    [1, 6, 13, 20, 29, 31],
    [1, 9, 15, 22, 28, 25],
    [1, 6, 12, 18, 25, 38],
    [1, 10, 16, 22, 28, 23],
]


def build(m, rho, psi):
    pts = []
    if m[0] >= 1:
        pts.append(np.zeros((m[0], 2)))
    for k in range(1, len(m)):
        a = np.arange(m[k]) * (2 * np.pi / m[k]) + psi[k - 1]
        r = min(max(rho[k - 1], 0.02), 1.15)
        pts.append(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    return np.vstack(pts)


def pack_x(m, rho, psi):
    return np.concatenate([rho, psi])


def unpack_x(m, x):
    K = len(m) - 1
    return x[:K], x[K:]


def objective(x, m):
    rho, psi = unpack_x(m, x)
    return covering_radius(build(m, rho, psi))


def init_x(m, rng, randomize):
    K = len(m) - 1
    rho = np.array([0.185 * k for k in range(1, K + 1)])
    psi = np.array([(k % 2) * np.pi / m[k + 1] for k in range(K)])
    if randomize:
        psi = psi + rng.uniform(0, 2 * np.pi / max(m[1], 1), size=K)
        rho = rho * rng.uniform(0.92, 1.08)
    return np.concatenate([rho, psi])


def coordinate_descent(C, R, steps=(2e-3, 4e-4), passes=2):
    C = C.copy()
    for step in steps:
        for _ in range(passes):
            improved = False
            for i in range(len(C)):
                for dim in (0, 1):
                    for h in (step, -step, 4 * step, -4 * step):
                        C[i, dim] += h
                        R2 = covering_radius(C)
                        if R2 < R - 1e-12:
                            R = R2
                            improved = True
                        else:
                            C[i, dim] -= h
            if not improved:
                break
    return C, R


def main():
    rng = np.random.default_rng(7)
    results = []
    t0 = time.time()
    for m in TOPOLOGIES:
        assert sum(m) == 100, m
        best = None
        for start in range(2):
            x0 = init_x(m, rng, randomize=(start > 0))
            res = minimize(objective, x0, args=(m,), method="Nelder-Mead",
                           options=dict(maxiter=2500, xatol=1e-8, fatol=1e-11,
                                        adaptive=True))
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        R, x = best
        rho, psi = unpack_x(m, x)
        C = build(m, rho, psi)
        results.append((R, m, C))
        print(f"topology {m}: R = {R:.7f}   [{time.time()-t0:.0f}s]", flush=True)

    results.sort(key=lambda t: t[0])
    print("\n--- coordinate descent on top 3 ---", flush=True)
    final = []
    for R, m, C in results[:3]:
        C2, R2 = coordinate_descent(C, R)
        final.append((R2, m, C2))
        print(f"polished {m}: R = {R:.7f} -> {R2:.7f}   [{time.time()-t0:.0f}s]",
              flush=True)

    final.sort(key=lambda t: t[0])
    R_best, m_best, C_best = final[0]
    print(f"\nBEST: R = {R_best:.8f}  with topology {m_best}", flush=True)
    out = dict(n=len(C_best), radius=R_best, topology=m_best,
               centers=[[float(a), float(b)] for a, b in C_best])
    with open("centers_best.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved centers_best.json", flush=True)


if __name__ == "__main__":
    main()
