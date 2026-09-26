# -*- coding: utf-8 -*-
"""
run_marathon3.py — Upper-bound marathon, round 3.

Seeds: 4x current GPU best (agentD_marathon/centers_gpu.json), 1x agentA best,
       3x NEW hexagonal-lattice seeds (different basin from ring topologies).
Then runs gpu_hybrid.main for GPU_HOURS (default 3.0), 8 chains, 8 workers.
GPU proposes, CPU exact evaluator decides (unchanged protocol).
"""
import json
import os
import sys
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

OUT = HERE


def hex_seed(a, theta0, n_target=100, r_expand=0.118, tag="hex"):
    """Triangular-lattice covering seed, pruned to n_target centers."""
    from geometry import covering_radius
    u1 = np.array([a, 0.0])
    u2 = np.array([a / 2, a * np.sqrt(3) / 2])
    R = np.array([[np.cos(theta0), -np.sin(theta0)],
                  [np.sin(theta0), np.cos(theta0)]])
    rng = int(np.ceil((1 + r_expand) / a)) + 2
    pts = []
    for i in range(-rng, rng + 1):
        for j in range(-rng, rng + 1):
            p = R @ (i * u1 + j * u2)
            if np.linalg.norm(p) <= 1 + r_expand:
                pts.append(p)
    pts = np.array(pts)
    # greedy prune: repeatedly drop the point whose removal hurts least
    while len(pts) > n_target:
        best_r, best_k = np.inf, -1
        for k in range(len(pts)):
            r = covering_radius(np.delete(pts, k, axis=0))
            if r < best_r:
                best_r, best_k = r, k
        pts = np.delete(pts, best_k, axis=0)
    C = pts
    R_exact = covering_radius(C)
    path = os.path.join(OUT, f"centers_{tag}_{tag and ''}{a:.4f}_{int(theta0*1000)}.json")
    json.dump(dict(n=len(C), radius=float(R_exact), topology="hex_lattice",
                   centers=[[float(x), float(y)] for x, y in C]),
              open(path, "w"), indent=1)
    print(f"[seed] {os.path.basename(path)}: n={len(C)} R={R_exact:.7f}", flush=True)
    return path


def main():
    from gpu_hybrid import main as marathon
    t0 = time.time()
    seeds = []
    try:
        seeds.append(hex_seed(0.2066, 0.0))
        seeds.append(hex_seed(0.2090, 0.13))
    except Exception as e:
        print(f"[seed] hex generation failed: {e}", flush=True)
    base = os.path.join(BASE, "agentD_marathon", "centers_gpu.json")
    latest = os.path.join(OUT, "centers_gpu.json")   # newest marathon output
    if os.path.exists(latest):
        base = latest
    aub = os.path.join(BASE, "agentA_ub", "centers_A.json")
    old = os.path.join(BASE, "agentD_marathon", "centers_gpu.json")
    starts = [base] * 4 + [old, base] + \
             ([aub] if os.path.exists(aub) else []) + seeds
    starts = [p for p in starts if os.path.exists(p)][:8]
    print(f"[marathon3] {len(starts)} chains, seeds ready at {time.time()-t0:.0f}s",
          flush=True)
    hours = float(os.environ.get("GPU_HOURS", "3.0"))
    workers = int(os.environ.get("GPU_WORKERS", "6"))
    marathon(starts, hours=hours, chains=len(starts), workers=workers, out=HERE)


if __name__ == "__main__":
    main()
