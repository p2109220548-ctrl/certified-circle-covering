# -*- coding: utf-8 -*-
"""
marathon2.py — Topology-diversity construction search (no agent, resumable).

Phase 1: ring-NM on 6 unexplored topologies (fast triage).
Phase 2: GPU-filtered ILS (gpu_hybrid) seeded with the best fresh topologies
         + the current global best.  Checkpoints on every improvement.
"""
import os
import sys
import json
import time
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from geometry import covering_radius
from construct import build, unpack_x, init_x, objective
from gpu_hybrid import main as gpu_main

OUT = os.path.join(HERE, "agentG_topo")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
CKPT = os.path.join(OUT, "marathon2_checkpoint.json")


def log(m):
    print("[%6.1fs] %s" % (time.time() - T0, m), flush=True)


TOPOS = [
    [1, 6, 12, 19, 28, 34],
    [1, 8, 14, 20, 25, 32],
    [1, 7, 13, 20, 26, 33],
    [1, 6, 13, 19, 27, 34],
    [1, 9, 16, 21, 26, 27],
    [0, 8, 14, 21, 27, 30],
]


def phase1():
    rng = np.random.default_rng(99)
    results = []
    for m in TOPOS:
        assert sum(m) == 100, m
        best = None
        for start in range(2):
            x0 = init_x(m, rng, randomize=(start > 0))
            res = minimize(objective, x0, args=(m,), method="Nelder-Mead",
                           options=dict(maxiter=1200, xatol=1e-8, fatol=1e-11,
                                        adaptive=True))
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x)
        R, x = best
        rho, psi = unpack_x(m, x)
        C = build(m, rho, psi)
        results.append((R, m, C))
        log("topo %s: R = %.7f" % (m, R))
        json.dump(dict(round="phase1", best_R=float(min(r[0] for r in results))),
                  open(CKPT, "w"))
    results.sort(key=lambda t: t[0])
    # save top-3 fresh configs
    starts = []
    for i, (R, m, C) in enumerate(results[:3]):
        p = os.path.join(OUT, "fresh_%d.json" % i)
        json.dump(dict(n=100, radius=float(R), topology=m,
                       centers=[[float(a), float(b)] for a, b in C]),
                  open(p, "w"), indent=1)
        starts.append(p)
    return starts


def main():
    log("=== marathon2: topology diversity ===")
    existing = [os.path.join(OUT, "fresh_%d.json" % i) for i in range(3)]
    if all(os.path.exists(p) for p in existing):
        starts = existing
        log("phase 1 skipped (fresh_0/1/2.json already on disk)")
    else:
        starts = phase1()
    log("phase-1 fresh starts: %s" % starts)
    starts.append(os.path.join(HERE, "agentD_marathon", "centers_gpu.json"))
    log("phase 2: GPU-filtered ILS, 3h, starts = fresh x3 + global best")
    gpu_main(starts, hours=3.0, chains=4, workers=6, out=OUT)
    log("marathon2 done")


if __name__ == "__main__":
    main()
