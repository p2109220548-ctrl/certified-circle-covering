"""
gpu_hybrid.py — GPU-filtered iterated local search (the full machine).

Pipeline per round (P chains in parallel):
  1. MASTER (GPU): for each chain, sample B=512 kick-proposals around the chain's
     current best; rank them by R_grid (BatchEvaluator, fixed sample set).
  2. MASTER (CPU): exact-evaluate the top-k proposals of each chain (40 ms each);
     keep the truly best proposal.  (R_grid is only a filter; acceptance is exact.)
  3. WORKERS (CPU pool): lloyd_polish(sweeps=4) + coordinate_descent on the chosen
     proposal; return the improved (C, R).
  4. Update chain bests; checkpoint after every round.

Correctness: R(C) reported/accepted only from geometry.covering_radius.  The GPU
never decides acceptance — it only chooses WHICH proposals deserve exact checks.
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import covering_radius
from lloyd_cover import lloyd_polish
from construct import coordinate_descent
# NOTE: gpu_batch/torch are imported lazily inside main() — spawned workers
# must never pay the CUDA-DLL import cost (WinError 1455 page-file explosion).

KICKS = (7e-4, 4e-4, 2e-4)
B = 512          # proposals per chain per round
TOPK = 12        # exact-evaluated after GPU ranking


def _polish_task(args):
    C, R, seed = args
    C2, R2 = lloyd_polish(C, R, sweeps=4, lams=(1.0, 0.5, 0.2), seed=seed, log=False)
    C2, R2 = coordinate_descent(C2, R2, steps=(2e-4, 3e-5), passes=1)
    return C2, R2


def main(start_paths, hours=6.0, chains=8, workers=8, out="agentD_marathon"):
    from gpu_batch import build_sample_set, BatchEvaluator   # master-only
    rng = np.random.default_rng(2026)
    pts = build_sample_set()
    ev = BatchEvaluator(pts)
    print(f"[gpu_hybrid] sample points: {ev.n}, device: {ev.device}", flush=True)

    chains_C, chains_R = [], []
    for p in start_paths:
        d = json.load(open(p))
        C = np.array(d["centers"], float)
        chains_C.append(C)
        chains_R.append(covering_radius(C))
    chains_C = chains_C[:chains]
    chains_R = chains_R[:chains]
    gbest_i = int(np.argmin(chains_R))
    gbest_C, gbest_R = chains_C[gbest_i].copy(), chains_R[gbest_i]
    print(f"[gpu_hybrid] start best = {gbest_R:.9f}", flush=True)

    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "gpu_hybrid_checkpoint.json")
    t0 = time.time()
    rnd = 0
    pool = Pool(workers)
    try:
        while time.time() - t0 < hours * 3600:
          try:
            # step 1+2: GPU filter + exact confirm, per chain
            chosen = []
            for k in range(len(chains_C)):
                kicks = rng.choice(KICKS, size=B)
                props = chains_C[k][None, :, :] + rng.normal(
                    0, 1, size=(B, *chains_C[k].shape)) * kicks[:, None, None]
                rg = ev.evaluate(props)
                top = np.argsort(rg)[:TOPK]
                best_p, best_r = None, chains_R[k]
                for j in top:
                    rex = covering_radius(props[j])
                    if rex < best_r:
                        best_r, best_p = rex, props[j]
                chosen.append((best_p if best_p is not None else chains_C[k].copy(),
                               best_r))
            # step 3: parallel polish
            results = pool.map(_polish_task,
                               [(c, r, int(rng.integers(1 << 30)))
                                for c, r in chosen])
            for k, (C2, R2) in enumerate(results):
                if R2 < chains_R[k] - 1e-12:
                    chains_C[k], chains_R[k] = C2, R2
                if R2 < gbest_R - 1e-12:
                    gbest_C, gbest_R = C2.copy(), R2
                    json.dump(dict(n=len(gbest_C), radius=float(gbest_R),
                                   topology="gpu_hybrid",
                                   centers=[[float(a), float(b)]
                                            for a, b in gbest_C]),
                              open(os.path.join(out, "centers_gpu.json"), "w"),
                              indent=1)
            rnd += 1
            print(f"[round {rnd}] chain Rs: "
                  + " ".join(f"{r:.7f}" for r in chains_R)
                  + f" | best {gbest_R:.9f} | {time.time()-t0:.0f}s", flush=True)
            json.dump(dict(round=rnd, best_R=float(gbest_R),
                           chain_Rs=[float(r) for r in chains_R]),
                      open(ckpt, "w"))
          except Exception as e:
            # OOM in a worker or transient failure: rebuild the pool, keep the
            # chain state (it lives in the master) and continue the marathon.
            print(f"[round {rnd+1}] RECOVERABLE {type(e).__name__}: {e}; "
                  f"rebuilding pool", flush=True)
            try:
                pool.terminate()
            except Exception:
                pass
            pool = Pool(workers)
            time.sleep(5)
    finally:
        pool.terminate()
    print(f"[gpu_hybrid] done: best = {gbest_R:.9f}")


if __name__ == "__main__":
    starts = ["agentA_ub/centers_A.json", "centers_best.json"]
    main(starts, hours=float(os.environ.get("GPU_HOURS", "4.0")),
         chains=8, workers=8, out="agentD_marathon")
