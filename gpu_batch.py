"""
gpu_batch.py — CUDA batch pre-filter for ILS proposals (RTX 5070 Ti / Blackwell).

Strategy: the EXACT covering radius (geometry.covering_radius, ~40 ms) is the
bottleneck and stays the only authority for acceptance.  The GPU computes a
CHEAP LOWER BOUND of R(C) on a FIXED sample set:

    R_grid(C) = max_{p in P} min_i ||p - c_i||  <=  R(C)          (P fixed, dense)

R_grid underestimates R by at most (Lipschitz constant 1) x (sample spacing).
It is used ONLY to rank/rank-filter perturbation proposals; every accepted move
is still confirmed by the exact evaluator.  Zero risk to the proof chain.

Batch kernel: for B configs x N sample points x 100 centers, chunked over the
point axis to fit VRAM (B x chunk x 100 distance matrix per chunk).
"""
import json
import time
import numpy as np
import torch

# NOTE: this module is imported ONLY by the master process (gpu_hybrid imports
# it lazily inside main()).  Spawned workers must never import it — the CUDA
# DLLs blow up the page file (WinError 1455).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_sample_set(n_boundary=8192, n_disk=65536, seed=0):
    """Fixed dense sample set for D: exact boundary ring + sunflower interior."""
    th = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
    bnd = np.stack([np.cos(th), np.sin(th)], 1)
    # sunflower (Vogel) spiral: near-uniform over the disk
    k = np.arange(n_disk) + 0.5
    rr = np.sqrt(k / n_disk)
    tt = k * np.pi * (3 - np.sqrt(5.0))
    disk = np.stack([rr * np.cos(tt), rr * np.sin(tt)], 1)
    pts = np.vstack([bnd, disk])
    np.save("gpu_sample_points.npy", pts)
    return pts


class BatchEvaluator:
    def __init__(self, points_np, device=DEVICE, chunk=8192):
        # Respect whatever VRAM the desktop is leaving us: size the proposal
        # batch so that (b_chunk x chunk x 100 x 2 x 4B) fits in ~40% of FREE
        # VRAM.  evaluate() degrades further to CPU if CUDA OOM still hits.
        b_chunk = 512
        if device == "cuda":
            try:
                free_b, _total = torch.cuda.mem_get_info()
                need_per_cfg = chunk * 100 * 2 * 4 * 2      # diff + d2 tensors
                b_chunk = max(8, int(free_b * 0.4 / need_per_cfg))
            except Exception:
                b_chunk = 512
        self.b_chunk = b_chunk
        self.pts = torch.as_tensor(points_np, dtype=torch.float32, device=device)
        self.device = device
        self.chunk = chunk
        self.n = self.pts.shape[0]

    @torch.no_grad()
    def evaluate(self, configs_np):
        """configs_np: (B, 100, 2) float -> R_grid (B,) numpy.
        Auto-degrades: proposal batching sized to free VRAM; on CUDA OOM the
        whole evaluator moves to CPU (filter speed only, proof chain intact)."""
        configs_np = np.asarray(configs_np, np.float32)
        B = configs_np.shape[0]
        out = np.empty(B, dtype=np.float32)
        b0 = 0
        while b0 < B:
            b1 = min(b0 + self.b_chunk, B)
            try:
                out[b0:b1] = self._eval_block(configs_np[b0:b1])
            except torch.cuda.OutOfMemoryError:
                if self.device == "cuda":
                    print("[gpu_batch] CUDA OOM -> falling back to CPU",
                          flush=True)
                    self.device = "cpu"
                    self.pts = self.pts.cpu()
                    self.b_chunk = 64
                    continue            # retry same slice on CPU
                raise
            b0 = b1
        return out

    def _eval_block(self, cfg_block):
        C = torch.as_tensor(cfg_block, device=self.device)
        Bn = C.shape[0]
        best = torch.full((Bn,), -1.0, device=self.device, dtype=torch.float32)
        for s in range(0, self.n, self.chunk):
            P = self.pts[s:s + self.chunk]                       # (m, 2)
            d2 = (P[None, :, None, :] - C[:, None, :, :]).pow(2).sum(-1)   # (B, m, 100)
            g = d2.min(-1).values.sqrt().max(-1).values          # (B,)
            best = torch.maximum(best, g)
        return best.cpu().numpy()


def self_test(n_cfg=24, seed=3):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from geometry import covering_radius
    pts = build_sample_set()
    ev = BatchEvaluator(pts)
    rng = np.random.default_rng(seed)
    cfgs = rng.uniform(-1, 1, size=(n_cfg, 100, 2))
    t0 = time.time()
    rg = ev.evaluate(cfgs)
    t_gpu = time.time() - t0
    gaps = []
    t0 = time.time()
    for i in range(n_cfg):
        rex = covering_radius(cfgs[i])
        gaps.append(rex - rg[i])
    t_cpu = time.time() - t0
    gaps = np.array(gaps)
    print(f"GPU batch: {n_cfg} configs in {t_gpu:.2f}s ({n_cfg/t_gpu:.0f} cfg/s)")
    print(f"CPU exact (for comparison): {t_cpu/n_cfg*1000:.0f} ms/config")
    print(f"R_exact - R_grid: min {gaps.min():.5f}, max {gaps.max():.5f} (all >= 0 required)")
    assert (gaps >= -1e-6).all(), "R_grid exceeded R_exact — kernel bug!"
    print("self-test PASSED")
    return t_gpu, t_cpu / n_cfg


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    if DEVICE == "cuda":
        print(torch.cuda.get_device_name(0))
    self_test()
