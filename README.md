# Certified Bounds for Covering the Unit Disk with 100 Equal Circles

**r_D(100) — certified window (2026-09-27):**

```text
0.1066235  ≤  r_D(100)  ≤  0.1169288      (width 0.0103)
```

- **Upper bound** `0.1169288 = 125551361/2^30`: explicit 100-center configuration
  (hexagonal-basin optimum, GPU-assisted search) + integer quadtree covering
  certificate (VERIFIED, 11237 boxes, 0.3 s re-verification, integer arithmetic only).
  Literature value was `1/√67 ≈ 0.12217` — a **4.28% improvement**.
- **Lower bound** `0.1066235`: K=320-ring measure LP + cutting planes, certified
  by a frozen-measure rigorous sup protocol (`sup_upper = 0.009999999016 < 1/100`,
  explicit margin 9.8e-10).
- **New theorem (rotation-averaging optimality):** for ANY Borel probability
  measure μ, `max_c μ(B(c,r)) ≥ V_radial(r)` — the radial family is optimal in
  the entire measure class. Hence **0.10664 is the exact ceiling of all
  measure/duality arguments**: further progress requires structural
  (non-measure) methods. See `agentH_radial_closure/OPTIMALITY_THEOREM.md`.

## Honest scope

Determining r_D(100) exactly with a global-optimality proof (requirements #1+#4
of the original problem) is an **open problem** — optimality is not proven for
any n ≥ 11. This repository makes no claim about the exact value. What IS
delivered: explicit construction (#2), a machine-verifiable covering proof (#3),
the exact-radius format at α = 1/10 (#1), and certified two-sided bounds with a
fully reproducible certificate chain.

## Repository map

| Path | Content |
|---|---|
| `PROJECT_SUMMARY.md` | Full work archive (Chinese): problem transcript, timeline, certificates, methodology lessons |
| `REPORT.md` | Engineering report, incl. audit sections §9–§10 (two retracted bounds, one repaired certificate) |
| `geometry.py` | Exact covering-radius evaluator (complete candidate set: Voronoi vertices, boundary arcs, antipodes) |
| `verify_cover.py` → `certificate.json` | Integer quadtree covering certificate — zero dependencies, re-runnable |
| `construct.py` / `lloyd_cover.py` / `ils.py` | Construction search suite |
| `gpu_batch.py` / `gpu_hybrid.py` | VRAM-adaptive GPU pre-filter + exact-eval acceptance pipeline |
| `agentB_measure/` | Radial measure lower bound (verified protocol) |
| `agentF_upgrade/` | Cutting-plane upgrade (⚠️ its JSON sup_upper > 1/100 — fall-through bug, kept as audit trail) |
| `agentH_radial_closure/` | Optimality theorem + K=320 machine + frozen-w certification + Voronoi scouting |
| `agentI_marathon3/` | Hexagonal-seed marathon (the 0.1169635 breakthrough basin) |
| `centers_*.json`, `agentI_marathon3/centers_gpu*.json` | Explicit certified configurations |

## Reproduce

```bash
python verify_cover.py agentI_marathon3/centers_gpu.json   # upper bound, VERIFIED expected
python agentH_radial_closure/certify_frozen.py both        # lower bound, ~5 min (needs numpy/scipy)
```

The certificate verifier itself has **zero dependencies** (pure integer arithmetic).

## Methodology ledger (the part worth reading)

1. Direction of every relaxation must be proven via an explicit pointwise map.
2. Nested-grid monotone convergence checks expose fake suprema.
3. Exclusion arguments must aggregate with `max` over viable configurations.
4. lower ≤ truth ≤ upper: any "bound" violating this dies without review.
5. **Certification loops must never fall through** — the flagship certificate
   once shipped with `sup_upper = 0.010000091 > 1/100` because of exactly this.
6. **Freeze the measure during certification**; re-solving the LP re-saturates
   the constraints and masks the true margin.
7. Chunk big kernels; prune LP solutions to their active support before rigorous
   verification; serialize memory-heavy parallel jobs.
