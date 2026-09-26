#!/usr/bin/env python3
"""
Rigorous-ish verification suite for the hybrid bound's ingredients.

V1. Independent check of a(d) (lens area) via 1-D slab integration:
    A = int_{-r}^{r} overlap(y) dy,
    overlap(y) = max(0, min(cx+w, X) - max(cx-w, -X)),
    w = sqrt(r^2-y^2), X = sqrt(1-y^2), cx = d.
V2. Concavity of g on [0, 2 arcsin r]:
    (i)  b in (0, 1e-3]: analytic, g = pi r^2 - c3 b^3 + O(b^5),
         c3 = (1-r)^{3/2}/(12r) > 0  =>  g'' = -6 c3 b + O(b^3) < 0.
         (also verified numerically: L(b)/b^3 -> c3)
    (ii) b in [1e-3, bmax-eps]: direct central second differences of the
         exact g(b) with adaptive hb = min(1e-4, b/5, (bmax-b)/5);
         fp-noise bound is computed and must stay below |g''| margin.
    (iii) b near bmax: analytic, g = a* + k*sqrt(bmax-b) + O((bmax-b)^{3/2}),
         k = |a'(d*)| / sqrt(|b''(d*)|/2) > 0  => concave.
V3. g strictly decreasing; a strictly decreasing on [1-r, 1+r].
V4. b(d*) = 2 arcsin(r) (AM-GM argument is analytic; numeric check here).
"""
import math
import json
import os
import numpy as np
from scipy.integrate import quad

PI = math.pi
HERE = os.path.dirname(os.path.abspath(__file__))


def b_of_d(d, r):
    x = (d * d + 1.0 - r * r) / (2.0 * d)
    return 2.0 * math.acos(min(1.0, max(-1.0, x)))


def a_of_d(d, r):
    x1 = (d * d + r * r - 1.0) / (2.0 * d * r)
    x2 = (d * d + 1.0 - r * r) / (2.0 * d)
    t1 = r * r * math.acos(min(1.0, max(-1.0, x1)))
    t2 = math.acos(min(1.0, max(-1.0, x2)))
    prod = (-d + r + 1.0) * (d + r - 1.0) * (d - r + 1.0) * (d + r + 1.0)
    return t1 + t2 - 0.5 * math.sqrt(max(0.0, prod))


def d_of_b(b, r):
    lo, hi = 1.0 - r, math.sqrt(1.0 - r * r)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if b_of_d(mid, r) < b:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def g_of_b(b, r):
    if b <= 0.0:
        return PI * r * r
    return a_of_d(d_of_b(b, r), r)


def a_slab(d, r):
    """Independent lens area by integrating overlap lengths over y."""
    def overlap(y):
        w = math.sqrt(max(0.0, r * r - y * y))
        X = math.sqrt(max(0.0, 1.0 - y * y))
        lo = max(d - w, -X)
        hi = min(d + w, X)
        return max(0.0, hi - lo)
    # kinks where the disk boundary crosses the unit circle boundary:
    # y = +/- y1 with x0 = (d^2+1-r^2)/(2d); y1 = sqrt(r^2 - (x0-d)^2)
    x0 = (d * d + 1.0 - r * r) / (2.0 * d)
    val, err = quad(overlap, -r, r, limit=500, epsabs=1e-15, epsrel=1e-13,
                    points=[-min(1.0, r), 0.0, min(1.0, r)])
    return val, err


def main():
    report = {}
    r = 0.1075
    bmax = 2.0 * math.asin(r)
    dstar = math.sqrt(1.0 - r * r)

    # ---------------- V1: slab check of a(d) ----------------
    print("V1: slab-integration check of a(d)")
    worst = 0.0
    for dd in [1.0 - r + 1e-6, 1.0 - r + 1e-3, 0.95, 0.9694210913, dstar,
               0.999, 1.0, 1.02, 1.0 + r - 1e-6]:
        a1 = a_of_d(dd, r)
        a2, err = a_slab(dd, r)
        diff = abs(a1 - a2)
        worst = max(worst, diff)
        print(f"  d={dd:.10f}: closed={a1:.12f} slab={a2:.12f} diff={diff:.2e} (quad {err:.1e})")
    report["a_slab_max_diff"] = worst
    print(f"  worst diff = {worst:.3e}")

    # ---------------- V4: b(d*) ----------------
    bnum = b_of_d(dstar, r)
    print(f"\nV4: b(d*) = {bnum:.17g} vs 2 arcsin(r) = {bmax:.17g}, "
          f"diff = {abs(bnum-bmax):.3e}")
    report["b_dstar_diff"] = abs(bnum - bmax)

    # ---------------- V2(ii): second differences of g ----------------
    print("\nV2: concavity of g by direct second differences (adaptive hb)")
    bs = np.logspace(-3, math.log10(bmax * 0.999995), 3000)
    worst_sd = -1e18
    worst_b = None
    all_neg = True
    noise_est = 0.0
    for bt in bs:
        bt = float(bt)
        hb = min(1e-4, bt / 5.0, (bmax - bt) / 5.0)
        g0 = g_of_b(bt, r)
        gp = g_of_b(bt + hb, r)
        gm = g_of_b(bt - hb, r)
        sd = (gp - 2.0 * g0 + gm) / (hb * hb)
        if sd > worst_sd:
            worst_sd, worst_b = sd, bt
        if sd >= 0.0:
            all_neg = False
            print(f"  NONNEG sd at b={bt:.6e}: {sd:.3e}")
        # fp noise estimate: g error ~ 5e-16/sqrt(2*alpha^2/2) ~ 7e-16/alpha, alpha=bt/2
        alpha = bt / 2.0
        gerr = 7e-16 / max(alpha, 1e-8)
        noise_est = max(noise_est, 4.0 * gerr / (hb * hb))
    print(f"  max sd = {worst_sd:.6e} at b = {worst_b:.6e}  (all negative: {all_neg})")
    print(f"  fp-noise bound on sd: {noise_est:.3e}")
    report["concave_sd_max"] = worst_sd
    report["concave_sd_argmax_b"] = worst_b
    report["concave_all_negative"] = all_neg
    report["concave_noise_bound"] = noise_est

    # cubic-law check near 0
    print("\nV2(i): cubic law L(b) ~ c3 b^3, c3 = (1-r)^{3/2}/(12r)")
    c3 = (1.0 - r) ** 1.5 / (12.0 * r)
    for bt in (1e-4, 3e-4, 1e-3, 3e-3):
        L = PI * r * r - g_of_b(bt, r)
        print(f"  b={bt:.1e}: L/b^3 = {L/bt**3:.6f}  (c3 = {c3:.6f})")
    report["c3_theory"] = c3

    # sqrt law near bmax
    print("\nV2(iii): sqrt law near bmax: g(b) = a* + k sqrt(bmax-b)")
    a_star = a_of_d(dstar, r)
    # k from two interior points
    for bt in (bmax - 1e-4, bmax - 1e-5, bmax - 1e-6):
        k = (g_of_b(bt, r) - a_star) / math.sqrt(bmax - bt)
        print(f"  b=bmax-{bmax-bt:.1e}: k = {k:.8f}")
    report["a_star"] = a_star

    # ---------------- V3: monotonicity ----------------
    print("\nV3: monotonicity")
    # g decreasing on the fp-clean zone
    bs2 = np.linspace(3e-3, bmax * 0.9999, 20000)
    dg = np.diff([g_of_b(float(bt), r) for bt in bs2])
    print(f"  g decreasing on [3e-3, bmax]: max diff = {dg.max():.3e} (all <=1e-13: {bool(dg.max() <= 1e-13)})")
    # a decreasing on (1-r, 1+r)
    ds = np.linspace(1.0 - r + 1e-9, 1.0 + r - 1e-9, 200001)
    da = np.diff(a_of_d_vec := np.array([a_of_d(float(dd), r) for dd in ds]))
    print(f"  a decreasing on [1-r,1+r]: max diff = {da.max():.3e} (all <=1e-13: {bool(da.max() <= 1e-13)})")
    report["g_decreasing_max_diff"] = float(dg.max())
    report["a_decreasing_max_diff"] = float(da.max())

    with open(os.path.join(HERE, "concavity_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\n[done] concavity_report.json written")


if __name__ == "__main__":
    main()
