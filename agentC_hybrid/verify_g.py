#!/usr/bin/env python3
"""
Diagnostics & independent verification for the hybrid bound.

1. Locate the anomalous second differences of the sampled g (suspect:
   arccos precision loss near b=0 where the argument is 1-O(b^2/4)...).
2. Closed-form check of g''(b) sign on a dense grid (concavity of g).
3. Independent computation of a(d) via polar integration (scipy.quad):
   A = 1/2 * integral over theta of [min(1,rho_+)^2 - rho_-^2] where
   rho_+- = d*cos(theta) +/- sqrt(r^2 - d^2 sin^2(theta))  (when real).
"""
import math
import numpy as np
from scipy.integrate import quad

PI = math.pi


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


def a_polar(d, r):
    """Independent lens area via polar integration around the origin."""
    def integrand(theta):
        s = math.sin(theta)
        disc = r * r - d * d * s * s
        if disc <= 0.0:
            return 0.0
        rt = math.sqrt(disc)
        c = math.cos(theta)
        rp = d * c + rt   # outer intersection of the ray with disk-r
        rm = d * c - rt   # inner
        upper = min(1.0, rp)
        if upper <= rm:
            return 0.0
        return upper * upper - rm * rm
    val, err = quad(integrand, 0.0, 2.0 * PI, limit=400, epsabs=1e-14, epsrel=1e-13)
    return 0.5 * val, err


# ----------------------------------------------------------------------
print("=" * 70)
r = 0.1075
bmax = 2.0 * math.asin(r)
N = 200001
b = np.linspace(0.0, bmax, N)
lo = np.full(N, 1.0 - r)
hi = np.full(N, math.sqrt(1.0 - r * r))
for _ in range(60):
    mid = 0.5 * (lo + hi)
    x = (mid * mid + 1.0 - r * r) / (2.0 * mid)
    bd = 2.0 * np.arccos(np.clip(x, -1.0, 1.0))
    less = bd < b
    lo = np.where(less, mid, lo)
    hi = np.where(less, hi, mid)
d = 0.5 * (lo + hi)
x1 = (d * d + r * r - 1.0) / (2.0 * d * r)
x2 = (d * d + 1.0 - r * r) / (2.0 * d)
t1 = r * r * np.arccos(np.clip(x1, -1.0, 1.0))
t2 = np.arccos(np.clip(x2, -1.0, 1.0))
prod = (-d + r + 1.0) * (d + r - 1.0) * (d - r + 1.0) * (d + r + 1.0)
g = t1 + t2 - 0.5 * np.sqrt(np.clip(prod, 0.0, None))

h = b[1] - b[0]
sd = g[2:] - 2.0 * g[1:-1] + g[:-2]
i = int(np.argmax(sd))
print(f"argmax second diff at index {i}, b = {b[i]:.6e} (b/bmax = {b[i]/bmax:.3e})")
print(f"  sd = {sd[i]:.6e}  ->  g'' ~ {sd[i]/h**2:.3f}")
print(f"  g[i-1..i+1] = {g[i-1]:.17f}, {g[i]:.17f}, {g[i+1]:.17f}")
diff = np.diff(g)
j = int(np.argmax(diff))
print(f"argmax diff(g) at index {j}, b = {b[j]:.6e}, diff = {diff[j]:.6e}")

# first few and last few sd values
print("sd[0:6]  =", np.array2string(sd[:6], precision=3))
print("sd[-6:]  =", np.array2string(sd[-6:], precision=3))
# interior sd (exclude first 200 points near b=0)
print(f"max sd for index >= 200: {sd[199:].max():.6e} at b = {b[199+int(np.argmax(sd[199:]))]:.6e}")
print(f"min sd overall: {sd.min():.6e} at b = {b[1+int(np.argmin(sd))]:.6e}")

# fix g[0] exactly and recheck
g[0] = PI * r * r
g[-1] = a_of_d(math.sqrt(1.0 - r * r), r)
sd2 = g[2:] - 2.0 * g[1:-1] + g[:-2]
print(f"\nafter fixing endpoints: max sd = {sd2.max():.6e} at index {int(np.argmax(sd2))+1}, "
      f"b = {b[1+int(np.argmax(sd2))]:.6e}")
print(f"monotone (diff <= 1e-14): {bool(np.all(np.diff(g) <= 1e-14))}")

# ----------------------------------------------------------------------
# closed-form g''(b) sign check
print("\n" + "=" * 70)
print("closed-form g''(b) sign check (double precision)")
def gpp_of_d(dv, r):
    """g''(b) as a function of d on the first branch."""
    beta = 1.0 - r * r
    # x = (d^2 + beta)/(2d): argument of arccos for b
    x = (dv * dv + beta) / (2.0 * dv)
    xp = (1.0 - beta / (dv * dv)) / 2.0
    xpp = beta / (dv ** 3)
    # y = (d^2 - beta)/(2 d r): argument of arccos for the r^2 term
    y = (dv * dv - beta) / (2.0 * dv * r)
    yp = (1.0 + beta / (dv * dv)) / (2.0 * r)
    ypp = -beta / (r * dv ** 3)
    sx = math.sqrt(max(0.0, 1.0 - x * x))
    sy = math.sqrt(max(0.0, 1.0 - y * y))
    bp = -2.0 * xp / sx
    bpp = -2.0 * (xpp * (1.0 - x * x) + x * xp * xp) / sx ** 3
    # a'(d)
    P = ((r + 1.0) ** 2 - dv * dv) * (dv * dv - (1.0 - r) ** 2)
    A = (r + 1.0) ** 2 - dv * dv
    B = dv * dv - (1.0 - r) ** 2
    Pp = -2.0 * dv * B + A * 2.0 * dv
    ap = -r * r * yp / sy - xp / sx - Pp / (4.0 * math.sqrt(P))
    # a''(d)
    app = (-r * r * (ypp / sy + yp * yp * y / sy ** 3)
           - (xpp / sx + xp * xp * x / sx ** 3)
           - (P * (-2.0 * B - 8.0 * dv * dv + 2.0 * A) - 0.5 * Pp * Pp) / (4.0 * P * math.sqrt(P)))
    return (app * bp * bp - ap * bpp) / bp ** 3

grid = np.linspace(1.0 - r, math.sqrt(1.0 - r * r), 100001)[1:-1]  # avoid exact endpoints
vals = np.array([gpp_of_d(dd, r) for dd in grid])
print(f"g'' over d-grid (99999 pts): min = {vals.min():.6f}, max = {vals.max():.6f}")
print(f"all negative: {bool(np.all(vals < 0))}")
k = int(np.argmax(vals))
print(f"  max at d = {grid[k]:.12f}, b = {b_of_d(grid[k], r):.12f}, g'' = {vals[k]:.6e}")

# cross-validate gpp against finite differences of the sampled g
def g_at(bb):
    return a_of_d(d_of_b(bb, r), r)
for btest in (0.01, 0.05, 0.1, 0.15, 0.2):
    hb = 1e-5
    fd = (g_at(btest + hb) - 2 * g_at(btest) + g_at(btest - hb)) / hb ** 2
    dt = d_of_b(btest, r)
    cl = gpp_of_d(dt, r)
    print(f"  b={btest:.3f}: g'' closed-form = {cl:.6f}, FD = {fd:.6f}")

# ----------------------------------------------------------------------
# independent polar-integration check of a(d) at key points
print("\n" + "=" * 70)
print("independent polar check of a(d):")
for (dd, rr) in [(math.sqrt(1 - 0.1075**2), 0.1075), (0.96942110, 0.1075),
                 (0.9675, 0.103449), (1.0, 0.103449), (0.9, 0.1075)]:
    a1 = a_of_d(dd, rr)
    a2, err = a_polar(dd, rr)
    print(f"  d={dd:.8f} r={rr}: closed={a1:.12f}  polar={a2:.12f}  diff={abs(a1-a2):.2e} (quad err {err:.1e})")

# g at the operating points
print("\ng at operating points:")
for (rr, m) in [(0.103449, 31), (0.1075, 30), (0.1080, 30)]:
    b0 = 2 * PI / m
    bmx = 2 * math.asin(rr)
    dt = d_of_b(min(b0, bmx * (1 - 1e-15)), rr)
    g1 = a_of_d(dt, rr)
    g2, err = a_polar(dt, rr)
    print(f"  r={rr} m={m}: d={dt:.10f} g_closed={g1:.12f} g_polar={g2:.12f} diff={abs(g1-g2):.2e}")
