"""
hybrid_corrected.py — CORRECTED area+arc hybrid lower bound.

Two fixes over the original agentC_hybrid formulation (both found in audit):
  (1) AGGREGATION: a covering with m crossers satisfies
        total area <= T(m,r) = (100-m)*pi*r^2 + m*ghat(2*pi/m)
      and total area >= pi.  Hence covering requires  max_m T(m,r) >= pi
      over all VIABLE m.  Exclusion needs max_m T(m,r) < pi  (the original
      code used min_m — wrong direction, unsound).
  (2) GEOMETRIC M-CAP: crossers (d > 1-r) cannot cover the open disk of
      radius 1-2r  (their reach is d - r > 1-2r).  That inner disk is covered
      by the 100-m interior disks alone, so (by area)
        (100-m)*pi*r^2 >= pi*(1-2*r)^2   =>   m <= 100 - (1-2r)^2/r^2.
      Viable m range: [ceil(pi/arcsin r), min(100, floor(100-(1-2r)^2/r^2))].

ghat = concave majorant of g(b) = max{a(d): b(d) = b}, b in [0, 2 arcsin r]
(g decreasing; ghat >= g makes Jensen sound for ANY g shape).
"""
import json
import numpy as np
from scipy.optimize import brentq

PI = float(np.pi)


def a_of_d(d, r):
    d = np.asarray(d, float)
    out = np.zeros_like(d)
    lo = d <= 1 - r
    hi = d >= 1 + r
    mid = ~lo & ~hi
    out[lo] = PI * r * r
    dm = d[mid]
    out[mid] = (r * r * np.arccos(np.clip((dm * dm + r * r - 1) / (2 * dm * r), -1, 1))
                + np.arccos(np.clip((dm * dm + 1 - r * r) / (2 * dm), -1, 1))
                - 0.5 * np.sqrt(np.clip((-dm + r + 1) * (dm + r - 1) * (dm - r + 1) * (dm + r + 1), 0, None)))
    return out


def b_of_d(d, r):
    d = np.asarray(d, float)
    out = np.zeros_like(d)
    live = (np.abs(d - 1) < r)
    out[live] = 2 * np.arccos(np.clip((d[live] ** 2 + 1 - r * r) / (2 * d[live]), -1, 1))
    return out


def g_curve(r, n=200000):
    """g(b) on a dense grid of b in [0, 2 arcsin r], first branch d in [1-r, sqrt(1-r^2)]."""
    d1, d2 = 1 - r, np.sqrt(1 - r * r)
    ds = np.linspace(d1, d2, n)
    bs = b_of_d(ds, r)          # monotone increasing on this branch
    As = a_of_d(ds, r)
    order = np.argsort(bs)
    return bs[order], As[order]


def ghat_eval(b, bs, gs):
    """Concave majorant of g evaluated at b (upper hull, computed on the grid)."""
    # upper concave hull via monotone scan (Andrew-style on upper envelope)
    pts = np.stack([bs, gs], 1)
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()       # middle point below/on chord -> not on upper hull
            else:
                break
        hull.append(tuple(p))
    hx = np.array([h[0] for h in hull])
    hy = np.array([h[1] for h in hull])
    b = np.atleast_1d(b)
    out = np.interp(b, hx, hy)
    out = np.minimum(out, PI * (b / max(b.max(), 1e-300)) ** 0 + 0)  # placeholder no-op
    return np.maximum(out, 0.0), hx, hy


def T_max_over_m(r, bs, gs, hull_x, hull_y):
    bmax = 2 * np.arcsin(r)
    m_min = int(np.ceil(PI / np.arcsin(r)))
    m_cap = 100 - (1 - 2 * r) ** 2 / (r * r)
    m_max = int(min(100, np.floor(m_cap)))
    if m_max < m_min:
        return float("-inf"), m_min, m_max, None
    Ts = []
    for m in range(m_min, m_max + 1):
        bload = 2 * PI / m
        if bload > bmax + 1e-15:
            continue
        gh, _, _ = ghat_eval(min(bload, bmax), bs, gs)
        # ghat must be evaluated against THIS r's hull; pass grids instead
        Ts.append((100 - m) * PI * r * r + m * gh[0])
    if not Ts:
        return float("-inf"), m_min, m_max, None
    imax = int(np.argmax(Ts))
    return max(Ts), m_min, m_max, (m_min + imax)


def certify(r_lo=0.095, r_hi=0.112, n_g=200000):
    def exc(r):
        bs, gs = g_curve(r, n_g)
        hull = []
        for x, y in zip(bs, gs):
            while len(hull) >= 2:
                (x1, y1), (x2, y2) = hull[-2], hull[-1]
                if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                    hull.pop()
                else:
                    break
            hull.append((x, y))
        hx = np.array([h[0] for h in hull]); hy = np.array([h[1] for h in hull])
        bmax = 2 * np.arcsin(r)
        m_min = int(np.ceil(PI / np.arcsin(r)))
        m_cap = 100 - (1 - 2 * r) ** 2 / (r * r)
        m_max = int(min(100, np.floor(m_cap)))
        if m_max < m_min:
            return True, None
        best, bm = float("-inf"), None
        for m in range(m_min, m_max + 1):
            bl = 2 * PI / m
            if bl > bmax:
                continue
            gh = float(np.interp(bl, hx, hy))
            T = (100 - m) * PI * r * r + m * gh
            if T > best:
                best, bm = T, m
        return (best < PI), (best, bm, m_min, m_max)

    # sanity: at r where a real covering exists (0.11785), must NOT be excluded
    ok, info = exc(0.1179)
    assert not ok, "soundness: real covering radius must not be excluded"
    print(f"sanity r=0.1179: not excluded (T_max={info[0]:.4f}, argmax m={info[1]}, "
          f"m in [{info[2]},{info[3]}])", flush=True)

    a, b = r_lo, r_hi
    ea, _ = exc(a); eb, _ = exc(b)
    assert ea and (not eb), "bracket: expect excluded at r_lo, feasible at r_hi"
    for _ in range(40):
        m = 0.5 * (a + b)
        e, _ = exc(m)
        if e:
            a = m
        else:
            b = m
    return a, b


if __name__ == "__main__":
    a, b = certify()
    safe = float(np.floor(a * 1e6) / 1e6) - 5e-6
    print(f"\ncorrected hybrid crossing: ({a:.9f}, {b:.9f})")
    print(f"CERTIFIED (corrected): r_D(100) > {safe:.6f}")
    print(f"comparison: radial measure 0.105445 | old (unsound) hybrid claim 0.106071")
    json.dump(dict(threshold_exact=float(a), threshold_reported=safe,
                   note=("corrected aggregation (max over viable m) + geometric "
                         "m-cap 100-(1-2r)^2/r^2; supersedes unsound min-over-m")),
              open("agentC_hybrid/hybrid_certificate_corrected.json", "w"), indent=1)
