# -*- coding: utf-8 -*-
"""Standalone final certification: reload (t, w) from the certificate, re-bisect r
with w frozen under the rigorous cap=target protocol (finer grid), and rewrite
measure_certificate.json with the tightest certified r."""
import numpy as np, json, os, time, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_bound import rigorous_sup, lp_feasible, log, OUT

target = 1.0 / 100.0
T0 = time.time()

with open(os.path.join(OUT, "measure_certificate.json")) as f:
    cert = json.load(f)
t_vec = np.array(cert["t"], float)
w0 = np.array(cert["w"], float)
assert abs(w0.sum() - 1.0) < 1e-9 and (w0 >= 0).all()

def ok_at(r, h0=5e-4, min_len=2e-10):
    ub, info = rigorous_sup(t_vec, w0, r, h0=h0, cap=target, min_len=min_len,
                            max_rounds=400)
    return ub <= target, ub, info

# bracket
hi = cert["r"] + 1e-4          # push hi above current cert to see if we can gain back
ok_h, ub_h, _ = ok_at(hi)
log("probe hi=%.9f -> sup=%.12f feasible=%s" % (hi, ub_h, ok_h))
lo = cert["r"] - 5e-4
ok_l, ub_l, _ = ok_at(lo)
while not ok_l:
    lo -= 5e-4
    ok_l, ub_l, _ = ok_at(lo)
log("bracket: lo=%.9f OK (sup=%.12f), hi=%.9f %s" % (lo, ub_l, hi, "OK" if ok_h else "violated"))

for i in range(45):
    mid = 0.5 * (lo + hi)
    ok_m, ub_m, info_m = ok_at(mid)
    if ok_m:
        lo = mid
    else:
        hi = mid
    if hi - lo < 2e-9:
        break
r_cert = lo
ok_f, ub_f, info_f = ok_at(r_cert)
log("CERTIFIED r=%.9f  sup_upper=%.13f  n_grid=%d  (%.1fs)"
    % (r_cert, ub_f, info_f['n_grid'], time.time() - T0))

# extra-strict re-verify: 4x finer base grid + smaller floor
ub_g, info_g = rigorous_sup(t_vec, w0, r_cert, h0=1.25e-4, cap=target,
                            min_len=1e-10, max_rounds=500)
log("strict re-verify h0=1.25e-4: sup_upper=%.13f  n_grid=%d" % (ub_g, info_g['n_grid']))
while ub_g > target:
    r_cert -= 1e-6
    ub_g, info_g = rigorous_sup(t_vec, w0, r_cert, h0=1.25e-4, cap=target,
                                min_len=1e-10, max_rounds=500)
    log("  shrink -> r=%.9f sup=%.13f" % (r_cert, ub_g))

cert["r"] = float(r_cert)
cert["sup_upper"] = float(ub_g)
cert["grid"].update(h0_final=1.25e-4, n_grid=info_g['n_grid'], min_len=1e-10,
                    protocol="fixed-w monotone bisection + cap=target refinement")
cert["w"] = [float(x) for x in w0]
cert["comparison"] = dict(area_bound=0.1, oler_claim=0.10397, radial_claim=0.10643)
with open(os.path.join(OUT, "measure_certificate.json"), "w") as f:
    json.dump(cert, f, indent=2)
log("certificate updated: r=%.9f" % r_cert)
log("comparison: area 0.1 | Oler 0.10397 | radial-claim 0.10643 | certified %.6f" % r_cert)
