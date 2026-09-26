"""
verify_cover.py — Rigorous, machine-checkable covering certificate (pure integer
arithmetic, no floating point in the decision logic, no scipy dependency).

CLAIM BEING CERTIFIED
    Let q_1, ..., q_100 be the rational centers (integers a_i, b_i divided by
    S = 2^30, read from centers_best.json) and let r = r_int / S (rational).
    Then every point x of the closed unit disk D satisfies
        min_i || x - q_i ||  <=  r.

METHOD (branch-and-bound with exact integer interval arithmetic)
    Subdivide the bounding square [-S, S]^2 by a quadtree.  A box
    B = [x0,x1] x [y0,y1] (integer endpoints) is dismissed iff
      (1) B lies entirely outside D:  the point of B nearest the origin has
          squared norm > S^2  (exact integer test), or
      (2) some center (a_i, b_i) satisfies
            max_{(x,y) in B} (x-a_i)^2 + (y-b_i)^2
              = dx^2 + dy^2,   dx = max(|x0-a_i|, |x1-a_i|),
                               dy = max(|y0-b_i|, |y1-b_i|)
          (exact integer identity: the max of the convex quadratic over the box
          is at a corner)  <=  r_int^2.
    Otherwise B is subdivided while its side exceeds MIN_SIZE, else it is
    reported as a failure.

SOUNDNESS: every acceptance is an exact integer inequality covering the whole
box, and only boxes outside D are dismissed, so acceptance of all boxes proves
the claim for ALL points of D.

TERMINATION (completeness): let R* = max_{x in D} min_i ||x - q_i|| and let
r - R* =: mu > 0.  For any box B of diameter delta and any x* in B, the nearest
center c of x* satisfies  max_{x in B} ||x - c||  <=  ||x* - c|| + delta
<= R* + delta  (1-Lipschitz property of x -> min_i ||x - c_i||).  Hence every
box with delta <= mu passes test (2); boxes of side MIN_SIZE have diameter
MIN_SIZE*sqrt(2) < mu*S, so the search terminates.

Run:  python verify_cover.py [input.json]
"""

import json
import sys
import math
import time

S = 1 << 30                     # global scale: coordinates are integers / S
MIN_SIZE = S >> 18              # smallest box side (diameter ~5.4e-6 < margin)
MARGIN = 1.0e-5                 # rational slack added to the floating radius


def verify(path="centers_best.json", verbose=True):
    data = json.load(open(path))
    centers = data["centers"]
    n = len(centers)
    R_float = float(data["radius"])

    # rationalize: round centers to the 1/S grid; inflate radius accordingly
    C = [(int(round(a * S)), int(round(b * S))) for a, b in centers]
    r_int = int(math.ceil((R_float + MARGIN) * S))

    stack = [(-S, -S, S, S)]
    covered = 0
    dismissed = 0
    deepest = 0
    failures = []
    t0 = time.time()

    while stack:
        x0, y0, x1, y1 = stack.pop()
        # (1) box entirely outside the closed disk?
        nx = min(max(x0, 0), x1)
        ny = min(max(y0, 0), y1)
        if nx * nx + ny * ny > S * S:
            dismissed += 1
            continue
        # (2) some center covers the whole box?  test the 6 centers whose
        # distance to the box center is smallest (any covering center works;
        # testing the nearest ones is only for speed)
        cxm, cym = (x0 + x1) // 2, (y0 + y1) // 2
        order = sorted(range(n),
                       key=lambda i: (C[i][0] - cxm) ** 2 + (C[i][1] - cym) ** 2)
        ok = False
        for i in order[:6]:
            a, b = C[i]
            # max over the box of (x-a)^2 is attained at a corner:
            dx = max(abs(x0 - a), abs(x1 - a))
            dy = max(abs(y0 - b), abs(y1 - b))
            if dx * dx + dy * dy <= r_int * r_int:
                ok = True
                break
        if ok:
            covered += 1
            continue
        side = x1 - x0
        deepest = max(deepest, side)
        if side <= MIN_SIZE:
            failures.append((x0, y0, x1, y1))
            if len(failures) >= 20:
                break
            continue
        xm, ym = (x0 + x1) // 2, (y0 + y1) // 2
        stack.append((x0, y0, xm, ym))
        stack.append((xm, y0, x1, ym))
        stack.append((x0, ym, xm, y1))
        stack.append((xm, ym, x1, y1))
        if verbose and (covered + dismissed) > 0 and (covered + dismissed) % 200000 == 0:
            print(f"  ... {covered+dismissed} boxes resolved, "
                  f"{len(stack)} open, {time.time()-t0:.0f}s", flush=True)

    cert = dict(
        n=n, S=S, r_int=r_int, r_float=r_int / S,
        centers_int=C,
        boxes_covered=covered, boxes_dismissed_outside_disk=dismissed,
        failures=[[x0 / S, y0 / S, x1 / S, y1 / S] for x0, y0, x1, y1 in failures],
        seconds=round(time.time() - t0, 1),
        claim=("every x with x^2+y^2 <= 1 satisfies min_i ||x - q_i|| <= r_int/S,"
               " q_i = centers_int/S"),
    )
    verdict = (len(failures) == 0)
    cert["verified"] = verdict
    with open("certificate.json", "w") as f:
        json.dump(cert, f, indent=1)

    print(f"\ncenters rationalized to the 1/2^30 grid")
    print(f"certified radius r = {r_int}/{S} = {r_int/S:.10f}  "
          f"(floating construction radius {R_float:.10f}, margin {MARGIN})")
    print(f"boxes covered: {covered}, dismissed (outside disk): {dismissed}")
    print(f"RESULT: {'VERIFIED' if verdict else 'FAILED'} "
          f"in {cert['seconds']}s")
    if failures:
        print("failure boxes (float coords):")
        for x0, y0, x1, y1 in failures[:10]:
            print(f"  [{x0/S:+.6f},{x1/S:+.6f}] x [{y0/S:+.6f},{y1/S:+.6f}]")
    return verdict


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "centers_best.json"
    ok = verify(path)
    sys.exit(0 if ok else 1)
