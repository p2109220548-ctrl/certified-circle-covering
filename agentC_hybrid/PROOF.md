# A Certified Area + Boundary-Arc Hybrid Lower Bound for r_D(100)

**Agent C (lb-hybrid).** All computation in `hybrid_bound.py` (primary) and
`concavity_check.py` (verification suite); machine-readable results in
`hybrid_certificate.json`.

## Theorem

> Let D be the closed unit disk. If 100 closed disks of radius r cover D,
> then necessarily
>
> **T(r) >= pi**, where
> T(r) = min over integers m in [ceil(pi/arcsin r), 100] of
>        (100 - m) * pi * r^2  +  m * g(2*pi/m),
>
> and g is the "arc-for-area frontier" defined below. Consequently
> **T(r) < pi  implies  r_D(100) > r.**
>
> Numerically, T(r) < pi exactly for r in [0.1, 0.1060719656...), so
> **r_D(100) > 0.106071** (reported threshold rounded DOWN; T(0.106071) =
> 3.1415012703 = pi - 9.14e-5, with the minimizer m = 30).

## Setup and exact formulas

Suppose 100 closed disks B_i = B(c_i, r) cover D. Classify each by
d_i = ||c_i||:

* **INTERIOR**: d_i + r <= 1. The disk lies in D, so it meets D in exactly
  area pi*r^2 and covers **zero length** of the boundary circle dD.
* **CROSSER**: 1 - r < d_i < 1 + r. It meets dD in an arc and covers a lens
  of D.
* **MIDAIR**: d_i >= 1 + r. Contributes nothing (area 0, arc 0).

For a crosser at distance d write

* **arc**:  b(d) = 2*arccos( (d^2 + 1 - r^2) / (2d) ).
  (Law of cosines at the origin; the arc of dD inside the disk has half-angle
  arccos((d^2+1-r^2)/(2d)).)  By AM-GM, (d^2 + 1 - r^2)/(2d) = (d + (1-r^2)/d)/2
  is minimized at d* = sqrt(1-r^2), where it equals sqrt(1-r^2) = cos(arcsin r).
  Hence **b(d) <= 2*arcsin(r)** for all d, with equality exactly at d*.  Also
  b is strictly increasing on the *first branch* d in [1-r, d*] (from 0 at the
  internal tangency d = 1-r up to the maximum 2 arcsin r at d*).

* **lens area**:  a(d) = r^2*arccos((d^2+r^2-1)/(2dr)) + arccos((d^2+1-r^2)/(2d))
  - 0.5*sqrt((-d+r+1)(d+r-1)(d-r+1)(d+r+1)).
  This is the standard two-circle intersection area (unit circle vs disk of
  radius r at distance d); verified independently by 1-D slab integration to
  2e-13 (see `concavity_check.py`, V1).  Boundary sanity: a(1-r) = pi*r^2
  (internal tangency, exact to 1e-17 at r = 0.1075) and a(d) -> 0 at d = 1+r.
  a is **strictly decreasing** in d on [1-r, 1+r] (verified numerically on a
  200001-point grid; max upward difference -6.6e-10, i.e. all differences < 0).

## Lemma 1 (crossers must pay arc)

The boundary circle dD (length 2*pi) is covered, and only crossers cover
positive boundary length, each covering at most b(d_i) <= 2 arcsin r. Hence,
with m = number of crossers,

    sum_{i=1}^{m} b(d_i) >= 2*pi    and    m >= pi / arcsin(r).

## Lemma 2 (frontier g, first-branch optimality)

For a crosser, its arc burden b = b(d_i) determines at most the area
g(b) := max{ a(d) : b(d) = b }.  The equation b(d) = b has (for
0 < b < 2 arcsin r) two roots d1 in [1-r, d*) and d2 in (d*, 1+r); since a is
strictly decreasing, the maximum is a(d1) on the **first branch**.  Thus

    g(b) = a( d1(b) ),  d1(b) = unique root of b(d) = b in [1-r, d*],

computed to ~1e-15 by 100-step bisection.  g is continuous, strictly
decreasing (verified), g(0) = pi*r^2, g(2 arcsin r) = a(d*).

Every crosser covers area a(d_i) <= g(b(d_i)) <= g(b_i) for any assigned
b_i = b(d_i), and every interior disk covers exactly pi*r^2. Since overlaps
only shrink the union, the total covered area of D is at most

    (100 - m)*pi*r^2  +  sum_{i=1}^{m} g(b_i),   subject to sum b_i >= 2*pi,
    b_i in [0, 2 arcsin r].

Because g is decreasing, the allocator's optimum uses sum b_i = 2*pi exactly
(any slack can be redistributed to reduce some b_i, never decreasing the
achievable sum... more precisely, decreasing any b_i can only increase g(b_i)).

## Lemma 3 (concavity of g; the envelope step)

**Claim.** g is strictly concave on [0, 2 arcsin r].  Then the least concave
majorant of g is g itself, and by Jensen's inequality, for every feasible
allocation,

    sum_i g(b_i) <= sum_i gbar(b_i) <= m * gbar( (sum b_i)/m ) = m * g(2*pi/m),

where gbar is any concave majorant of g (gbar >= g pointwise and gbar
concave).  This is the only relaxation in the whole argument, and under
concavity it is *exact* (no sampling error enters the envelope step).

**Verification of the claim (three regimes).**

1. *Near b = 0 (analytic).*  Offsetting an internally tangent disk outward by
   eps (d = 1 - r + eps), the part of the disk leaving D is, to leading order,
   the "poke" region beyond dD of radial extent eps - (1-r)y^2/(2r) at
   transverse distance y, supported on |y| < sqrt(2*eps*r/(1-r)).  Its area is

       pi*r^2 - a(d) = (4/3) * sqrt(2r/(1-r)) * eps^{3/2} + O(eps^{5/2}).

   Inverting b(d) = 2*sqrt(2*eps*r/(1-r)) + O(eps^{3/2}) gives

       g(b) = pi*r^2 - c3 * b^3 + O(b^5),   c3 = (1-r)/(12r) > 0,

   so g''(b) = -6*c3*b + O(b^3) < 0 for small b > 0.  Numerically
   L(b)/b^3 -> 0.692061 at b = 1e-3 vs the theoretical c3 = 0.691860
   (agreement 0.03%); see `concavity_check.py` V2(i).

2. *Bulk (numeric, noise-controlled).*  Direct central second differences of
   the exact g(b) (bisection inversion, 100 steps) with adaptive step
   hb = min(1e-4, b/5, (bmax-b)/5) at 3000 log-spaced points in
   [1e-3, bmax*(1-5e-6)]: **all 3000 values strictly negative**; the largest
   is -4.13e-3 (at b = 1e-3) while the propagated floating-point noise bound
   there is ~2.4e-4 (signal-to-noise > 15).  The dominant fp hazard - arccos
   of an argument 1 - O(alpha^2) near tangency, with absolute error
   ~1e-16/sqrt(2 delta) - is confined to b <~ 1e-3, which is covered
   analytically by regime 1.  Near bmax the second differences are ~-17
   (curvature singularity below), far above the ~1e-5 noise there.

3. *Near b = 2 arcsin r (analytic).*  At d* = sqrt(1-r^2) the arc function has
   a quadratic maximum: b(d) = bmax - (|b''(d*)|/2)(d-d*)^2 + ..., with
   b''(d*) = -2(1-r^2)/(d*^3 r) < 0, while a'(d*) < 0 is finite.  Inverting,

       g(b) = a(d*) + k*sqrt(bmax - b) + O((bmax-b)^{3/2}),
       k = |a'(d*)| / sqrt(|b''(d*)|/2) > 0,

   and sqrt is concave: g''(b) = -k/(4 (bmax-b)^{3/2}) < 0, -> -infinity at
   bmax.  Numerically k = 0.070276 (stable to 7 digits across
   bmax-b = 1e-4 .. 1e-6).

**Remark (why a plain fine grid is not enough).** On a uniform 200001-point
grid the second differences show spurious positive values up to +8.3e-13 near
b = 1e-3 (arccos noise) and the naive "all second differences <= 0" test
fails.  The three-regime protocol above is what makes the concavity claim
trustworthy.  For the same reason the concave-hull-of-samples construction is
used only as a *cross-check* (it agrees with the exact g to 4.2e-9 at the
threshold; its chord interpolation necessarily *under*-estimates a concave g
between samples, so it can only weaken, never invalidate, the bound).

## Theorem (proof)

Take any covering of D by 100 disks of radius r, let m be the number of
crossers and b_1..b_m their arcs. By Lemma 1, m >= ceil(pi/arcsin r) and
sum b_i >= 2*pi; by Lemma 2 the covered area is at most
(100-m)*pi*r^2 + sum g(b_i); by Lemma 3 (Jensen on the concave majorant gbar = g),

    covered area <= (100-m)*pi*r^2 + m*g(2*pi/m).

Taking the minimum over all feasible m (the actual m is unknown to us, and
every bound in the chain holds for *each* m), the covered area is at most
T(r) as defined.  Covering D requires area >= pi, so T(r) < pi makes a
covering of radius r impossible.

**Strictness / the reported number.** T(r) is continuous in r (a finite
minimum of functions continuous in r; at a bracket boundary m -> m+1 the new
term enters with b0 = 2 arcsin r exactly, joining continuously).  Since
T(0.106071) = pi - 9.14e-5 < pi, continuity yields a delta > 0 with T < pi on
[r_report, r_report + delta]; all those radii are impossible, hence
**r_D(100) > 0.106071**.  For r <= 0.1 the classical area bound (T(r) <=
100*pi*r^2 <= pi, since g <= pi*r^2) already gives impossibility for r < 0.1,
and a 304-point fine grid (step 2e-5) verifies T(r) < pi on all of
[0.1, 0.106071] (max 3.1404595 at r = 0.10606).

**Numerical hygiene.** (i) g is evaluated by 100-step bisection on b(d) = b
(conditional width ~1e-24, far below double precision; g accurate to ~1e-15
away from the tangency endpoints, none of which are evaluation points since
2*pi/m in [2*pi/100, 2*pi/30] is bounded away from 0 and bmax).  (ii) The
final comparison uses margin 9.1e-5, six orders of magnitude above any
floating-point or sampling slack in the chain.  (iii) The reported threshold
is rounded DOWN to 0.106071.

## The T(r) curve and the minimizer

T is a sawtooth: on each m-bracket r in [sin(pi/(m+1)), sin(pi/m)] the
minimizer is m_min = ceil(pi/arcsin r) (for convex loss L = pi*r^2 - g with
L(0) = 0, m*L(2*pi/m) decreases in m, so T increases in m at fixed r), and T
drops whenever the bracket index falls.  The crossing of pi happens inside the
m = 30 bracket:

| r | T(r) | argmin m | T - pi |
|-------|-----------|----|---------|
| 0.1000 | 2.7850674473 | 32 | -3.565e-01 |
| 0.1010 | 2.8680804548 | 32 | -2.735e-01 |
| 0.1020 | 2.8653408705 | 31 | -2.763e-01 |
| 0.1030 | 2.9605444035 | 31 | -1.810e-01 |
| 0.1040 | 3.0467459801 | 31 | -9.485e-02 |
| 0.1050 | 3.0295096266 | 30 | -1.121e-01 |
| 0.1060 | 3.1347534966 | 30 | -6.839e-03 |
| **0.106071** | **3.1415012703** | **30** | **-9.138e-05** |
| 0.1065 | 3.1812368287 | 30 | +3.964e-02 |
| 0.1075 | 3.2692616234 | 30 | +1.277e-01 |
| 0.1080 | 3.3117571704 | 30 | +1.702e-01 |
| 0.1085 | 3.2437220712 | 29 | +1.021e-01 |
| 0.1100 | 3.4031508110 | 29 | +2.616e-01 |
| 0.1120 | 3.4330887411 | 28 | +2.915e-01 |

Every later bracket re-enters above pi (left-endpoint values: 3.2437 at m=29,
3.4331 at m=28, increasing), so the first crossing is the global one.

At the threshold the bound has the concrete form: 30 crossers must each carry
arc 2*pi/30 = 0.209440 (98.5% of the per-disk maximum 2 arcsin(0.106071) =
0.212542), forcing each crosser's center near d* = 0.994359 and limiting its
inside-area to g(0.209440) = 0.0222422 (loss of 0.0131041 = 37.1% of its disk
area pi*r^2 = 0.0353462), while the other 70 disks contribute at most
pi*r^2 each:

    T = 70*pi*r^2 + 30*g(2*pi/30) = pi - 9.14e-5  at  r = 0.106071.

## Certification statement

    r_D(100) > 0.106071

is certified, conditional on the numerical concavity verification of Lemma 3
(all analytic anchors verified; bulk verified with signal-to-noise > 15 on
3000 adaptive-second-difference samples).  Every other step of the proof is
exact algebra/geometry or carries margin >= 9e-5 against ~1e-11-sized
numerical slack.  The value corrects the team's preliminary hand estimate of
0.1075-0.108 (at r = 0.1075 the bound reads T = pi + 0.128, so 0.1075 is NOT
excluded by this method).

## Files

* `hybrid_bound.py` - primary computation (table, bisection, certificate).
* `concavity_check.py` - independent a(d) slab-integration check + three-regime concavity verification.
* `hybrid_certificate.json` - machine-readable certificate.
* `final_run_log.txt`, `concavity_report.json` - raw outputs.
