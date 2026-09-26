"""
lower_bounds.py — Rigorous LOWER bounds for r_D(100), in the exact format
demanded by requirement 1 (integer polynomial P(t), rational isolation interval
(a,b), unique root alpha in (a,b)), plus the proof that these alphas are true
lower bounds (requirement 4, for those alphas).

===========================================================================
THEOREM 1 (area bound).  For every configuration C in (R^2)^100,
      R_D(C) >= 1/10.
Proof. If 100 disks of radius R cover the unit disk D then, by monotonicity of
Lebesgue area,
    pi = area(D) <= sum_{i=1}^{100} area(B(c_i, R) ∩ D)
                 <= sum_{i=1}^{100} area(B(c_i, R)) = 100 pi R^2,
hence R^2 >= 1/100, i.e. R >= 1/10.                                    QED

Requirement-1 format for this alpha:
    P(t) = 100 t^2 - 1          (coefficients: 100, 0, -1)
    a = 0, b = 1/2.
    P(0) = -1 < 0 < 24 = P(1/2),  so P has a root in (0, 1/2) (IVT);
    P'(t) = 200 t > 0 on (0, 1/2], so P is strictly increasing there and the
    root is unique:  alpha = 1/10.
Requirement-4 for this alpha is exactly Theorem 1:  for ALL C, R_D(C) >= alpha.
===========================================================================

REMARK (what is provable and what is not).  Theorem 1 is sharp enough to be
"the" exact algebraic lower bound available by elementary means.  Any stronger
lower bound alpha > 1/10 (e.g. the claimed 0.10397 "Oler" value or the claimed
0.10643 "radial measure" value) requires a genuine covering-duality argument
and an audit; see REPORT.md.  This script prints the status of both routes.
"""

from fractions import Fraction


def requirement1_certificate():
    """Return the requirement-1 data for the area bound."""
    coeff = [100, 0, -1]                    # P(t) = 100 t^2 + 0 t - 1
    a, b = Fraction(0), Fraction(1, 2)
    # exact root isolation by sign + monotonicity, all in exact rationals
    P = lambda t: 100 * t * t - 1
    assert P(a) < 0 < P(b)
    # uniqueness: P strictly increasing on [0, 1/2] since P'(t) = 200 t > 0
    alpha = Fraction(1, 10)
    assert P(alpha) == 0 and a < alpha < b
    return coeff, a, b, alpha


def theorem1_statement():
    return (
        "Theorem 1 (area bound). For every C in (R^2)^100:  R_D(C) >= 1/10.\n"
        "Proof. pi = area(D) <= sum_i area(B(c_i,R) ∩ D) <= 100 pi R^2.\n"
        "Hence R >= 1/10.  QED"
    )


def boundary_arc_bound():
    """Weaker but also exact-algebraic bound: r >= sin(pi/100). Reported for
    completeness; NOT stronger than 1/10, so Theorem 1 dominates."""
    import math
    return math.sin(math.pi / 100)


def stronger_bounds_status():
    return """
STATUS OF STRONGER LOWER BOUNDS (requirement 4 at alpha > 1/10)
---------------------------------------------------------------------
Route                       | Claimed | Status this audit
----------------------------|---------|----------------------------------
Oler-type inequality        | 0.10397 | NOT ESTABLISHED here.  Oler's
                            |         | inequality is an UPPER bound on the
                            |         | size of a 2r-separated set; to kill a
                            |         | covering of radius r one must EXHIBIT
                            |         | 101 points pairwise > 2r in D.  The
                            |         | best construction route (hexagonal
                            |         | lattice in the disk) reaches only
                            |         | ~93-101 points at r ~ 0.095-0.099;
                            |         | a proven 101-point packing at
                            |         | 2r = 0.2079 (which is what 0.10397
                            |         | needs) is a hard packing instance.
Radial measure duality      | 0.10643 | NOT ESTABLISHED here.  The method is
                            |         | sound in principle (find mu with
                            |         | mu(ball(c,r)) <= 1/100 - delta for all
                            |         | centers c, then n balls of radius r
                            |         | miss mass n*delta > 0), but the
                            |         | claimed instance must be audited:
                            |         | the sup over centers must be a true
                            |         | sup (a grid max is a LOWER estimate
                            |         | and yields a FALSE bound - the exact
                            |         | failure mode already caught once in
                            |         | the project log).
---------------------------------------------------------------------
Conclusion: the only alpha satisfying requirements 1+4 SIMULTANEOUSLY and
provably today is alpha = 1/10 (Theorem 1).  Everything above it is open.
"""


if __name__ == "__main__":
    coeff, a, b, alpha = requirement1_certificate()
    print("Requirement-1 certificate (area bound):")
    print(f"  P(t) coefficients (descending): {coeff}")
    print(f"  isolation interval: a = {a}, b = {b}")
    print(f"  unique root alpha = {alpha} = {float(alpha)}")
    print()
    print(theorem1_statement())
    print()
    print(f"boundary-arc bound (dominated): sin(pi/100) = {boundary_arc_bound():.6f}")
    print(stronger_bounds_status())
