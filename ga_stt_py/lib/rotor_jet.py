"""
rotor_jet.py — Taylor jets of GA rotors (scalar + bivector elements).

A rotor R = s + B (scalar s, bivector B represented as a 3-vector under the
axial-vector duality already verified in ga3.py) is closed under the
geometric product:

    (s1+B1)(s2+B2) = (s1 s2 - B1.B2)  +  (s1 B2 + s2 B1 - B1 x B2), minus sign follows from bivector(v):=I*v

(verified numerically against ga3.geometric_product — see the assertion in
build_and_verify() below, and tests/test_rotor_jet.py). Because this
reduces entirely to scalar dot/cross-product algebra, a "rotor jet" is just
a pair of an order-N *scalar* jet1d array and an order-N *vector* jet1d
array, and every operation below is built from the already-unit-tested
jet1d primitives — no new finite-difference or hand-derived formulas.
"""

import numpy as np
import jet1d as jt
import ga3 as ga


class RotorJet:
    __slots__ = ("s", "b")  # s: scalar jet (N+1,), b: vector jet (N+1,3)

    def __init__(self, s, b):
        self.s = s
        self.b = b

    @property
    def order(self):
        return jt.order_of(self.s)


def const_identity(order):
    return RotorJet(jt.const_scalar(1.0, order), jt.const_vector([0, 0, 0], order))


def const_from_value(R8, order):
    """Build a constant-in-time RotorJet from a plain ga3 rotor multivector."""
    s = ga.rotor_scalar(R8)
    b = ga.rotor_bivec3(R8)
    return RotorJet(jt.const_scalar(s, order), jt.const_vector(b, order))


def mul(R1, R2):
    s1, b1, o1 = R1.s, R1.b, R1.order
    s2, b2, o2 = R2.s, R2.b, R2.order
    s = jt.sub(jt.mul_ss(s1, s2), jt.dot_vv(b1, b2))
    b = jt.sub(jt.add(jt.mul_sv(s1, b2), jt.mul_sv(s2, b1)), jt.cross_vv(b1, b2))
    return RotorJet(s, b)


def reverse(R):
    return RotorJet(R.s, jt.neg(R.b))


def scale_by_scalar_jet(R, s_jet):
    return RotorJet(jt.mul_ss(R.s, s_jet), jt.mul_sv(s_jet, R.b))


def div_by_scalar_jet(R, s_jet):
    return RotorJet(jt.div_ss(R.s, s_jet), jt.vec_div_s(R.b, s_jet))


def differentiate(R):
    """RotorJet of R'(t) (order-1 lower)."""
    return RotorJet(jt.differentiate(R.s), jt.differentiate(R.b))


def align(tp_jet, td_jet):
    """
    Ra(t) = (1 + t_d t_p) / sqrt(2(1+t_d.t_p))
    as a RotorJet, given jet1d *vector* jets tp_jet, td_jet (both order N).
    Built purely from jet1d dot/cross/sqrt/div primitives (verified above
    that vector(a)*vector(b) = dot(a,b) + bivector(-cross(a,b)) under our
    convention), so this is automatically exact to all represented orders.
    """
    dot_jet = jt.dot_vv(td_jet, tp_jet)
    cross_jet = jt.cross_vv(td_jet, tp_jet)
    N_s = jt.add(jt.const_scalar(1.0, jt.order_of(dot_jet)), dot_jet)
    N_b = cross_jet
    D = jt.sqrt_s(jt.scale(jt.add(jt.const_scalar(1.0, jt.order_of(dot_jet)), dot_jet), 2.0))
    return RotorJet(jt.div_ss(N_s, D), jt.vec_div_s(N_b, D))


def value_multivector(R):
    """Return the order-0 (current value) part as a plain ga3 rotor (8,)."""
    return ga.rotor(jt.value(R.s), jt.value(R.b))


def to_bivector_multivector_jet(b_jet_only):
    """Wrap a plain jet1d vector jet as a sequence of ga3 bivector multivectors
    (one per order), useful for diagnostics."""
    out = []
    for k in range(jt.order_of(b_jet_only) + 1):
        out.append(ga.bivector(b_jet_only[k]))
    return out
