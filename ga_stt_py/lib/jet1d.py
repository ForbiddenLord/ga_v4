"""
jet1d.py — order-N Taylor-jet ("automatic differentiation to arbitrary
order") arithmetic for scalar and R^3-vector-valued functions of a single
real parameter (time).

A jet of order N representing f(t0+tau) is stored as its Taylor
coefficients  c_k = f^(k)(t0) / k!   for k = 0..N, so that
    f(t0+tau) = sum_k c_k * tau^k + O(tau^{N+1}).

This is *exact* differentiation (up to floating-point round-off), not a
finite-difference approximation — multiplication uses the Cauchy-product
convolution, and sqrt/exp/log use the standard power-series composition
recurrences (Brent 1976 / standard AD literature). All recurrences below
are verified against known closed-form derivatives in
tests/test_jet1d.py.

Shapes:
  scalar jet: ndarray, shape (N+1,)
  vector jet: ndarray, shape (N+1, 3)
"""

import numpy as np


def const_scalar(value, order):
    c = np.zeros(order + 1)
    c[0] = value
    return c


def const_vector(value3, order):
    c = np.zeros((order + 1, 3))
    c[0] = np.asarray(value3, dtype=float)
    return c


def time_jet(t0, order):
    """Jet of the identity function f(t)=t about t0: c0=t0, c1=1, rest 0."""
    c = np.zeros(order + 1)
    c[0] = t0
    if order >= 1:
        c[1] = 1.0
    return c


def linear_kinematics_jet(pos0, vel0, order):
    """Jet of a constant-velocity point o(t) = pos0 + vel0*t (exact to all orders)."""
    c = np.zeros((order + 1, 3))
    c[0] = pos0
    if order >= 1:
        c[1] = vel0
    return c


def order_of(c):
    return c.shape[0] - 1


def _match_order(a, b):
    oa, ob = order_of(a), order_of(b)
    if oa == ob:
        return a, b, oa
    o = min(oa, ob)
    return a[:o + 1], b[:o + 1], o


def add(a, b):
    a, b, o = _match_order(a, b)
    return a + b


def sub(a, b):
    a, b, o = _match_order(a, b)
    return a - b


def neg(a):
    return -a


def scale(a, s):
    """Multiply a jet by a plain python/np scalar (not itself a jet)."""
    return a * s


def mul_ss(a, b):
    """Cauchy product of two SCALAR jets -> scalar jet."""
    a, b, o = _match_order(a, b)
    out = np.zeros(o + 1)
    for k in range(o + 1):
        out[k] = np.dot(a[:k + 1], b[k::-1])
    return out


def mul_sv(s, v):
    """Scalar jet (N+1,) times vector jet (N+1,3) -> vector jet (N+1,3)."""
    s, v, o = _match_order(s, v)
    out = np.zeros((o + 1, 3))
    for k in range(o + 1):
        # sum_{i=0}^k s_i * v_{k-i}
        out[k] = np.tensordot(s[:k + 1], v[k::-1], axes=(0, 0))
    return out


def div_ss(a, b):
    """Scalar jet division a/b via the standard recurrence h=a/b => a=h*b."""
    a, b, o = _match_order(a, b)
    h = np.zeros(o + 1)
    h[0] = a[0] / b[0]
    for k in range(1, o + 1):
        s = a[k] - np.dot(h[:k], b[k:0:-1])
        h[k] = s / b[0]
    return h


def recip(b):
    one = const_scalar(1.0, order_of(b))
    return div_ss(one, b)


def sqrt_s(a):
    """Scalar jet sqrt via h*h=a recurrence."""
    o = order_of(a)
    h = np.zeros(o + 1)
    h[0] = np.sqrt(max(a[0], 0.0))
    for k in range(1, o + 1):
        if k == 1:
            inner = 0.0
        else:
            inner = np.dot(h[1:k], h[k - 1:0:-1])
        h[k] = (a[k] - inner) / (2.0 * h[0])
    return h


def exp_s(a):
    o = order_of(a)
    h = np.zeros(o + 1)
    h[0] = np.exp(a[0])
    for k in range(1, o + 1):
        idx = np.arange(1, k + 1)
        h[k] = np.dot(idx * a[1:k + 1], h[k - 1::-1]) / k
    return h


def log_s(a):
    o = order_of(a)
    h = np.zeros(o + 1)
    h[0] = np.log(a[0])
    for k in range(1, o + 1):
        idx = np.arange(1, k)
        inner = np.dot(idx * h[1:k], a[k - 1:0:-1]) if k > 1 else 0.0
        h[k] = (a[k] - inner / k) / a[0]
    return h


def dot_vv(u, v):
    """Dot product of two vector jets -> scalar jet."""
    u, v, o = _match_order(u, v)
    out = np.zeros(o + 1)
    for k in range(o + 1):
        terms = u[:k + 1] * v[k::-1]
        out[k] = np.sum(terms)
    return out


def cross_vv(u, v):
    """Cross product of two vector jets -> vector jet (Cauchy product per axis)."""
    u, v, o = _match_order(u, v)
    out = np.zeros((o + 1, 3))
    for k in range(o + 1):
        acc = np.zeros(3)
        for i in range(k + 1):
            acc += np.cross(u[i], v[k - i])
        out[k] = acc
    return out


def norm_v(v):
    return sqrt_s(dot_vv(v, v))


def vec_div_s(v, s):
    """Vector jet divided by scalar jet, elementwise via div_ss per axis."""
    v, s, o = _match_order(v, s)
    out = np.zeros((o + 1, 3))
    for ax in range(3):
        out[:, ax] = div_ss(v[:, ax], s)
    return out


def coeff_to_derivative(c, k):
    """k-th true derivative f^(k)(t0) = k! * c_k."""
    from math import factorial
    return factorial(k) * c[k]


def differentiate(c):
    """
    Given the jet of f(t) (order N), return the jet of f'(t) (order N-1).
    If f = sum c_k tau^k, then f' = sum_{k>=0} (k+1) c_{k+1} tau^k.
    Works for both scalar jets (N+1,) and vector jets (N+1,3).
    """
    o = order_of(c)
    if o < 1:
        raise ValueError("Cannot differentiate a 0-order jet (no info left).")
    idx = np.arange(1, o + 1)
    if c.ndim == 1:
        return idx * c[1:]
    else:
        return (idx[:, None]) * c[1:]


def value(c):
    return c[0]
