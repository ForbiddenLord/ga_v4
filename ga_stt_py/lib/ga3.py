"""
ga3.py — Minimal, rigorous 3-D Geometric Algebra (Cl(3,0)) engine.

Built from first principles via basis-blade bitmask multiplication, so the
geometric product / reverse / rotors are correct by construction (derived
from the algebra's axioms e_i e_j = -e_j e_i (i!=j), e_i^2 = +1), not by
hand-transcribing formulas (which is error-prone for sign conventions).

A general multivector in G3 has 8 real components indexed by bitmask:
    0 = 0b000  -> 1            (scalar,    grade 0)
    1 = 0b001  -> e1           (vector,    grade 1)
    2 = 0b010  -> e2           (vector,    grade 1)
    4 = 0b100  -> e3           (vector,    grade 1)
    3 = 0b011  -> e1e2         (bivector,  grade 2)
    5 = 0b101  -> e1e3         (bivector,  grade 2)
    6 = 0b110  -> e2e3         (bivector,  grade 2)
    7 = 0b111  -> e1e2e3 = I   (trivector, grade 3)

Conventions follow the GA-STT letter (Bhat et al.):
  - rotor R in Spin(3): R Rrev = 1
  - v' = R v Rrev                      (eq. 3, sandwich rotation)
  - Rrev = exp(+n_hat theta/2) = cos(theta/2) + n_hat sin(theta/2)
    => R  = exp(-n_hat theta/2) = cos(theta/2) - n_hat sin(theta/2)
  - Rdot = -(1/2) R Omega_b  <=>  Omega_b = -2 Rrev Rdot      (eq. 4, body-frame)
  - bivectors b1 e2e3 + b2 e3e1 + b3 e1e2  are represented externally as a
    plain 3-vector (b1,b2,b3) — this is the standard "axial vector" dual,
    and the commutator product of two such bivectors reduces exactly to the
    ordinary 3-vector cross product.
"""

import numpy as np

N_BASIS = 8  # 2^3

# ---------------------------------------------------------------------
# Build the geometric-product structure constants table once, generically,
# from the bitmask blade-multiplication rule for an orthonormal Euclidean
# basis. This is the standard trick for implementing Clifford algebras:
# blade_a (bitmask a) * blade_b (bitmask b) = sign(a,b) * blade(a xor b)
# where sign(a,b) counts the transpositions needed to interleave-sort the
# index lists of a and b, plus a -1 for every repeated index pair (since
# e_i^2 = +1 contributes no extra sign, e_i anti-commutes with e_j).
# ---------------------------------------------------------------------


def _indices(bitmask):
    return [i for i in range(3) if (bitmask >> i) & 1]


def _blade_mul_sign(a_mask, b_mask):
    """Return sign for the product of basis blades with bitmasks a_mask,b_mask."""
    a_idx = _indices(a_mask)
    b_idx = _indices(b_mask)
    seq = a_idx + b_idx
    sign = 1
    # Bubble-sort the concatenated index sequence; each swap of two
    # *distinct* adjacent indices contributes a factor of -1 (anticommuting
    # orthonormal vectors). Equal adjacent indices annihilate (e_i^2=+1,
    # contributes no sign) and both are removed.
    arr = seq[:]
    i = 0
    while i < len(arr) - 1:
        if arr[i] > arr[i + 1]:
            arr[i], arr[i + 1] = arr[i + 1], arr[i]
            sign *= -1
            if i > 0:
                i -= 1
            continue
        elif arr[i] == arr[i + 1]:
            sign *= 1  # e_i^2 = +1
            del arr[i:i + 2]
            i = max(i - 1, 0)
            continue
        i += 1
    result_mask = 0
    for k in arr:
        result_mask ^= (1 << k)
    return result_mask, sign


# structure[a, b] = (result_blade_index, sign) for basis blade a * basis blade b
_STRUCT_IDX = np.zeros((N_BASIS, N_BASIS), dtype=int)
_STRUCT_SIGN = np.zeros((N_BASIS, N_BASIS), dtype=float)
for _a in range(N_BASIS):
    for _b in range(N_BASIS):
        _res_mask, _sign = _blade_mul_sign(_a, _b)
        _STRUCT_IDX[_a, _b] = _res_mask
        _STRUCT_SIGN[_a, _b] = _sign


def geometric_product(A, B):
    """Full geometric product of two multivectors (each shape (...,8))."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    out = np.zeros(np.broadcast(A[..., 0], B[..., 0]).shape + (8,))
    for a in range(N_BASIS):
        Aa = A[..., a]
        if np.all(Aa == 0):
            continue
        for b in range(N_BASIS):
            Bb = B[..., b]
            s = _STRUCT_SIGN[a, b]
            if s == 0:
                continue
            idx = _STRUCT_IDX[a, b]
            out[..., idx] += s * Aa * Bb
    return out


# Grade of each of the 8 basis-blade slots (0,1,1,2,1,2,2,3)
_GRADE = np.array([bin(i).count("1") for i in range(N_BASIS)])
_REVERSE_SIGN = np.array([1.0 if (g * (g - 1) // 2) %
                         2 == 0 else -1.0 for g in _GRADE])


def reverse(M):
    """Reversion: flips sign of grade-2 and grade-3 parts."""
    M = np.asarray(M, dtype=float)
    return M * _REVERSE_SIGN


def grade(M, k):
    M = np.asarray(M, dtype=float)
    out = np.zeros_like(M)
    mask = (_GRADE == k)
    out[..., mask] = M[..., mask]
    return out


# --- basis elements -----------------------------------------------------
def _basis(mask):
    e = np.zeros(8)
    e[mask] = 1.0
    return e


E1, E2, E3 = _basis(1), _basis(2), _basis(4)
ONE = _basis(0)
E12 = geometric_product(E1, E2)   # bivector slot 3
E13 = geometric_product(E1, E3)   # bivector slot 5 (note: e1e3 = -e3e1)
E23 = geometric_product(E2, E3)   # bivector slot 6
I3 = geometric_product(E12, E3)   # pseudoscalar e1e2e3, slot 7


def vector(v3):
    """Pack a plain 3-vector (...,3) -> multivector (...,8), grade-1 part only."""
    v3 = np.asarray(v3, dtype=float)
    out = np.zeros(v3.shape[:-1] + (8,))
    out[..., 1] = v3[..., 0]
    out[..., 2] = v3[..., 1]
    out[..., 4] = v3[..., 2]
    return out


def to_vector3(M):
    """Extract grade-1 part of a multivector as a plain 3-vector."""
    M = np.asarray(M, dtype=float)
    return np.stack([M[..., 1], M[..., 2], M[..., 4]], axis=-1)


def bivector(b3):
    """
    Pack a plain 3-vector "axis-angle generator" (b1,b2,b3) -> multivector
    representing the bivector (b1*e2e3 + b2*e3e1 + b3*e1e2)  i.e. the dual
    of the vector (b1,b2,b3) via the pseudoscalar I = e1e2e3, with the sign
    fixed so that exp_bivector(axis, theta) sandwich-rotates a vector by
    +theta about +axis under the *right-hand rule*.
    """
    b3 = np.asarray(b3, dtype=float)
    out = np.zeros(b3.shape[:-1] + (8,))
    out[..., 6] += +b3[..., 0]   # +e2e3 coefficient
    out[..., 5] += -b3[..., 1]   # +e3e1 = -e1e3 -> stored at slot5 as -b2
    out[..., 3] += +b3[..., 2]   # +e1e2 coefficient
    return out


def to_bivector3(M):
    """Inverse of bivector(): extract (b1,b2,b3) axial-vector representation."""
    M = np.asarray(M, dtype=float)
    b1 = M[..., 6]
    b2 = -M[..., 5]
    b3 = M[..., 3]
    return np.stack([b1, b2, b3], axis=-1)

def sandwich_derivative(R, omega_b3, v3):
    """
    Exact d/dt[ R v Rrev ] for a FIXED vector v (e.g. e3) given the rotor's
    own current value R and its body angular velocity bivector omega_b3,
    using Rdot = -1/2 R Omega_b (eq.4) and plain product-rule differentiation
    through the existing geometric_product/reverse.
    """
    Rdot = -0.5 * geometric_product(R, bivector(omega_b3))
    Rdot_rev = reverse(Rdot)
    V = vector(v3)
    term1 = geometric_product(geometric_product(Rdot, V), reverse(R))
    term2 = geometric_product(geometric_product(R, V), Rdot_rev)
    return to_vector3(term1 + term2)


def sandwich_bivector(X, biv3):
    """X * bivector(biv3) * Xrev, returned as a 3-vector (axial) representation."""
    B = bivector(biv3)
    Xr = reverse(X)
    out = geometric_product(geometric_product(X, B), Xr)
    return to_bivector3(out)


def rotor(scalar, biv3):
    """Build a rotor multivector from scalar part + bivector-as-3vector part."""
    scalar = np.asarray(scalar, dtype=float)
    out = bivector(biv3)
    out[..., 0] += scalar
    return out


def rotor_scalar(R):
    return np.asarray(R)[..., 0]


def rotor_bivec3(R):
    return to_bivector3(R)


def norm(M):
    """||M|| = sqrt(<M Mrev>_0). For pure rotors this is the unit norm."""
    Mr = reverse(M)
    s = geometric_product(M, Mr)[..., 0]
    return np.sqrt(np.clip(s, 0.0, None))


def sandwich(R, v3):
    """Rotate a plain 3-vector v3 by rotor R: v' = R v Rrev. Returns 3-vector."""
    V = vector(v3)
    Rr = reverse(R)
    out = geometric_product(geometric_product(R, V), Rr)
    return to_vector3(out)


def commutator(A, B):
    """[A,B] = 1/2 (AB - BA), the GA commutator product."""
    return 0.5 * (geometric_product(A, B) - geometric_product(B, A))


def exp_bivector(axis3, theta):
    """
    R = cos(theta/2) + sin(theta/2) * bivector(axis3)
            = cos(theta/2) - sin(theta/2) * n̂_biv   [since bivector() = -n̂_biv]
            = exp(-n̂_biv * theta/2)  matching paper eq.(3)
    Rrev = cos(theta/2) - sin(theta/2) * bivector(axis3)
            = cos(theta/2) + sin(theta/2) * n̂_biv
    """
    axis3 = np.asarray(axis3, dtype=float)
    theta = np.asarray(theta, dtype=float)
    c = np.cos(theta / 2)[..., None] if theta.ndim > 0 else np.cos(theta / 2)
    s = np.sin(theta / 2)[..., None] if theta.ndim > 0 else np.sin(theta / 2)
    return rotor(np.cos(theta / 2), -s * axis3 if theta.ndim > 0 else -np.sin(theta / 2) * axis3)


def rotor_identity(batch_shape=()):
    out = np.zeros(batch_shape + (8,))
    out[..., 0] = 1.0
    return out


def normalize_rotor(R):
    n = norm(R)
    n = np.where(n < 1e-12, 1.0, n)
    return R / n[..., None]


def align_rotor(t_p, t_d):
    """
    Rotor aligning unit vector t_p to unit vector t_d:  Ra t_p Ra~ = t_d.
    Ra = (1 + t_d t_p) / sqrt(2(1+t_d.t_p))
    """
    t_p = np.asarray(t_p, dtype=float)
    t_d = np.asarray(t_d, dtype=float)
    dot = np.sum(t_d * t_p, axis=-1)
    N = ONE + geometric_product(vector(t_d), vector(t_p))
    D = np.sqrt(np.clip(2.0 * (1.0 + dot), 1e-12, None))
    return N / D[..., None]


def rotor_log_angle_axis(Re):
    """
    Given an error rotor Re = cos(theta_e/2) - sin(theta_e/2) n_hat_e
    (bivector form), return (theta_e in [0, 2pi), n_hat_e unit bivector-3vec).
    """
    s0 = rotor_scalar(Re)
    biv = rotor_bivec3(Re)
    biv_norm = np.linalg.norm(biv, axis=-1)
    s0c = np.clip(s0, -1.0, 1.0)
    # in [0, pi], since sin(half_theta) = biv_norm >= 0
    half_theta = np.arccos(s0c)
    theta_e = 2.0 * half_theta
    eps = 1e-9
    n_hat_e = -biv / (biv_norm[..., None] + eps)
    return theta_e, n_hat_e


__all__ = [
    "geometric_product", "reverse", "grade", "vector", "to_vector3",
    "bivector", "to_bivector3", "rotor", "rotor_scalar", "rotor_bivec3",
    "norm", "sandwich", "commutator", "exp_bivector", "rotor_identity",
    "normalize_rotor", "align_rotor", "rotor_log_angle_axis",
    "E1", "E2", "E3", "ONE", "I3",
]
