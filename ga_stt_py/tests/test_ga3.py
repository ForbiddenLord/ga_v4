"""Unit tests for ga3.py — run before trusting anything built on top of it."""
import numpy as np
import sys, os
from scipy.spatial.transform import Rotation as scipyR
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import ga3 as ga

rng = np.random.default_rng(0)
TOL = 1e-9
nfail = 0


def check(name, cond):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name}")


# 1. Basis bivector algebra: e23^2 = -1, e31^2=-1, e12^2=-1 (each squares to -1)
for name, e in [("e2e3", ga.E23), ("e3e1", -ga.E13), ("e1e2", ga.E12)]:
    sq = ga.geometric_product(e, e)
    check(f"{name}^2 == -1", np.allclose(sq, -ga.ONE, atol=TOL))

# 2. Pseudoscalar squares to -1 (paper: I^2 = -1)
check("I^2 == -1", np.allclose(ga.geometric_product(ga.I3, ga.I3), -ga.ONE, atol=TOL))

# 3. e1 e2 e3 anticommute pairwise, square to +1
check("e1^2==1", np.allclose(ga.geometric_product(ga.E1, ga.E1), ga.ONE, atol=TOL))
check("e1e2 == -e2e1", np.allclose(ga.geometric_product(ga.E1, ga.E2),
                                   -ga.geometric_product(ga.E2, ga.E1), atol=TOL))

# 4. Associativity of geometric product on random multivectors
A = rng.normal(size=8)
B = rng.normal(size=8)
C = rng.normal(size=8)
lhs = ga.geometric_product(ga.geometric_product(A, B), C)
rhs = ga.geometric_product(A, ga.geometric_product(B, C))
check("associativity (AB)C == A(BC)", np.allclose(lhs, rhs, atol=TOL))

# 5. vector <-> bivector pack/unpack round-trip
v = rng.normal(size=3)
check("vector roundtrip", np.allclose(ga.to_vector3(ga.vector(v)), v, atol=TOL))
b = rng.normal(size=3)
check("bivector roundtrip", np.allclose(
    ga.to_bivector3(ga.bivector(b)), b, atol=TOL))

# 6. Bivector commutator == ordinary 3D cross product (key GA<->vector-calculus link)
b1 = rng.normal(size=3)
b2 = rng.normal(size=3)
comm = ga.to_bivector3(ga.commutator(ga.bivector(b1), ga.bivector(b2)))
cross = -np.cross(b1, b2)
check("bivector commutator == cross product",
      np.allclose(comm, cross, atol=TOL))

# 7. Rotor unit norm: R Rrev = 1 for exp_bivector-constructed rotors
axis = np.array([0.2, -0.5, 0.7])
axis = axis/np.linalg.norm(axis)
theta = 1.234
R = ga.exp_bivector(axis, theta)
RRrev = ga.geometric_product(R, ga.reverse(R))
check("R Rrev == 1", np.allclose(RRrev, ga.ONE, atol=TOL))
check("||R|| == 1", np.isclose(ga.norm(R), 1.0, atol=TOL))

# 8. Sandwich rotation matches scipy Rotation (independent reference) for several
#    random axis-angle rotations, with the SAME v'=R v Rrev sign convention.
max_err = 0.0
for _ in range(200):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = rng.uniform(-2*np.pi, 2*np.pi)
    Rr = ga.exp_bivector(ax, th)   
    v = rng.normal(size=3)
    v_ga = ga.sandwich(Rr, v)
    v_scipy = scipyR.from_rotvec(ax * th).apply(v)  
    max_err = max(max_err, np.max(np.abs(v_ga - v_scipy)))
check(
    f"sandwich matches scipy Rotation (max err={max_err:.2e})", max_err < 1e-8)

# 9. Rotor composition matches scipy: R_total = R2 R1 should equal compose(R1 then R2)
ax1 = rng.normal(size=3)
ax1 /= np.linalg.norm(ax1)
th1 = 0.7
ax2 = rng.normal(size=3)
ax2 /= np.linalg.norm(ax2)
th2 = -1.1
R1 = ga.exp_bivector(ax1, th1)
R2 = ga.exp_bivector(ax2, th2)
Rtot = ga.geometric_product(R2, R1)
v = rng.normal(size=3)
v_ga = ga.sandwich(Rtot, v)
sciR1 = scipyR.from_rotvec(ax1*th1)
sciR2 = scipyR.from_rotvec(ax2*th2)
v_sci = sciR2.apply(sciR1.apply(v))
check("rotor composition R2*R1 matches scipy R2.apply(R1.apply(v))",
      np.allclose(v_ga, v_sci, atol=1e-8))

# 10. align_rotor: Ra aligns t_p to t_d
tp = rng.normal(size=3)
tp /= np.linalg.norm(tp)
td = rng.normal(size=3)
td /= np.linalg.norm(td)
Ra = ga.align_rotor(tp, td)
td_check = ga.sandwich(Ra, tp)
check("align_rotor aligns t_p -> t_d", np.allclose(td_check, td, atol=1e-7))
check("align_rotor produces unit rotor",
      np.isclose(ga.norm(Ra), 1.0, atol=1e-7))

# 11. rotor_log_angle_axis recovers (theta, axis) from exp_bivector
ax = rng.normal(size=3)
ax /= np.linalg.norm(ax)
th = 1.9  # in (0, 2pi) for a well-posed test (half-angle in [0,pi))
R = ga.exp_bivector(ax, th)
th_rec, ax_rec = ga.rotor_log_angle_axis(R)
check("rotor_log_angle_axis recovers theta", np.isclose(th_rec, th, atol=1e-7))
check("rotor_log_angle_axis recovers axis", np.allclose(ax_rec, ax, atol=1e-7))

# 12. Body angular-velocity bivector identity: Rdot = -(1/2) R Omega_b
#     Verify by finite-difference differentiating a known rotor curve R(t)
#     and checking Omega_b = -2 Rrev Rdot matches the construction omega.
# constant body angular velocity bivector
omega_true = np.array([0.3, -0.2, 0.5])
dt = 1e-6


def R_of_t(t):
    # Solve Rdot = -1/2 R Omega_b for constant Omega_b analytically via exp map
    th = np.linalg.norm(omega_true) * t
    ax = omega_true / np.linalg.norm(omega_true)
    # Rdot=-1/2 R Omega  => R(t) = R(0) * exp(-Omega t/2) in this convention
    # exp(-axis*theta/2)-style builder check below
    Rrev_t = ga.exp_bivector(ax, -th)
    return Rrev_t


# Build R(t) by repeatedly right-multiplying small steps consistent with Rdot=-1/2 R Omega
# i.e. R(t+dt) ~ R(t) * exp_bivector(axis, |omega|*dt)   [since Rrev=exp(n theta/2)]
axis = omega_true/np.linalg.norm(omega_true)
mag = np.linalg.norm(omega_true)
step_fwd = ga.exp_bivector(axis, mag*dt)   # small "Rrev-style" right factor
R_t = ga.rotor_identity()
R_t1 = ga.geometric_product(R_t, step_fwd)
Rdot_fd = (R_t1 - R_t) / dt
Omega_b_check = ga.to_bivector3(-2 *
                                ga.geometric_product(ga.reverse(R_t), Rdot_fd))
check("Omega_b = -2 Rrev Rdot matches constructed omega (eq.4)",
      np.allclose(Omega_b_check, omega_true, atol=1e-4))

print(f"\n{'ALL TESTS PASSED' if nfail == 0 else f'{nfail} TEST(S) FAILED'}")
sys.exit(0 if nfail == 0 else 1)
