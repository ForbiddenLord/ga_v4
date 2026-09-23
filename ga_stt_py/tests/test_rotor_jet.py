import sympy as sp
from math import factorial
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import ga3 as ga
import jet1d as jt
import rotor_jet as rj

nfail = 0


def check(name, cond, extra=""):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name} {extra}")

# Synthetic smooth td(t), tp(t) trajectories (unit vectors)


def td_raw(t):
    v = np.array([np.sin(t) + 0.3*np.sin(3*t),
                 np.cos(t), 1.5 + 0.2*np.cos(2*t)])
    return v / np.linalg.norm(v)


def tp_raw(t):
    v = np.array([0.1*t, 0.0, 1.0])
    return v / np.linalg.norm(v)


def Rd_direct(t):
    """Ground truth Rd(t) = Ra(t) (Rp = identity here) via direct ga3 construction."""
    return ga.align_rotor(tp_raw(t), td_raw(t))


t0 = 1.234

# --- value check ---
Rd0_direct = Rd_direct(t0)

# --- jet-based computation ---
order = 3
# Build td_jet, tp_jet as jet1d vector jets using sympy


def raw_vec_jet(order, t0, fn_components):
    """fn_components: list of 3 callables t-> derivative-k via sympy for exactness."""
    import sympy as sp
    t = sp.symbols('t')
    c = np.zeros((order+1, 3))
    for ax in range(3):
        expr = fn_components[ax](t)
        for k in range(order+1):
            c[k, ax] = float(sp.diff(expr, t, k).subs(t, t0)) / factorial(k)
    return c


t_sym = sp.symbols('t')
td_num = sp.Matrix([sp.sin(t_sym) + sp.Rational(3, 10)*sp.sin(3*t_sym), sp.cos(t_sym),
                    sp.Rational(3, 2) + sp.Rational(2, 10)*sp.cos(2*t_sym)])
td_norm = sp.sqrt((td_num.T*td_num)[0])
td_unit = td_num / td_norm

tp_num = sp.Matrix([sp.Rational(1, 10)*t_sym, 0, 1])
tp_norm = sp.sqrt((tp_num.T*tp_num)[0])
tp_unit = tp_num / tp_norm


def vec_jet_from_sym(expr_vec, t0, order):
    c = np.zeros((order+1, 3))
    for k in range(order+1):
        for ax in range(3):
            c[k, ax] = float(sp.diff(expr_vec[ax], t_sym, k).subs(
                t_sym, t0)) / factorial(k)
    return c


td_jet = vec_jet_from_sym(td_unit, t0, order)
tp_jet = vec_jet_from_sym(tp_unit, t0, order)

check("td_jet value matches td_raw", np.allclose(
    td_jet[0], td_raw(t0), atol=1e-8))
check("tp_jet value matches tp_raw", np.allclose(
    tp_jet[0], tp_raw(t0), atol=1e-8))

Ra_jet = rj.align(tp_jet, td_jet)
Ra0 = ga.rotor(jt.value(Ra_jet.s), jt.value(Ra_jet.b))
check("Ra_jet value matches direct align_rotor", np.allclose(Ra0, Rd0_direct, atol=1e-7),
      f"max diff={np.max(np.abs(Ra0-Rd0_direct)):.2e}")

# Omega_d = -2 Rrev Rdot  (GA-STT eq.43), Omega_dot_d via one more differentiate
Rd_jet = Ra_jet  # Rp = identity here
Rd_rev = rj.reverse(Rd_jet)
Rd_dot_jet = rj.differentiate(Rd_jet)  # order-1 lower
Omega_d_jet_full = rj.mul(Rd_rev, Rd_dot_jet)
# scale by -2 (this RotorJet's "b" component IS Omega_d itself, scalar part should be ~0)
Omega_d_s = jt.scale(Omega_d_jet_full.s, -2.0)
Omega_d_b = jt.scale(Omega_d_jet_full.b, -2.0)

check("Omega_d scalar part ~ 0 (since R~ Rdot is a pure bivector for unit rotor)",
      abs(jt.value(Omega_d_s)) < 1e-7, f"(got {jt.value(Omega_d_s):.2e})")

Omega_d_value = jt.value(Omega_d_b)
Omega_d_dot_value = jt.coeff_to_derivative(Omega_d_b, 1)

# --- Ground truth via finite differences of the DIRECTLY constructed Rd(t) ---
h = 1e-4


def Omega_d_fd_at(t):
    Rp_, Rm_ = Rd_direct(t+h/2), Rd_direct(t-h/2)
    Rdot = (Rp_ - Rm_) / h
    Rval = Rd_direct(t)
    Om = -2 * ga.geometric_product(ga.reverse(Rval), Rdot)
    return ga.to_bivector3(Om)


Omega_d_fd = Omega_d_fd_at(t0)
hh = 1e-3
Omega_d_dot_fd = (Omega_d_fd_at(t0+hh) - Omega_d_fd_at(t0-hh)) / (2*hh)

check("Omega_d matches finite-difference ground truth",
      np.allclose(Omega_d_value, Omega_d_fd, atol=1e-4),
      f"(jet={Omega_d_value}, fd={Omega_d_fd})")
check("Omega_dot_d matches finite-difference ground truth",
      np.allclose(Omega_d_dot_value, Omega_d_dot_fd, atol=2e-2),
      f"(jet={Omega_d_dot_value}, fd={Omega_d_dot_fd})")

print(f"\n{'ALL TESTS PASSED' if nfail == 0 else f'{nfail} TEST(S) FAILED'}")
sys.exit(0 if nfail == 0 else 1)
