import sympy as sp
import numpy as np
from math import factorial, sin, cos, sqrt as msqrt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import jet1d as jt

nfail = 0


def check(name, cond, extra=""):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name} {extra}")


t0 = 0.83
N = 4

# f(t) = sin(2t) * sqrt(t+3) / (1+t^2)   — exercise mul, sqrt, div all at once
# Build via jets and compare against sympy's exact derivatives at t0.
t = sp.symbols('t')
f_sym = sp.sin(2*t) * sp.sqrt(t+3) / (1+t**2)
true_derivs = [float(sp.diff(f_sym, t, k).subs(t, t0)) for k in range(N+1)]

# Build jets for sin(2t), sqrt(t+3), (1+t^2) by using primitives
# (no jet_sin yet, so construct sin/cos via known closed-form Taylor coeffs
#  at t0 directly using math derivatives of sin)


def sin_jet(omega, t0, order):
    c = np.zeros(order+1)
    for k in range(order+1):
        # d^k/dt^k sin(omega t) = omega^k sin(omega t + k pi/2)
        c[k] = (omega**k) * sin(omega*t0 + k*np.pi/2) / factorial(k)
    return c


sqrt_arg = jt.const_scalar(t0+3, N)
sqrt_arg[1] = 1.0  # d/dt(t+3)=1, higher derivs 0
sin_part = sin_jet(2.0, t0, N)
sqrt_part = jt.sqrt_s(sqrt_arg)
denom = jt.const_scalar(1+t0**2, N)
denom[1] = 2*t0
denom[2] = 1.0  # 1/2! * d2/dt2(t^2)=1*... wait Taylor coeff_2 = (1/2!)*2=1
num = jt.mul_ss(sin_part, sqrt_part)
f_jet = jt.div_ss(num, denom)

computed_derivs = [jt.coeff_to_derivative(f_jet, k) for k in range(N+1)]
for k in range(N+1):
    ok = np.isclose(computed_derivs[k], true_derivs[k], rtol=1e-6, atol=1e-8)
    check(f"sin*sqrt/poly: d^{k}f/dt^{k}", ok,
          f"(got {computed_derivs[k]:.6f}, want {true_derivs[k]:.6f})")

# exp_s and log_s tests: f(t) = exp(0.5 t) and g(t)=log(2+t), compare to sympy
f2 = sp.exp(0.5*t)
true2 = [float(sp.diff(f2, t, k).subs(t, t0)) for k in range(N+1)]
arg = jt.const_scalar(0.5*t0, N)
arg[1] = 0.5
ej = jt.exp_s(arg)
comp2 = [jt.coeff_to_derivative(ej, k) for k in range(N+1)]
for k in range(N+1):
    check(f"exp(0.5t): d^{k}", np.isclose(comp2[k], true2[k], rtol=1e-6),
          f"(got {comp2[k]:.6f}, want {true2[k]:.6f})")

g2 = sp.log(2+t)
true3 = [float(sp.diff(g2, t, k).subs(t, t0)) for k in range(N+1)]
arg2 = jt.const_scalar(2+t0, N)
arg2[1] = 1.0
lj = jt.log_s(arg2)
comp3 = [jt.coeff_to_derivative(lj, k) for k in range(N+1)]
for k in range(N+1):
    check(f"log(2+t): d^{k}", np.isclose(comp3[k], true3[k], rtol=1e-6),
          f"(got {comp3[k]:.6f}, want {true3[k]:.6f})")

# Vector ops: u(t)=(t, t^2, sin t), v(t)=(cos t, t, 1) — test dot, cross, norm
ux, uy = t, t**2
vx, vz = sp.cos(t), t
u_sym = sp.Matrix([t, t**2, sp.sin(t)])
v_sym = sp.Matrix([sp.cos(t), t, 1])
dot_sym = (u_sym.T*v_sym)[0]
cross_sym = u_sym.cross(v_sym)
norm_sym = sp.sqrt((u_sym.T*u_sym)[0])


def vec_jet_from_sym(expr_vec, t0, order):
    c = np.zeros((order+1, 3))
    for k in range(order+1):
        for ax in range(3):
            c[k, ax] = float(
                sp.diff(expr_vec[ax], t, k).subs(t, t0)) / factorial(k)
    return c


u_jet = vec_jet_from_sym(u_sym, t0, N)
v_jet = vec_jet_from_sym(v_sym, t0, N)

dot_jet = jt.dot_vv(u_jet, v_jet)
true_dot = [float(sp.diff(dot_sym, t, k).subs(t, t0)) for k in range(N+1)]
comp_dot = [jt.coeff_to_derivative(dot_jet, k) for k in range(N+1)]
for k in range(N+1):
    check(f"dot(u,v): d^{k}", np.isclose(comp_dot[k], true_dot[k], rtol=1e-6),
          f"(got {comp_dot[k]:.6f}, want {true_dot[k]:.6f})")

cross_jet = jt.cross_vv(u_jet, v_jet)
for ax in range(3):
    true_c = [float(sp.diff(cross_sym[ax], t, k).subs(t, t0))
              for k in range(N+1)]
    comp_c = [jt.coeff_to_derivative(cross_jet[:, ax], k) for k in range(N+1)]
    for k in range(N+1):
        check(f"cross(u,v)[{ax}]: d^{k}", np.isclose(comp_c[k], true_c[k], rtol=1e-6, atol=1e-7),
              f"(got {comp_c[k]:.6f}, want {true_c[k]:.6f})")

norm_jet = jt.norm_v(u_jet)
true_norm = [float(sp.diff(norm_sym, t, k).subs(t, t0)) for k in range(N+1)]
comp_norm = [jt.coeff_to_derivative(norm_jet, k) for k in range(N+1)]
for k in range(N+1):
    check(f"norm(u): d^{k}", np.isclose(comp_norm[k], true_norm[k], rtol=1e-6),
          f"(got {comp_norm[k]:.6f}, want {true_norm[k]:.6f})")

print(f"\n{'ALL TESTS PASSED' if nfail == 0 else f'{nfail} TEST(S) FAILED'}")
sys.exit(0 if nfail == 0 else 1)
