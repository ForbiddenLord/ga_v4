import sys, os
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import stt
from jet1d import coeff_to_derivative

nfail = 0


def check(name, cond, extra=""):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name} {extra}")


params = stt.STTParams(eta=[8, 8, 8], s0=[1, 1, 1], tc=16.0,
                       k1=0.3, k2=1.0, k3=0.5,
                       rho_max=1.0, rho_min=0.15, u=8.0)
obstacles = [
    stt.Obstacle([4, 4, 4], [0.05, -0.02, 0.03], 0.6),
    stt.Obstacle([3, 5, 4.5], [-0.03, 0.04, 0.0], 0.5),
    stt.Obstacle([5, 3, 5], [0.0, 0.0, -0.02], 0.4),
]


def rhs(t, sigma):
    return stt.sigma_value(sigma, t, obstacles, params)


t0 = 6.37  # an interior time, away from t=0 and t=tc
sol = solve_ivp(rhs, [0, t0], params.s0, method="RK45",
                rtol=1e-12, atol=1e-13, dense_output=True)
sigma0 = sol.y[:, -1]
print("sigma(t0) from high-accuracy ODE solve:", sigma0)

# Ground-truth higher derivatives via repeated high-accuracy finite differences
# of the dense_output solution (5-point central stencils at decreasing h to
# extrapolate), used purely as an independent check on the jet machinery.


def sigma_of_t(t):
    if t < 0:
        # extend solution backward is unnecessary; just don't query negative t
        raise ValueError
    sol2 = solve_ivp(rhs, [0, t], params.s0,
                     method="RK45", rtol=1e-12, atol=1e-13)
    return sol2.y[:, -1]


h = 1e-3
# 1st derivative: central difference, O(h^4) via 5-point stencil


def fd1(t0, h):
    fm2, fm1, fp1, fp2 = sigma_of_t(
        t0-2*h), sigma_of_t(t0-h), sigma_of_t(t0+h), sigma_of_t(t0+2*h)
    return (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12*h)


def fd2(t0, h):
    fm2, fm1, f0, fp1, fp2 = (sigma_of_t(t0-2*h), sigma_of_t(t0-h), sigma_of_t(t0),
                              sigma_of_t(t0+h), sigma_of_t(t0+2*h))
    return (-fp2 + 16*fp1 - 30*f0 + 16*fm1 - fm2) / (12*h**2)


def fd3(t0, h):
    fm2, fm1, fp1, fp2 = sigma_of_t(
        t0-2*h), sigma_of_t(t0-h), sigma_of_t(t0+h), sigma_of_t(t0+2*h)
    return (fp2 - 2*fp1 + 2*fm1 - fm2) / (2*h**3)


d1_fd = fd1(t0, h)
d2_fd = fd2(t0, h)
d3_fd = fd3(t0, h)

order = 3
sjet = stt.sigma_jet(sigma0, t0, obstacles, params, order)
d0_jet = coeff_to_derivative(sjet, 0)
d1_jet = coeff_to_derivative(sjet, 1)
d2_jet = coeff_to_derivative(sjet, 2)
d3_jet = coeff_to_derivative(sjet, 3)

print("d1 jet:", d1_jet, " fd:", d1_fd)
print("d2 jet:", d2_jet, " fd:", d2_fd)
print("d3 jet:", d3_jet, " fd:", d3_fd)

check("sigma(t0) matches", np.allclose(d0_jet, sigma0, atol=1e-9))
check("sigma_dot matches direct eq.6 eval",
      np.allclose(d1_jet, stt.sigma_value(sigma0, t0, obstacles, params), atol=1e-9))
check("sigma_dot jet matches finite-diff (rtol 1e-4)", np.allclose(d1_jet, d1_fd, rtol=1e-4, atol=1e-5),
      f"max diff={np.max(np.abs(d1_jet-d1_fd)):.2e}")
check("sigma_ddot jet matches finite-diff (rtol 1e-2)", np.allclose(d2_jet, d2_fd, rtol=1e-2, atol=1e-3),
      f"max diff={np.max(np.abs(d2_jet-d2_fd)):.2e}")
check("sigma_dddot jet matches finite-diff (rtol 5e-2)", np.allclose(d3_jet, d3_fd, rtol=5e-2, atol=1e-2),
      f"max diff={np.max(np.abs(d3_jet-d3_fd)):.2e}")

# Also test rho_p jet against finite differences


def rho_of_t(t):
    s = sigma_of_t(t) if t > 1e-9 else params.s0
    return stt.rho_p_value(s, t, obstacles, params)


rp_jet = stt.rho_p_jet(sjet, t0, obstacles, params, order)
rp_d0 = coeff_to_derivative(rp_jet, 0)
rp_d1 = coeff_to_derivative(rp_jet, 1)
rp_true0 = stt.rho_p_value(sigma0, t0, obstacles, params)
hh = 1e-3
rp_d1_fd = (rho_of_t(t0+hh) - rho_of_t(t0-hh)) / (2*hh)
check("rho_p(t0) matches direct eval", np.isclose(rp_d0, rp_true0, atol=1e-9))
check("rho_p_dot jet matches finite-diff", np.isclose(rp_d1, rp_d1_fd, rtol=1e-3, atol=1e-4),
      f"(jet={rp_d1:.6f}, fd={rp_d1_fd:.6f})")

print(f"\n{'ALL TESTS PASSED' if nfail == 0 else f'{nfail} TEST(S) FAILED'}")
sys.exit(0 if nfail == 0 else 1)
