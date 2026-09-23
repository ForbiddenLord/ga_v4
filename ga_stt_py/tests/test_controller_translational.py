"""Mirrors ga_stt_cpp/tests/test_controller_translational.cpp: verifies the
new Taylor/Picard resolve_translational_chain against a direct finite-
difference of the SAME u1(p,v,t) function (RK4-integrated), on an
obstacle-free (well-conditioned) scenario -- see the C++ test for why a
near-singular obstacle configuration isn't a meaningful FD ground truth
here (that regime is instead covered by the accel_cap and by
test_closed_loop_convergence)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import stt
import ga_stt_controller as ctrl

nfail = 0


def check(name, cond, extra=""):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name} {extra}")


stt_params = stt.STTParams(eta=[8, 8, 8], s0=[1, 1, 1], tc=16.0,
                           k1=0.3, k2=1.0, k3=0.5,
                           rho_max=1.0, rho_min=0.15, u=8.0)
obstacles = [
    stt.Obstacle([4, 4, 4], [0.05, -0.02, 0.03], 0.6),
    stt.Obstacle([3, 5, 4.5], [-0.03, 0.04, 0.0], 0.5),
]
# Gains chosen so accel_cap is not triggered (isolates the Picard/Taylor
# ODE-propagation arithmetic itself; see C++ test for the same choice).
drone = ctrl.DroneParams(mass=0.03, J_diag=[1.6e-5, 1.6e-5, 2.9e-5],
                         f_min=0.0, f_max=1.0, kappa1=1.0, kappa_v=1.0,
                         kR=6.0, kOmega=0.01)

t0 = 3.1
# sigma0 with NO active obstacle nearby.
sigma0 = np.array([1.5, 1.5, 1.5])
p0 = sigma0 + np.array([0.05, -0.03, 0.02])
v0 = np.array([0.2, -0.1, 0.05])

sjet = stt.sigma_jet(sigma0, t0, obstacles, stt_params, ctrl.JET_ORDER_SIGMA)
rjet = stt.rho_p_jet(sjet, t0, obstacles, stt_params, ctrl.JET_ORDER_SIGMA)

Fd_jet, fd_jet, that_d_jet = ctrl.resolve_translational_chain(
    t0, p0, v0, sjet, rjet, drone)

m, g = drone.m, 9.81
E3 = np.array([0., 0., 1.])


def eval_deriv(c, k, t, t0):
    """m-th derivative of the Taylor polynomial with coefficients c, at t."""
    dt = t - t0
    val = np.zeros(c.shape[1:]) if c.ndim > 1 else 0.0
    for j in range(k, c.shape[0]):
        coeff = 1.0
        for i in range(k):
            coeff *= (j - i)
        dt_pow = 1.0 if (j - k == 0) else dt ** (j - k)
        val = val + coeff * c[j] * dt_pow
    return val


def u1_value(t, p, v):
    sigma_t = eval_deriv(sjet, 0, t, t0)
    sigma_dot_t = eval_deriv(sjet, 1, t, t0)
    sigma_ddot_t = eval_deriv(sjet, 2, t, t0)
    rho_t = eval_deriv(rjet, 0, t, t0)
    p_tilde = p - sigma_t
    norm_pt = np.sqrt(p_tilde @ p_tilde + 1e-6)
    e1 = norm_pt / rho_t
    eps_p = np.log((1 + e1) / (1 - e1))
    p_hat = p_tilde / norm_pt
    xi_p_over_rho = 2.0 / (rho_t * (1 - e1 * e1))
    z_v = v - sigma_dot_t
    return sigma_ddot_t - drone.kappa_v * z_v - drone.kappa1 * xi_p_over_rho * eps_p * p_hat


def Fd_value(p, v, t):
    return m * (u1_value(t, p, v) + g * E3)


def rk4_traj(t_target):
    n = max(1, int(abs(t_target - t0) / 1e-4))
    dt = (t_target - t0) / n
    p, v, t = p0.copy(), v0.copy(), t0
    for _ in range(n):
        def deriv(pp, vv, tt):
            return vv, u1_value(tt, pp, vv)
        k1p, k1v = deriv(p, v, t)
        k2p, k2v = deriv(p + k1p * dt / 2, v + k1v * dt / 2, t + dt / 2)
        k3p, k3v = deriv(p + k2p * dt / 2, v + k2v * dt / 2, t + dt / 2)
        k4p, k4v = deriv(p + k3p * dt, v + k3v * dt, t + dt)
        p = p + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
    return p, v


h = 2e-3
p_plus, v_plus = rk4_traj(t0 + h)
p_minus, v_minus = rk4_traj(t0 - h)
Fd0_direct = Fd_value(p0, v0, t0)
Fd_plus = Fd_value(p_plus, v_plus, t0 + h)
Fd_minus = Fd_value(p_minus, v_minus, t0 - h)
Fd_dot_direct = (Fd_plus - Fd_minus) / (2 * h)
Fd_ddot_direct = (Fd_plus - 2 * Fd0_direct + Fd_minus) / (h * h)

check("Fd(t0) finite and positive", np.all(np.isfinite(Fd_jet[0])) and np.linalg.norm(Fd_jet[0]) > 0)
check("Fd(t0) matches direct evaluation",
     np.linalg.norm(Fd_jet[0] - Fd0_direct) < 1e-3,
     f"jet={Fd_jet[0]} direct={Fd0_direct}")
check("Fd_dot(t0) matches RK4 finite-difference ground truth",
     np.linalg.norm(Fd_jet[1] - Fd_dot_direct) < 5e-2 * max(1.0, np.linalg.norm(Fd_dot_direct)),
     f"jet={Fd_jet[1]} fd={Fd_dot_direct}")
check("Fd_ddot(t0)/2 matches RK4 finite-difference ground truth",
     np.linalg.norm(2 * Fd_jet[2] - Fd_ddot_direct) < 5e-1 * max(1.0, np.linalg.norm(Fd_ddot_direct)),
     f"jet={2*Fd_jet[2]} fd={Fd_ddot_direct}")
check("thrust direction is a unit vector", abs(np.linalg.norm(that_d_jet[0]) - 1.0) < 1e-3)
check("commanded thrust magnitude positive and finite", np.isfinite(fd_jet[0]) and fd_jet[0] > 0)

print("\n" + ("ALL TESTS PASSED" if nfail == 0 else f"{nfail} TEST(S) FAILED"))
sys.exit(0 if nfail == 0 else 1)
