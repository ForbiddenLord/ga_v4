import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import ga3 as ga
import stt
import ga_stt_controller as ctrl
import dynamics as dyn
import matplotlib.pyplot as plt

nfail = 0


def check(name, cond, extra=""):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name} {extra}")


stt_params = stt.STTParams(eta=[8, 8, 8], s0=[1, 1, 1], tc=16.0,
                           k1=0.15, k2=1.0, k3=0.5,
                           rho_max=1.0, rho_min=0.15, u=8.0,
                           rhoR0=1.0, rhoR_inf=0.05, kR=0.5)
drone = ctrl.DroneParams(mass=0.5, J_diag=[3.0e-3, 3.0e-3, 5.5e-3], f_min=0.0, f_max=12.0,
                         kappa1=1.5, kappa_v=2.0, kR=8.0, kOmega=0.2)
obstacles = [stt.Obstacle([4, 4, 4], [0.05, -0.02, 0.03], 0.6),
             stt.Obstacle([3, 5, 4.5], [-0.03, 0.04, 0.0], 0.5)]

# Start the drone WELL inside the tube (no translational saturation) and
# with a substantial initial attitude error, to isolate rotational-loop
# convergence under REAL dynamics.
state = dyn.UAVState(np.array([1.0, 1.0, 1.0]), np.zeros(3),
                     ga.exp_bivector(
                         np.array([0.3, 0.5, 0.1]) / np.linalg.norm([0.3, 0.5, 0.1]), 0.6),
                     np.zeros(3))
sigma = stt_params.s0.copy()
dt = 0.002
theta_e_hist = []
e_p_hist = []
for i in range(4001):
    t = i * dt
    ud = dict(p=state.p, v=state.v, R=state.R,
              Omega_b=state.Omega_b, sigma=sigma)
    f, tau, diag = ctrl.compute_control(
        t, ud, obstacles, stt_params, drone, ga.rotor_identity())
    theta_e_hist.append(diag["theta_e"])
    rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
    e_p_hist.append(np.linalg.norm(state.p - sigma) / rho_p_now)

    sigma_dot = stt.sigma_value(sigma, t, obstacles, stt_params)
    sigma = sigma + sigma_dot * dt

    def cfn(tl, s):
        return f, tau

    state, _, _ = dyn.rk4_step(state, dt, drone.m, drone.J_diag, cfn,
                               lambda tt: (np.zeros(3), np.zeros(3)), t)
    if not np.isfinite(f):
        break

theta_e_hist = np.array(theta_e_hist)
e_p_hist = np.array(e_p_hist)

print("theta_e: start=%.4f, t=1s=%.4f, t=4s=%.4f, t=8s=%.4f" %
      (theta_e_hist[0], theta_e_hist[int(1/dt)], theta_e_hist[int(4/dt)], theta_e_hist[int(8/dt)]))
print("e_p: max over run =", e_p_hist.max())

check("attitude error decreases substantially within 1s",
      theta_e_hist[int(1/dt)] < 0.5 * theta_e_hist[0])
check("attitude error stays bounded throughout (consistent with ISS, not necessarily monotonic)",
      np.all(theta_e_hist < 1.5), f"(max={theta_e_hist.max():.4f})")
check("attitude error eventually converges by t=8s",
      theta_e_hist[int(8/dt) - 1] < 0.05)
check("position stays strictly inside the translational tube throughout (e_p<1)",
      np.all(e_p_hist < 1.0), f"(max e_p={e_p_hist.max():.4f})")
check("no NaN/divergence over the full 8s run",
      np.all(np.isfinite(theta_e_hist)))

print(f"\n{'ALL TESTS PASSED' if nfail == 0 else f'{nfail} TEST(S) FAILED'}")
sys.exit(0 if nfail == 0 else 1)

plt.plot(theta_e_hist)
plt.show()
