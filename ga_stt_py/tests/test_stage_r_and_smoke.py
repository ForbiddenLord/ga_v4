"""Mirrors ga_stt_cpp/tests/test_stage_r_and_smoke.cpp -- see that file's
comments for why the old "exact closed-form V_R_dot match" check was
replaced with a Lyapunov-decrease check plus a no-singularity check."""
import numpy as np
import scipy.integrate as scint
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import ga_stt_controller as ctrl
import stt
import ga3 as ga

nfail = 0


def check(name, cond, extra=""):
    global nfail
    status = "PASS" if cond else "FAIL"
    if not cond:
        nfail += 1
    print(f"[{status}] {name} {extra}")


stt_params = stt.STTParams(eta=[8, 8, 8], s0=[1, 1, 1], tc=16.0,
                           k1=0.3, k2=1.0, k3=0.5,
                           rho_max=1.0, rho_min=0.15, u=8.0,
                           rhoR0=1.2, rhoR_inf=0.05, kR=0.8)
drone = ctrl.DroneParams(mass=0.03, J_diag=[1.6e-5, 1.6e-5, 2.9e-5],
                         f_min=0.0, f_max=1.0, kappa1=4.0, kappa_v=2.5,
                         kR=6.0, kOmega=0.01)


def eval_stage_r(Rd, R, t):
    Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd), R))
    e_R_vee = ga.rotor_bivec3(Re)
    Psi_e = 1.0 - ga.rotor_scalar(Re)
    rho_R_t = stt.rho_R(t, stt_params)
    e_R = np.clip(Psi_e / rho_R_t, 0.0, 1.0 - 1e-6)
    eps_R = -np.log(max(1.0 - e_R, 1e-6))
    xi_R = 1.0 / (rho_R_t * (1.0 - e_R))
    Omega_c = drone.kR * xi_R * e_R_vee  # Omega_d = 0 for this isolated check; sign per corrigendum
    return dict(Psi_e=Psi_e, e_R=e_R, eps_R=eps_R, xi_R=xi_R, e_R_vee=e_R_vee, Omega_c=Omega_c)


rng = np.random.default_rng(3)
Rd = ga.normalize_rotor(ga.exp_bivector(np.array([0.1, 0.8, 0.2]) / np.linalg.norm([0.1, 0.8, 0.2]), 0.4))

# --- Check #1: V_R strictly decreases under z_Omega = 0 ---
lyapunov_ok = True
lyapunov_detail = ""
for theta_e_target in [0.02, 0.1, 0.3, 0.5, 1.0, 1.5]:
    for t0 in [0.0, 2.0, 8.0]:
        axis = np.array([0.3, -0.2, 0.5]); axis = axis / np.linalg.norm(axis)
        Re0 = ga.exp_bivector(axis, theta_e_target)
        R0 = ga.normalize_rotor(ga.geometric_product(Rd, Re0))
        s0 = eval_stage_r(Rd, R0, t0)

        dt = 1e-5
        Rdot = -0.5 * ga.geometric_product(R0, ga.bivector(s0["Omega_c"]))
        R1 = ga.normalize_rotor(R0 + Rdot * dt)
        t1 = t0 + dt
        s1 = eval_stage_r(Rd, R1, t1)

        VR0 = 0.5 * s0["eps_R"] ** 2
        VR1 = 0.5 * s1["eps_R"] ** 2
        VR_dot_numeric = (VR1 - VR0) / dt

        if s0["eps_R"] > 1e-4 and VR_dot_numeric > 1e-3:
            lyapunov_ok = False
            lyapunov_detail = f"theta_e={theta_e_target} t0={t0} VR_dot={VR_dot_numeric}"

check("Stage-R: V_R strictly decreases under z_Omega=0 (all sampled errors/times)",
     lyapunov_ok, lyapunov_detail)

# --- Check #2: no singularity / no chatter as theta_e -> 0 ---
smooth_ok = True
axis = np.array([0.3, -0.2, 0.5]); axis = axis / np.linalg.norm(axis)
prev_norm, prev_theta = -1.0, -1.0
for theta_e_target in [1e-4, 1e-3, 1e-2, 1e-1, 3e-1]:
    Re0 = ga.exp_bivector(axis, theta_e_target)
    R0 = ga.normalize_rotor(ga.geometric_product(Rd, Re0))
    s0 = eval_stage_r(Rd, R0, 2.0)
    if not np.all(np.isfinite(s0["Omega_c"])):
        smooth_ok = False
    if prev_norm >= 0.0:
        ratio_theta = theta_e_target / prev_theta
        ratio_norm = np.linalg.norm(s0["Omega_c"]) / max(prev_norm, 1e-9)
        if ratio_norm > 3.0 * ratio_theta or ratio_norm < 0.3 * ratio_theta:
            smooth_ok = False
    prev_norm = np.linalg.norm(s0["Omega_c"])
    prev_theta = theta_e_target

check("Stage-R: Omega_c finite and ~linear in theta_e near 0 (no 1/sin singularity)", smooth_ok)

# --- Smoke test: full pipeline at near-equilibrium ---
obstacles = [stt.Obstacle([4, 4, 4], [0.05, -0.02, 0.03], 0.6)]
t_test = 0.3
sol = scint.solve_ivp(lambda t, s: stt.sigma_value(s, t, obstacles, stt_params),
                      [0, t_test], stt_params.s0, method="RK45", rtol=1e-12, atol=1e-13)
sigma_now = sol.y[:, -1]
sigma_dot_now = stt.sigma_value(sigma_now, t_test, obstacles, stt_params)

state = dict(p=sigma_now.copy(), v=sigma_dot_now.copy(),
            R=ga.rotor_identity(), Omega_b=np.zeros(3), sigma=sigma_now.copy())
Rp_value = ga.rotor_identity()
f_cmd, tau, diag = ctrl.compute_control(t_test, state, obstacles, stt_params, drone, Rp_value)

check("thrust command within actuator bounds", drone.f_min <= f_cmd <= drone.f_max)
check("thrust command near-hover when drone exactly tracks the tube center",
     abs(f_cmd - drone.m * 9.81) < 0.1, f"(f_cmd={f_cmd:.4f}, mg={drone.m*9.81:.4f})")
check("torque finite and small at near-equilibrium",
     np.all(np.isfinite(tau)) and np.linalg.norm(tau) < 1.0, f"(tau={tau})")
check("Rd is a valid unit rotor", np.isclose(ga.norm(diag["Rd"]), 1.0, atol=1e-6))

print(f"\n{'ALL TESTS PASSED' if nfail == 0 else f'{nfail} TEST(S) FAILED'}")
sys.exit(0 if nfail == 0 else 1)
