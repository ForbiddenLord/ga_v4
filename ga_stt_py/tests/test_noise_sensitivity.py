"""
test_noise_sensitivity.py -- analyses the effect of measurement noise of 
Omega_b/v on GA-STT's Taylor-jet translational chain.

Sweeps Omega_b noise std from 0 (control case) up through plausible
gyro/EKF magnitudes, and separately v noise, and reports the first control
step at which EITHER roll or pitch torque saturates (|tau_i| >= tau_max_i),
plus peak |Omega| reached in the first 0.5s.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import ga3 as ga
import stt
import ga_stt_controller as ctrl
import simulate

STT_PARAMS = stt.STTParams(
    eta=[0.0, 0.0, 4.0], s0=[0.0, 0.0, 2.0], tc=20.0,
    k1=0.15, k2=1.0, k3=0.5,
    rho_max=2.0, rho_min=0.15, u=8.0,
    rhoR0=1.0, rhoR_inf=0.05, kR=0.5
)
DRONE = ctrl.DroneParams(
    mass=2.0, J_diag=[0.02167, 0.02167, 0.04],
    # kappa_v added (translational velocity-damping gain, required by the
    # corrected Stage-1 law -- see ga_stt_controller.py docstring); value
    # not independently re-tuned/validated for this specific scenario's
    # dynamics, only for the smaller drone in test_closed_loop_convergence.py.
    f_min=0.0, f_max=30.0, kappa1=3.0, kappa_v=3.0, kR=6.0, kOmega=2.91
)
TAU_MAX_VEC = np.array([1.3, 1.3, 0.4])
OBSTACLES = []
DT = 0.01
T_END = 1.0
P0 = np.array([0.0, 0.0, 2.0])
R0 = ga.rotor_identity()


def make_noise_fn(omega_std, v_std, p_std=0.0, r_angle_std=0.0, seed=0):
    """Gaussian measurement noise, sampled once per control step (100Hz),
    independent of the true RK4 physics. omega_std/v_std in rad/s, m/s."""
    def noise_fn(state, rng):
        out = {}
        if omega_std > 0:
            out['Omega_b'] = state.Omega_b + rng.normal(0, omega_std, 3)
        if v_std > 0:
            out['v'] = state.v + rng.normal(0, v_std, 3)
        if p_std > 0:
            out['p'] = state.p + rng.normal(0, p_std, 3)
        if r_angle_std > 0:
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.normal(0, r_angle_std)
            dR = ga.exp_bivector(axis, angle)
            out['R'] = ga.normalize_rotor(ga.geometric_product(state.R, dR))
        return out
    return noise_fn


def run_case(label, omega_std, v_std, p_std=0.0, r_angle_std=0.0):
    noise_fn = (make_noise_fn(omega_std, v_std, p_std, r_angle_std)
                if (omega_std or v_std or p_std or r_angle_std) else None)

    log = simulate.run_simulation(
        ctrl.compute_control, STT_PARAMS, DRONE, OBSTACLES,
        p0=P0, R0=R0, t_end=T_END, dt=DT,
        controller_kwargs=dict(Rp_value=ga.rotor_identity()),
        sensor_noise_fn=noise_fn,
    )

    tau = log["tau"]                       # (N, 3), physical N*m (post-fix)
    sat_rp = np.abs(tau[:, :2]) >= (TAU_MAX_VEC[:2] - 1e-6)
    sat_steps = np.where(sat_rp.any(axis=1))[0]
    first_sat = int(sat_steps[0]) if len(sat_steps) else None
    first_sat_t = log["t"][first_sat] if first_sat is not None else None

    theta_e = log["theta_e"]
    peak_theta_e_deg = np.degrees(np.nanmax(theta_e[:int(0.5 / DT)]))

    print(f"{label:32s}  first_sat_step={str(first_sat):>4}  "
          f"first_sat_t={('%.3f' % first_sat_t) if first_sat_t is not None else '  -- ':>6}  "
          f"peak_theta_e(0.5s)={peak_theta_e_deg:6.2f} deg")
    return first_sat_t, peak_theta_e_deg


if __name__ == "__main__":
    print("=== Control case: zero measurement noise ===")
    run_case("noiseless", 0.0, 0.0)

    print("\n=== Omega_b (gyro) noise sweep, v clean ===")
    for omega_std in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        run_case(f"omega_std={omega_std:.3f} rad/s", omega_std, 0.0)

    print("\n=== v (EKF velocity) noise sweep, Omega_b clean ===")
    for v_std in [0.01, 0.02, 0.05, 0.1, 0.2]:
        run_case(f"v_std={v_std:.3f} m/s", 0.0, v_std)

    print("\n=== Combined, plausible SITL-realistic magnitudes ===")
    run_case("omega=0.02 rad/s, v=0.02 m/s", 0.02, 0.02)
    run_case("omega=0.05 rad/s, v=0.05 m/s", 0.05, 0.05)
