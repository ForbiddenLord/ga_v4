"""
run_cf2_validation_realistic.py -- Crazyflie 2 with motor/ESC dynamics,
sensor noise, and communication delay, using gains derived via the same
corrected time-scale-separation methodology as
run_x500_validation_realistic.py. See TIME_SCALE_SEPARATION_ANALYSIS.md
for the full derivation and, importantly, the same negative result found
here as for the X500: the ORIGINAL (ideal-actuator) tube schedule in
run_cf2_validation.py (rho_max=0.35, tc=8.0) could NOT be kept safe under
a 20ms motor-lag assumption with any gains tried, theoretically-motivated
or otherwise; a relaxed schedule (rho_max=0.6, tc=14.0) is what actually
works. Rotor arm length here (0.0608 m, i.e. (x,y)=(+-0.043,+-0.043)) is
an approximation of the real CF2's ~4.6 cm arm (not re-derived from the
Menagerie mesh); k_m=0.005 is a rough estimate for CF2-scale propellers
(smaller than the X500's 0.02, consistent with much smaller, lower-drag
props) -- neither is hardware-measured.

GAINS (corrected time-scale-separation method):
    tau_motor = 0.02 s  (CF2's tiny coreless motors: faster than the
                        X500's 5010-class motors -- literature-typical
                        guess, not measured)
    tau_Omega = 3*tau_motor = 0.06 s  -> kappa_Omega = 0.00027
    tau_R     = 3*tau_Omega = 0.18 s  -> kappa_R     = 2.778
    kappa1=0.2, kappa_v from the zeta=0.9 heuristic (2.68): found,
    empirically, inside the same "slower than Stage R, fast enough for
    the relaxed tube" window as the X500 case.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import mujoco
import ga3 as ga
import stt
import ga_stt_controller_final as ctrl
from run_cf2_validation import quat_wxyz_to_rotor, rotor_to_quat_wxyz
from hardware_realism import QuadMixer, MotorModel, SensorNoise, CommDelay

ROTOR_XY = [(0.043, -0.043), (-0.043, 0.043), (0.043, 0.043), (-0.043, -0.043)]
ROTOR_SPIN = ['ccw', 'ccw', 'cw', 'cw']
K_M = 0.005
TAU_MOTOR = 0.02
COMM_DELAY_STEPS = 3

KAPPA1, KAPPA_V, KR, KOMEGA = 0.2, 2.68, 2.7778, 0.00027
RHO_MAX, TC = 0.6, 14.0


def make_scenario():
    stt_params = stt.STTParams(eta=[1.2, 1.2, 1.0], s0=[0.0, 0.0, 0.4], tc=TC,
                               k1=0.2, k2=1.0, k3=0.5, rho_max=RHO_MAX, rho_min=0.08, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=1.0)
    drone = ctrl.DroneParams(mass=0.027, J_diag=[2.3951e-5, 2.3951e-5, 3.2347e-5],
                             f_min=0.0, f_max=0.60,
                             kappa1=KAPPA1, kappa_v=KAPPA_V, kR=KR, kOmega=KOMEGA,
                             tau_max=0.012)
    obstacles = [stt.Obstacle([0.6, 0.6, 0.5], [0.0, 0.0, 0.0], 0.15)]
    return stt_params, drone, obstacles


def run(duration=14.0, dt=0.002, verbose_every=500, seed=0):
    stt_params, drone, obstacles = make_scenario()

    xml_path = os.path.join(os.path.dirname(__file__), "crazyflie_menagerie", "cf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    cf2_id = model.body("cf2").id

    data.qpos[0:3] = stt_params.s0
    data.qpos[3:7] = rotor_to_quat_wxyz(ga.rotor_identity())
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    mixer = QuadMixer(ROTOR_XY, ROTOR_SPIN, K_M)
    motor_max = 1.6 * drone.f_max / 4.0
    motors = MotorModel(tau_motor=TAU_MOTOR, t_min=0.0, t_max=motor_max)
    motors.reset(mixer.wrench_to_motors(drone.m * 9.81, np.zeros(3)))
    noise = SensorNoise(gyro_noise_std=0.01, gyro_bias_walk_std=0.002,
                        pos_noise_std=0.01, vel_noise_std=0.02, seed=seed)
    delay = CommDelay(COMM_DELAY_STEPS, shape=(4,), init_value=[drone.m * 9.81, 0.0, 0.0, 0.0])

    sigma = stt_params.s0.copy()
    theta_e_hist, e_p_hist, t_hist = [], [], []
    n_steps = int(duration / dt)
    diverged = False

    for i in range(n_steps + 1):
        t = i * dt
        p_true = data.xpos[cf2_id].copy()
        R = quat_wxyz_to_rotor(data.xquat[cf2_id])
        v_true = data.cvel[cf2_id][3:6].copy()
        Omega_true = data.cvel[cf2_id][0:3].copy()

        p_noisy, v_noisy, Omega_noisy = noise.apply(p_true, v_true, Omega_true, dt)
        ud = dict(p=p_noisy, v=v_noisy, R=R, Omega_b=Omega_noisy, sigma=sigma)
        f_cmd, tau, diag = ctrl.compute_control(t, ud, obstacles, stt_params, drone, ga.rotor_identity())
        if not (np.isfinite(f_cmd) and np.all(np.isfinite(tau))):
            diverged = True
            break

        delayed = delay.push_and_get(np.array([f_cmd, tau[0], tau[1], tau[2]]))
        T_cmd = mixer.wrench_to_motors(delayed[0], delayed[1:4])
        T_actual = motors.step(T_cmd, dt)
        f_actual, tau_actual = mixer.motors_to_wrench(T_actual)

        data.xfrc_applied[cf2_id, 0:3] = ga.sandwich(R, np.array([0.0, 0.0, f_actual]))
        data.xfrc_applied[cf2_id, 3:6] = ga.sandwich(R, tau_actual)

        theta_e_hist.append(diag["theta_e"])
        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        e_p_hist.append(np.linalg.norm(p_true - sigma) / rho_p_now)
        t_hist.append(t)

        sigma = sigma + stt.sigma_value(sigma, t, obstacles, stt_params) * dt
        mujoco.mj_step(model, data)

        if verbose_every and i % verbose_every == 0:
            print(f"t={t:6.3f}  p={p_true}  theta_e={diag['theta_e']:.4f}  e_p={e_p_hist[-1]:.4f}")

    return dict(t=np.array(t_hist), theta_e=np.array(theta_e_hist),
               e_p=np.array(e_p_hist), diverged=diverged)


if __name__ == "__main__":
    print("Running headless MuJoCo validation on the Crazyflie 2 WITH "
         "motor/ESC lag, sensor noise, and comm delay "
         f"(RELAXED tube: rho_max={RHO_MAX}, tc={TC} -- see module docstring)...")
    res = run()
    nfail = 0

    def check(name, cond, extra=""):
        global nfail
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            nfail += 1

    check("no NaN/divergence over the full run", not res["diverged"])
    if not res["diverged"]:
        print(f"e_p: max over run = {res['e_p'].max():.4f}")
        print(f"theta_e: mean(last 2s) = {res['theta_e'][-1000:].mean():.4f}")
        check("position stays strictly inside the tube (e_p<1)",
             np.all(res["e_p"] < 1.0), f"(max={res['e_p'].max():.4f})")

    print(f"\n{'ALL CHECKS PASSED' if nfail == 0 else f'{nfail} CHECK(S) FAILED'}")
    sys.exit(0 if nfail == 0 else 1)
