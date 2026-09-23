"""
run_x500_validation_realistic.py -- X500 with motor/ESC dynamics, sensor
noise, and communication delay, AND gains derived via a time-scale-
separation methodology rather than the ideal-actuator tune_gains.py
search. See TIME_SCALE_SEPARATION_ANALYSIS.md for the full derivation,
what was verified vs. corrected from the uploaded methodology document,
and -- importantly -- a negative result: THIS SCRIPT DOES NOT USE THE
SAME (aggressive) TUBE SCHEDULE AS run_x500_validation.py. Under 50ms
motor lag, no gain choice found (including the theoretically-motivated
ones) kept the original rho_max=0.8/tc=10.0 scenario inside the tube;
relaxing the schedule (rho_max=1.2, tc=15.0 -- a genuinely less
aggressive mission, not a way of hiding the problem) is what actually
restores a comfortable margin. Read the analysis doc before assuming this
script's success generalizes to a more aggressive mission profile.

GAINS: derived via the corrected time-scale-separation method (see the
analysis doc for the correction to the uploaded document's lambda_R^min
formula):
    tau_motor = 0.05 s  (assumption, see run_x500_validation.py docstring)
    tau_Omega = 3*tau_motor = 0.15 s  -> kappa_Omega = 0.1463
    tau_R     = 3*tau_Omega = 0.45 s  -> kappa_R     = 1.111
    kappa1, kappa_v: Stage 1 has no clean time constant (acknowledged in
    the analysis doc); kappa1=0.08 with kappa_v from the zeta=0.9
    heuristic was found, empirically, to sit inside the window bounded
    above by "slower than Stage R" and below by "fast enough to track
    the relaxed tube schedule."
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

ROTOR_XY = [(0.174, -0.174), (-0.174, 0.174), (0.174, 0.174), (-0.174, -0.174)]
ROTOR_SPIN = ['ccw', 'ccw', 'cw', 'cw']
K_M = 0.02
TAU_MOTOR = 0.05
COMM_DELAY_STEPS = 3

KAPPA1, KAPPA_V, KR, KOMEGA = 0.08, 0.849, 1.111, 0.1463
RHO_MAX, TC = 1.2, 15.0


def make_scenario():
    stt_params = stt.STTParams(eta=[3.0, 3.0, 2.0], s0=[0.0, 0.0, 1.0], tc=TC,
                               k1=0.2, k2=1.0, k3=0.5, rho_max=RHO_MAX, rho_min=0.15, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=0.8)
    drone = ctrl.DroneParams(mass=2.0643, J_diag=[0.02384515, 0.02384515, 0.04389396],
                             f_min=0.0, f_max=40.0,
                             kappa1=KAPPA1, kappa_v=KAPPA_V, kR=KR, kOmega=KOMEGA,
                             tau_max=2.5)
    obstacles = [stt.Obstacle([1.5, 1.5, 1.0], [0.0, 0.0, 0.0], 0.3)]
    return stt_params, drone, obstacles


def run(duration=14.0, dt=0.002, verbose_every=500, seed=0):
    stt_params, drone, obstacles = make_scenario()

    xml_path = os.path.join(os.path.dirname(__file__), "x500", "x500.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    x500_id = model.body("x500").id

    data.qpos[0:3] = stt_params.s0
    data.qpos[3:7] = rotor_to_quat_wxyz(ga.rotor_identity())
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    mixer = QuadMixer(ROTOR_XY, ROTOR_SPIN, K_M)
    motor_max = 1.6 * drone.f_max / 4.0
    motors = MotorModel(tau_motor=TAU_MOTOR, t_min=0.0, t_max=motor_max)
    motors.reset(mixer.wrench_to_motors(drone.m * 9.81, np.zeros(3)))
    noise = SensorNoise(gyro_noise_std=0.01, gyro_bias_walk_std=0.002,
                        pos_noise_std=0.03, vel_noise_std=0.05, seed=seed)
    delay = CommDelay(COMM_DELAY_STEPS, shape=(4,), init_value=[drone.m * 9.81, 0.0, 0.0, 0.0])

    sigma = stt_params.s0.copy()
    theta_e_hist, e_p_hist, t_hist = [], [], []
    n_steps = int(duration / dt)
    diverged = False

    for i in range(n_steps + 1):
        t = i * dt
        p_true = data.xpos[x500_id].copy()
        R = quat_wxyz_to_rotor(data.xquat[x500_id])
        v_true = data.cvel[x500_id][3:6].copy()
        Omega_true = data.cvel[x500_id][0:3].copy()

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

        data.xfrc_applied[x500_id, 0:3] = ga.sandwich(R, np.array([0.0, 0.0, f_actual]))
        data.xfrc_applied[x500_id, 3:6] = ga.sandwich(R, tau_actual)

        theta_e_hist.append(diag["theta_e"])
        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        e_p_hist.append(np.linalg.norm(p_true - sigma) / rho_p_now)
        t_hist.append(t)

        data.mocap_pos[model.body("obstacle0").mocapid[0]] = obstacles[0].position(t)
        data.mocap_pos[model.body("target_marker").mocapid[0]] = stt_params.eta

        sigma = sigma + stt.sigma_value(sigma, t, obstacles, stt_params) * dt
        mujoco.mj_step(model, data)

        if verbose_every and i % verbose_every == 0:
            print(f"t={t:6.3f}  p={p_true}  theta_e={diag['theta_e']:.4f}  e_p={e_p_hist[-1]:.4f}")

    return dict(t=np.array(t_hist), theta_e=np.array(theta_e_hist),
               e_p=np.array(e_p_hist), diverged=diverged)


if __name__ == "__main__":
    print("Running headless MuJoCo validation on the Holybro X500 WITH "
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
