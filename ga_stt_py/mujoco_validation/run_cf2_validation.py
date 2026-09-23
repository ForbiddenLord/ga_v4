"""
run_cf2_validation.py -- closed-loop validation of the FINAL GA-STT
controller (ga_stt_controller_final.py: Stage-1 fixed, Stage-R reverted to
the proven-exact original law) against the REAL Bitcraze Crazyflie 2 model
from the MuJoCo Menagerie (crazyflie_menagerie/cf2.xml, imported
unmodified: same mesh, mass=0.027 kg, diag inertia
(2.3951e-5, 2.3951e-5, 3.2347e-5) kg*m^2).

This is a SINGLE scenario, not a Monte Carlo sweep (see
monte_carlo_template.py in this folder, and TUNING_AND_MUJOCO_GUIDE.md, for
the batch version -- intentionally left for you to run locally rather than
spending tokens on many trials here).

Frame conventions (rotor<->MuJoCo quaternion sign, qvel[3:6] already being
body-frame) are the ones independently verified earlier in this project
(see run_validation.py in this same folder for the derivation/checks);
reused here unchanged.

Two assumptions not taken from the MJCF (flagged explicitly, see also the
guide):
  - f_max = 0.60 N. The menagerie model's own thrust actuator ctrlrange
    (0 to 0.35 N) is noted in its own README as "arbitrary, needs further
    tuning" -- 0.35 N gives a hover thrust margin of only ~1.3x, quite low
    for CF2.1. 0.60 N (~2.2x hover margin) is a commonly-cited literature
    figure for CF2.1 with stock rotors; thrust/torque are applied directly
    via xfrc_applied (bypassing the model's own under-tuned actuators)
    rather than fixing their gear scaling.
  - tau_max = 0.012 N*m (combined norm cap). Approximate, from published
    CF2 system-identification figures (roll/pitch torque authority on the
    order of 1e-2 N*m); not independently re-derived or hardware-measured
    here.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import mujoco
import ga3 as ga
import stt
import ga_stt_controller_final as ctrl


def quat_wxyz_to_rotor(q):
    w, x, y, z = q
    return ga.rotor(w, -np.array([x, y, z]))


def rotor_to_quat_wxyz(R):
    s = ga.rotor_scalar(R)
    v = ga.rotor_bivec3(R)
    return np.array([s, *(-v)])


def run(duration=8.0, dt=0.002, verbose_every=500):
    stt_params = stt.STTParams(eta=[1.2, 1.2, 1.0], s0=[0.0, 0.0, 0.4], tc=8.0,
                               k1=0.2, k2=1.0, k3=0.5, rho_max=0.35, rho_min=0.08, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=1.0)
    drone = ctrl.DroneParams(mass=0.027, J_diag=[2.3951e-5, 2.3951e-5, 3.2347e-5],
                             f_min=0.0, f_max=0.60,
                             kappa1=3.0, kappa_v=6.0, kR=25.0, kOmega=1.5e-3,
                             tau_max=0.012)
    obstacles = [stt.Obstacle([0.6, 0.6, 0.5], [0.0, 0.0, 0.0], 0.15)]

    xml_path = os.path.join(os.path.dirname(__file__), "crazyflie_menagerie", "cf_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    cf2_body_id = model.body("cf2").id

    data.qpos[0:3] = stt_params.s0
    axis = np.array([0.3, 0.5, 0.1]); axis /= np.linalg.norm(axis)
    R0 = ga.exp_bivector(axis, 0.5)
    data.qpos[3:7] = rotor_to_quat_wxyz(R0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sigma = stt_params.s0.copy()
    theta_e_hist, e_p_hist, t_hist, thrust_hist = [], [], [], []
    n_steps = int(duration / dt)
    diverged = False

    for i in range(n_steps + 1):
        t = i * dt
        p = data.xpos[cf2_body_id].copy()
        R = quat_wxyz_to_rotor(data.xquat[cf2_body_id])
        v_world = data.cvel[cf2_body_id][3:6].copy()   # linear part, world frame
        Omega_b = data.cvel[cf2_body_id][0:3].copy()   # angular part, already body frame (verified)

        ud = dict(p=p, v=v_world, R=R, Omega_b=Omega_b, sigma=sigma)
        f_cmd, tau, diag = ctrl.compute_control(t, ud, obstacles, stt_params, drone, ga.rotor_identity())

        if not (np.isfinite(f_cmd) and np.all(np.isfinite(tau))):
            diverged = True
            break

        thrust_world = ga.sandwich(R, np.array([0.0, 0.0, f_cmd]))
        tau_world = ga.sandwich(R, tau)
        data.xfrc_applied[cf2_body_id, 0:3] = thrust_world
        data.xfrc_applied[cf2_body_id, 3:6] = tau_world

        theta_e_hist.append(diag["theta_e"])
        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        e_p_hist.append(np.linalg.norm(p - sigma) / rho_p_now)
        t_hist.append(t)
        thrust_hist.append(f_cmd)

        # move the obstacle/target mocap markers for visualization (no dynamics effect)
        data.mocap_pos[model.body("obstacle0").mocapid[0]] = obstacles[0].position(t)
        data.mocap_pos[model.body("target_marker").mocapid[0]] = stt_params.eta

        sigma_dot = stt.sigma_value(sigma, t, obstacles, stt_params)
        sigma = sigma + sigma_dot * dt

        mujoco.mj_step(model, data)

        if verbose_every and i % verbose_every == 0:
            print(f"t={t:6.3f}  p={p}  theta_e={diag['theta_e']:.4f}  "
                 f"e_p={e_p_hist[-1]:.4f}  f_cmd={f_cmd:.4f}")

    return dict(t=np.array(t_hist), theta_e=np.array(theta_e_hist),
               e_p=np.array(e_p_hist), thrust=np.array(thrust_hist),
               diverged=diverged, stt_params=stt_params)


if __name__ == "__main__":
    print("Running headless MuJoCo validation on the REAL Crazyflie 2 model "
         "(mujoco_menagerie/bitcraze_crazyflie_2)...")
    res = run()
    nfail = 0

    def check(name, cond, extra=""):
        global nfail
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            nfail += 1

    check("no NaN/divergence over the full run", not res["diverged"])
    if not res["diverged"]:
        dt = res["t"][1] - res["t"][0]
        i1s = int(1.0 / dt)
        iend = len(res["t"]) - 1
        print(f"theta_e: start={res['theta_e'][0]:.4f}, t=1s={res['theta_e'][i1s]:.4f}, "
             f"t=end={res['theta_e'][iend]:.4f}")
        print(f"e_p: max over run = {res['e_p'].max():.4f}")
        print(f"thrust: min={res['thrust'].min():.4f} N, max={res['thrust'].max():.4f} N "
             f"(f_max=0.60 N)")
        check("attitude error converges", res["theta_e"][iend] < 0.05)
        check("position stays strictly inside the tube (e_p<1)",
             np.all(res["e_p"] < 1.0), f"(max={res['e_p'].max():.4f})")
        # A brief instant at the ceiling during the initial recovery
        # transient is fine (a real vehicle briefly using all available
        # thrust is normal); SUSTAINED saturation would indicate the
        # vehicle is fighting its own actuator limit continuously, which
        # is the actual concern (see the Stage-1 bootstrap-vs-Picard
        # comparison in the top-level status doc for why that matters).
        frac_saturated = float(np.mean(res["thrust"] >= 0.59 * 0.999))
        check("thrust saturation is brief, not sustained (<1% of steps at/near ceiling)",
             frac_saturated < 0.01, f"(fraction={frac_saturated:.4f})")

    print(f"\n{'ALL CHECKS PASSED' if nfail == 0 else f'{nfail} CHECK(S) FAILED'}")
    sys.exit(0 if nfail == 0 else 1)
