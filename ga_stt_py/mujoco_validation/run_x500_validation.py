"""
run_x500_validation.py -- closed-loop validation of the FINAL GA-STT
controller against a Holybro X500 quadrotor, physical parameters
transcribed from the PX4 Gazebo SDF you provided
(Tools/simulation/gz/models/x500_base/model.sdf):
    base_link: mass=2.0 kg, diag inertia (0.021667, 0.021667, 0.04) kg*m^2
    4 rotors: mass=0.016077 kg each, at (+-0.174, +-0.174, 0.06) m
Total (parallel-axis corrected, cross-checked against MuJoCo's own
composite rigid-body dynamics to 4 significant figures -- see the
delivery chat): mass=2.0643 kg,
diag inertia (0.023845, 0.023845, 0.043894) kg*m^2.

MuJoCo doesn't have a native Crazyflie-style ready-made X500 asset the
way the Menagerie does for the CF2, and the SDF you gave me references
mesh files (.dae/.stl) that weren't included -- only the SDF's XML text
was. x500/x500.xml is therefore a PRIMITIVE-GEOMETRY reconstruction
(box body + cylinder arms/rotors), not a mesh import: the physically
relevant numbers (mass, per-body inertia, rotor positions) are exact
transcriptions from your SDF; the visual shapes are not. See x500/x500.xml
for the full account and the cross-check against MuJoCo's own dynamics.

This is the IDEAL-ACTUATOR baseline (matches the style of
run_cf2_validation.py): no motor lag, no sensor noise, no comm delay.
See run_x500_validation_realistic.py for those added.

Assumptions not in the SDF (SDF gives geometry/mass, not
thrust/torque limits):
  - f_max = 40.0 N (~2:1 thrust-to-weight for a 2.06 kg vehicle -- a
    commonly-cited range for X500-class builds with 5010/920KV motors
    and 10-13 inch props; not hardware-measured here).
  - tau_max = 2.5 N*m (combined norm cap; order-of-magnitude estimate
    from arm length (0.246 m) x plausible differential thrust, not
    hardware-measured or system-ID'd).
Gains (kappa1=0.8, kappa_v=4.5, kR=36.0, kOmega=0.5) were found with
tune_gains.py; see the delivery chat for the exact command.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import mujoco
import ga3 as ga
import stt
import ga_stt_controller_final as ctrl
from run_cf2_validation import quat_wxyz_to_rotor, rotor_to_quat_wxyz


def make_scenario():
    stt_params = stt.STTParams(eta=[3.0, 3.0, 2.0], s0=[0.0, 0.0, 1.0], tc=10.0,
                               k1=0.2, k2=1.0, k3=0.5, rho_max=0.8, rho_min=0.15, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=0.8)
    drone = ctrl.DroneParams(mass=2.0643, J_diag=[0.02384515, 0.02384515, 0.04389396],
                             f_min=0.0, f_max=40.0,
                             kappa1=0.8, kappa_v=4.5, kR=36.0, kOmega=0.5,
                             tau_max=2.5)
    obstacles = [stt.Obstacle([1.5, 1.5, 1.0], [0.0, 0.0, 0.0], 0.3)]
    return stt_params, drone, obstacles


def run(duration=10.0, dt=0.002, verbose_every=500):
    stt_params, drone, obstacles = make_scenario()

    xml_path = os.path.join(os.path.dirname(__file__), "x500", "x500.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    x500_id = model.body("x500").id

    data.qpos[0:3] = stt_params.s0
    axis = np.array([0.3, 0.5, 0.1]); axis /= np.linalg.norm(axis)
    R0 = ga.exp_bivector(axis, 0.4)
    data.qpos[3:7] = rotor_to_quat_wxyz(R0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sigma = stt_params.s0.copy()
    theta_e_hist, e_p_hist, t_hist, thrust_hist = [], [], [], []
    n_steps = int(duration / dt)
    diverged = False

    for i in range(n_steps + 1):
        t = i * dt
        p = data.xpos[x500_id].copy()
        R = quat_wxyz_to_rotor(data.xquat[x500_id])
        v_world = data.cvel[x500_id][3:6].copy()
        Omega_b = data.cvel[x500_id][0:3].copy()

        ud = dict(p=p, v=v_world, R=R, Omega_b=Omega_b, sigma=sigma)
        f_cmd, tau, diag = ctrl.compute_control(t, ud, obstacles, stt_params, drone, ga.rotor_identity())
        if not (np.isfinite(f_cmd) and np.all(np.isfinite(tau))):
            diverged = True
            break

        data.xfrc_applied[x500_id, 0:3] = ga.sandwich(R, np.array([0.0, 0.0, f_cmd]))
        data.xfrc_applied[x500_id, 3:6] = ga.sandwich(R, tau)

        theta_e_hist.append(diag["theta_e"])
        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        e_p_hist.append(np.linalg.norm(p - sigma) / rho_p_now)
        t_hist.append(t)
        thrust_hist.append(f_cmd)

        data.mocap_pos[model.body("obstacle0").mocapid[0]] = obstacles[0].position(t)
        data.mocap_pos[model.body("target_marker").mocapid[0]] = stt_params.eta

        sigma = sigma + stt.sigma_value(sigma, t, obstacles, stt_params) * dt
        mujoco.mj_step(model, data)

        if verbose_every and i % verbose_every == 0:
            print(f"t={t:6.3f}  p={p}  theta_e={diag['theta_e']:.4f}  "
                 f"e_p={e_p_hist[-1]:.4f}  f_cmd={f_cmd:.3f}")

    return dict(t=np.array(t_hist), theta_e=np.array(theta_e_hist),
               e_p=np.array(e_p_hist), thrust=np.array(thrust_hist), diverged=diverged)


if __name__ == "__main__":
    print("Running headless MuJoCo validation on the Holybro X500 "
         "(from your uploaded PX4 Gazebo SDF, ideal actuators)...")
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
        print(f"thrust: min={res['thrust'].min():.4f} N, max={res['thrust'].max():.4f} N (f_max=40.0 N)")
        check("attitude error converges", res["theta_e"][iend] < 0.05)
        check("position stays strictly inside the tube (e_p<1)",
             np.all(res["e_p"] < 1.0), f"(max={res['e_p'].max():.4f})")

    print(f"\n{'ALL CHECKS PASSED' if nfail == 0 else f'{nfail} CHECK(S) FAILED'}")
    sys.exit(0 if nfail == 0 else 1)
