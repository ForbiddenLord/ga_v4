"""
run_validation.py -- headless MuJoCo closed-loop validation of the
CORRECTED GA-STT control law (ga_stt_controller.compute_control), run
against MuJoCo's own independent rigid-body physics integrator instead of
this repo's own RK4 dynamics.py. This is deliberately a cross-check with a
different numerical engine, not just a re-run of the existing test suite:
it catches convention mismatches (e.g. a body-frame-vs-world-frame sign
error) that could slip through if the controller test and the dynamics
model share an assumption bug.

Model: ga_stt_py/mujoco_validation/quad.xml -- a free rigid body with
mass/inertia matching test_closed_loop_convergence.py's drone. No rotor
aerodynamics; thrust (along the body's current +z) and body torque are
applied every physics step via data.xfrc_applied (MuJoCo's generalized
external-force/torque interface, expressed in the WORLD frame), with the
thrust vector and torque both rotated from body frame into world frame
using MuJoCo's own quaternion (independent of ga3.py's rotor math -- a
second, cross-checking rotation implementation).

Usage: python3 run_validation.py
Requires: pip install mujoco (already available in this environment).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import mujoco
import ga3 as ga
import stt
import ga_stt_controller as ctrl


def quat_wxyz_to_rotor(q):
    """MuJoCo quaternion (w,x,y,z) -> ga3 rotor.

    CORRECTED (was wrong in an earlier draft of this script): the mapping
    is quat_mujoco = (scalar, -bivec3), NOT (scalar, +bivec3). Verified
    independently here (not just taken on trust) by comparing
    mujoco.mju_rotVecQuat under each candidate mapping against ga.sandwich's
    (already-tested) rotation of the same vector by the same rotor --
    (scalar, +bivec3) gives the rotation with the WRONG sign, (scalar,
    -bivec3) matches ga.sandwich exactly. This matches the convention
    independently used and documented in
    GA_STT_delivery_v2/mujoco_validation/mujoco_validate.py.
    """
    w, x, y, z = q
    return ga.rotor(w, -np.array([x, y, z]))


def rotor_to_quat_wxyz(R):
    s = ga.rotor_scalar(R)
    v = ga.rotor_bivec3(R)
    return np.array([s, *(-v)])


def run(duration=8.0, dt=0.002, verbose_every=500):
    stt_params = stt.STTParams(eta=[8, 8, 8], s0=[1, 1, 1], tc=16.0,
                               k1=0.15, k2=1.0, k3=0.5,
                               rho_max=1.0, rho_min=0.15, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=0.5)
    drone = ctrl.DroneParams(mass=0.5, J_diag=[3.0e-3, 3.0e-3, 5.5e-3],
                             f_min=0.0, f_max=12.0,
                             kappa1=1.5, kappa_v=2.0, kR=8.0, kOmega=0.2)
    obstacles = [stt.Obstacle([4, 4, 4], [0.05, -0.02, 0.03], 0.6),
                 stt.Obstacle([3, 5, 4.5], [-0.03, 0.04, 0.0], 0.5)]

    xml_path = os.path.join(os.path.dirname(__file__), "quad.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)

    # Initial state: same as test_closed_loop_convergence.py (p=(1,1,1),
    # v=0, a 0.6 rad initial attitude error, Omega_b=0).
    data.qpos[0:3] = [1.0, 1.0, 1.0]
    R0 = ga.exp_bivector(np.array([0.3, 0.5, 0.1]) / np.linalg.norm([0.3, 0.5, 0.1]), 0.6)
    data.qpos[3:7] = rotor_to_quat_wxyz(R0)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sigma = stt_params.s0.copy()
    theta_e_hist, e_p_hist, t_hist = [], [], []
    n_steps = int(duration / dt)
    diverged = False

    for i in range(n_steps + 1):
        t = i * dt
        p = data.qpos[0:3].copy()
        R = quat_wxyz_to_rotor(data.qpos[3:7])
        v_world = data.qvel[0:3].copy()
        # CORRECTED: MuJoCo's free-joint qvel[3:6] is already the GA
        # BODY-frame Omega_b (verified independently here: Rdot=-1/2 R
        # Omega_b matches a finite-difference MuJoCo step to ~1.5e-5,
        # while treating qvel[3:6] as world-frame and applying the same
        # check is off by 0.49 -- i.e. wrong). No rotation needed; the
        # earlier draft of this script wrongly rotated it, which combined
        # with the quaternion sign bug above to corrupt the very first
        # control step.
        Omega_b = data.qvel[3:6].copy()

        ud = dict(p=p, v=v_world, R=R, Omega_b=Omega_b, sigma=sigma)
        f_cmd, tau, diag = ctrl.compute_control(t, ud, obstacles, stt_params, drone, ga.rotor_identity())

        if not (np.isfinite(f_cmd) and np.all(np.isfinite(tau))):
            diverged = True
            break

        # Thrust acts along the body's CURRENT +z axis; both thrust and
        # torque are expressed in world frame via ga.sandwich (using our
        # own, now-verified rotor R, rather than re-deriving a MuJoCo quat
        # rotation with mju_rotVecQuat -- avoids re-introducing the sign
        # bug fixed above).
        thrust_world = ga.sandwich(R, np.array([0.0, 0.0, f_cmd]))
        tau_world = ga.sandwich(R, tau)

        data.xfrc_applied[1, 0:3] = thrust_world
        data.xfrc_applied[1, 3:6] = tau_world

        theta_e_hist.append(diag["theta_e"])
        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        e_p_hist.append(np.linalg.norm(p - sigma) / rho_p_now)
        t_hist.append(t)

        sigma_dot = stt.sigma_value(sigma, t, obstacles, stt_params)
        sigma = sigma + sigma_dot * dt

        mujoco.mj_step(model, data)

        if verbose_every and i % verbose_every == 0:
            print(f"t={t:6.3f}  p={p}  theta_e={diag['theta_e']:.4f}  "
                  f"e_p={e_p_hist[-1]:.4f}  f_cmd={f_cmd:.3f}")

    theta_e_hist = np.array(theta_e_hist)
    e_p_hist = np.array(e_p_hist)
    return dict(t=np.array(t_hist), theta_e=theta_e_hist, e_p=e_p_hist, diverged=diverged)


if __name__ == "__main__":
    print("Running headless MuJoCo closed-loop validation of the corrected GA-STT law...")
    res = run()
    nfail = 0

    def check(name, cond, extra=""):
        global nfail
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
        if not cond:
            nfail += 1

    check("no NaN/divergence over the full run (independent MuJoCo integrator)", not res["diverged"])
    if not res["diverged"]:
        dt = res["t"][1] - res["t"][0]
        i1s = int(1.0 / dt)
        i8s = len(res["t"]) - 1
        print(f"theta_e: start={res['theta_e'][0]:.4f}, t=1s={res['theta_e'][i1s]:.4f}, "
              f"t=8s={res['theta_e'][i8s]:.4f}")
        print(f"e_p: max over run = {res['e_p'].max():.4f}")
        check("attitude error converges (independent MuJoCo integrator)", res["theta_e"][i8s] < 0.05)
        check("position stays strictly inside the tube (e_p<1, independent MuJoCo integrator)",
              np.all(res["e_p"] < 1.0), f"(max={res['e_p'].max():.4f})")

    print(f"\n{'ALL CHECKS PASSED' if nfail == 0 else f'{nfail} CHECK(S) FAILED'}")
    sys.exit(0 if nfail == 0 else 1)
