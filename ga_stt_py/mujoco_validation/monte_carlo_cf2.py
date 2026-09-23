"""
monte_carlo_cf2.py -- Monte Carlo scaling study of the FINAL GA-STT
controller against the REAL Crazyflie 2 MuJoCo model, for YOU to run
locally (this was intentionally not run in the delivery session -- batch
runs of many trials are exactly the kind of workload that should happen
on your own machine, not be burned as tokens one trial at a time).

What this does, per trial:
  1. Randomly places n_obstacles spheres in a bounded region between a
     random start and the fixed target, with randomized (small) constant
     velocities -- same generative idea as the STT paper's own dynamic-
     obstacle scenarios, not a fixed obstacle field.
  2. Runs run_cf2_validation.py's run(), but with THIS trial's start/
     obstacles substituted in, against the real Crazyflie 2 MuJoCo model.
  3. Records: converged (bool, no NaN), max e_p (tube-safety margin, <1
     means never left the tube), final theta_e, whether thrust saturated
     for more than a threshold fraction of the run (a proxy for "the
     vehicle was fighting its actuator limit", the failure mode flagged
     as a possible root cause of tube violations at high obstacle density
     in an earlier Monte Carlo study -- see the top-level status doc).

Usage:
    python3 monte_carlo_cf2.py --n-obstacles 5 --trials 20 --out results_n5.csv
    python3 monte_carlo_cf2.py --n-obstacles 15 --trials 20 --out results_n15.csv
    python3 monte_carlo_cf2.py --n-obstacles 25 --trials 20 --out results_n25.csv

Each trial with a fresh MuJoCo model load + an 8s sim at dt=0.002 (4000
steps) takes on the order of a few seconds on a modern laptop CPU; a
20-trial sweep should take low-single-digit minutes. Scale --trials up
once you've confirmed it runs cleanly.
"""
import sys, os, csv, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import mujoco
import ga3 as ga
import stt
import ga_stt_controller_final as ctrl
from run_cf2_validation import quat_wxyz_to_rotor, rotor_to_quat_wxyz


def make_trial(rng, n_obstacles, arena_half_size=1.0):
    """Random start, fixed target, n_obstacles random dynamic spheres
    scattered between them (rejected/resampled if they overlap the start
    or target by more than a small margin, so every trial is at least
    nominally feasible)."""
    start = rng.uniform([-0.2, -0.2, 0.3], [0.2, 0.2, 0.5])
    target = np.array([1.2, 1.2, 1.0])

    obstacles = []
    tries = 0
    while len(obstacles) < n_obstacles and tries < n_obstacles * 20:
        tries += 1
        pos = rng.uniform([-0.1, -0.1, 0.2], [1.4, 1.4, 1.2])
        if np.linalg.norm(pos - start) < 0.25 or np.linalg.norm(pos - target) < 0.25:
            continue
        vel = rng.uniform(-0.05, 0.05, size=3)
        vel[2] *= 0.2  # obstacles drift mostly horizontally
        radius = rng.uniform(0.06, 0.14)
        obstacles.append(stt.Obstacle(pos, vel, radius))
    return start, target, obstacles


def run_trial(seed, n_obstacles, drone, duration=8.0, dt=0.002, xml_path=None):
    rng = np.random.default_rng(seed)
    start, target, obstacles = make_trial(rng, n_obstacles)

    stt_params = stt.STTParams(eta=target, s0=start, tc=duration,
                               k1=0.2, k2=1.0, k3=0.5, rho_max=0.30, rho_min=0.06, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=1.0)

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    cf2_body_id = model.body("cf2").id

    data.qpos[0:3] = start
    data.qpos[3:7] = rotor_to_quat_wxyz(ga.rotor_identity())
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    sigma = start.copy()
    max_ep, n_saturated, n_steps = 0.0, 0, int(duration / dt)
    diverged = False
    theta_e_final = None

    for i in range(n_steps + 1):
        t = i * dt
        p = data.xpos[cf2_body_id].copy()
        R = quat_wxyz_to_rotor(data.xquat[cf2_body_id])
        v_world = data.cvel[cf2_body_id][3:6].copy()
        Omega_b = data.cvel[cf2_body_id][0:3].copy()

        ud = dict(p=p, v=v_world, R=R, Omega_b=Omega_b, sigma=sigma)
        f_cmd, tau, diag = ctrl.compute_control(t, ud, obstacles, stt_params, drone, ga.rotor_identity())
        if not (np.isfinite(f_cmd) and np.all(np.isfinite(tau))):
            diverged = True
            break

        thrust_world = ga.sandwich(R, np.array([0.0, 0.0, f_cmd]))
        tau_world = ga.sandwich(R, tau)
        data.xfrc_applied[cf2_body_id, 0:3] = thrust_world
        data.xfrc_applied[cf2_body_id, 3:6] = tau_world

        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        ep = np.linalg.norm(p - sigma) / rho_p_now
        max_ep = max(max_ep, ep)
        if f_cmd >= 0.99 * drone.f_max:
            n_saturated += 1
        theta_e_final = diag["theta_e"]

        sigma_dot = stt.sigma_value(sigma, t, obstacles, stt_params)
        sigma = sigma + sigma_dot * dt
        mujoco.mj_step(model, data)

    return dict(seed=seed, n_obstacles=n_obstacles, diverged=diverged,
               max_ep=max_ep, tube_safe=(not diverged) and max_ep < 1.0,
               theta_e_final=theta_e_final, frac_saturated=n_saturated / n_steps,
               start=start.tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obstacles", type=int, default=5)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", type=str, default="monte_carlo_results.csv")
    args = ap.parse_args()

    xml_path = os.path.join(os.path.dirname(__file__), "crazyflie_menagerie", "cf_scene.xml")
    drone = ctrl.DroneParams(mass=0.027, J_diag=[2.3951e-5, 2.3951e-5, 3.2347e-5],
                             f_min=0.0, f_max=0.60,
                             kappa1=3.0, kappa_v=6.0, kR=25.0, kOmega=1.5e-3,
                             tau_max=0.012)

    rows = []
    t0 = time.time()
    for k in range(args.trials):
        seed = args.seed0 + k
        res = run_trial(seed, args.n_obstacles, drone, xml_path=xml_path)
        rows.append(res)
        print(f"[{k+1}/{args.trials}] seed={seed} n_obs={args.n_obstacles} "
             f"diverged={res['diverged']} max_ep={res['max_ep']:.3f} "
             f"tube_safe={res['tube_safe']} frac_saturated={res['frac_saturated']:.3f}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_safe = sum(r["tube_safe"] for r in rows)
    print(f"\n{n_safe}/{len(rows)} trials tube-safe "
         f"({100*n_safe/len(rows):.1f}%) for n_obstacles={args.n_obstacles}")
    print(f"Elapsed: {time.time()-t0:.1f}s. Results written to {args.out}")
