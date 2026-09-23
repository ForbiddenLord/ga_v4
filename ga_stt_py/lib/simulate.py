"""simulate.py — closed-loop simulation harness."""
import numpy as np
import ga3 as ga
import stt
import dynamics as dyn


def run_simulation(controller_fn, stt_params, drone_params, obstacles,
                    p0, R0, t_end, dt=0.005, disturbance_fn=None,
                    controller_kwargs=None, sensor_noise_fn=None):
    """
    sensor_noise_fn(state, rng) -> dict with any of {'p','v','R','Omega_b'}
    overridden. If provided, the controller sees a NOISY copy of the state
    (mimicking real EKF/gyro measurement error), while the physics (RK4
    integration below) always advances using the TRUE, exact `state`.
    """
    if disturbance_fn is None:
        disturbance_fn = lambda t: (np.zeros(3), np.zeros(3))
    controller_kwargs = controller_kwargs or {}
    rng = np.random.default_rng(0)

    state = dyn.UAVState(p0, np.zeros(3), R0, np.zeros(3))
    sigma = stt_params.s0.copy()

    n_steps = int(t_end / dt)
    log = {k: [] for k in ["t", "p", "v", "sigma", "rho_p", "f", "tau",
                            "Rd", "Omega_d", "theta_e", "e_R", "Re",
                            "min_obstacle_clearance", "wall_time", "dcm_ok"]}

    import time as _time

    for i in range(n_steps + 1):
        t = i * dt

        ud = dict(p=state.p, v=state.v, R=state.R, Omega_b=state.Omega_b, sigma=sigma)
        if sensor_noise_fn is not None:
            ud.update(sensor_noise_fn(state, rng))   # overrides p/v/R/Omega_b only

        t_wall0 = _time.perf_counter()
        f_cmd, tau, diag = controller_fn(t, ud, obstacles, stt_params, drone_params,
                                          **controller_kwargs)
        t_wall1 = _time.perf_counter()

        rho_p_now = stt.rho_p_value(sigma, t, obstacles, stt_params)
        clearances = [np.linalg.norm(state.p - o.position(t)) - o.radius for o in obstacles]
        min_clear = min(clearances) if clearances else np.inf

        log["t"].append(t)
        log["p"].append(state.p.copy())
        log["v"].append(state.v.copy())
        log["sigma"].append(sigma.copy())
        log["rho_p"].append(rho_p_now)
        log["f"].append(f_cmd)
        log["tau"].append(tau.copy())
        log["Rd"].append(diag.get("Rd"))
        log["Omega_d"].append(diag.get("Omega_d"))
        log["theta_e"].append(diag.get("theta_e", np.nan))
        log["e_R"].append(diag.get("e_R", np.nan))
        log["Re"].append(diag.get("Re"))
        log["min_obstacle_clearance"].append(min_clear)
        log["wall_time"].append(t_wall1 - t_wall0)
        log["dcm_ok"].append(diag.get("dcm_ok", True))

        if i == n_steps:
            break

        # advance sigma (the STT's own internal reference) with the same dt
        sigma_dot = stt.sigma_value(sigma, t, obstacles, stt_params)
        sigma = sigma + sigma_dot * dt  # STT center is a designed reference, Euler is fine (smooth)

        wp, wtau = disturbance_fn(t)

        def cfn(t_local, s):
            return f_cmd, tau  # frozen-at-step control (standard zero-order hold over dt)

        new_state, _, _ = dyn.rk4_step(state, dt, drone_params.m, drone_params.J_diag,
                                        cfn, disturbance_fn, t)
        state = new_state

    for k in ["p", "v", "sigma", "tau", "Omega_d"]:
        log[k] = np.array(log[k])
    log["t"] = np.array(log["t"])
    log["f"] = np.array(log["f"])
    log["rho_p"] = np.array(log["rho_p"])
    log["theta_e"] = np.array(log["theta_e"])
    log["e_R"] = np.array(log["e_R"])
    log["min_obstacle_clearance"] = np.array(log["min_obstacle_clearance"])
    log["wall_time"] = np.array(log["wall_time"])
    log["dcm_ok"] = np.array(log["dcm_ok"])
    return log
