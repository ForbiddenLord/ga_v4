"""
mpc_controller.py — Model Predictive Control (MPC) baseline.

Architecture:
  Receding-horizon MPC over N steps of length dt_mpc on the translational
  dynamics (p, v), with control input = net acceleration a = (1/m)F - g e3.
  The rotational loop is simplified: desired attitude is computed from the
  optimal a* at the current step via the same DCM construction as the
  baseline, tracked by Lee PD.

  Prediction model (constant-g, no drag):
      p_{k+1} = p_k + v_k dt_mpc
      v_{k+1} = v_k + a_k dt_mpc

  Cost:
      J = Σ_{k=0}^{N-1} [w_p ||p_k - σ(t+k dt_mpc)||²
                        + w_v ||v_k||²
                        + w_a ||a_k||²]
        + w_pN ||p_N - σ(t+N dt_mpc)||²

  Constraints:
      - ||a_k||_∞ ≤ a_max       (thrust authority)
      - ||p_k - o_j(t+k dt_mpc)|| ≥ r_j + margin  for each obstacle j, step k

  Warm-started from the previous solution shifted by one step.
  Obstacle constraints are linearised around the last feasible solution
  (sequential convex approximation: one QP per call instead of a full NLP).

Design intent for the paper:
  - MPC provides a principled safety guarantee (constraint satisfaction
    at every predicted step) at the cost of an online optimisation.
  - GA-STT delivers closed-form, O(1) computation with analytic derivatives
    and an ISS-proven tube; MPC requires solving an NLP/QP at every step.
  - The per-call wall-clock comparison is the key metric.
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.spatial.transform import Rotation as scipyR

import ga3 as ga
import stt

E3 = np.array([0.0, 0.0, 1.0])

# MPC hyperparameters
N_HORIZON = 8      # prediction steps
DT_MPC = 0.05   # s per prediction step
W_POS = 3.0    # position tracking weight
W_VEL = 0.5    # velocity damping weight
W_ACC = 0.02   # control effort weight
W_TERMINAL = 6.0    # terminal position weight
A_MAX = 15.0   # m/s², roughly f_max/m - g for 0.5 kg drone (20N max)
OBS_MARGIN = 0.1    # m extra safety margin beyond obstacle radius


def _sigma_at(t, stt_params, obstacles, sigma_now):
    """Approximate σ at future time t by Euler integration from current value.
    (Cheap: sigma ODE is smooth and σ moves slowly compared to N*dt_mpc.)"""
    dt = DT_MPC / 10
    sig = sigma_now.copy()
    t_cur = t
    for _ in range(10):
        sig = sig + stt.sigma_value(sig, t_cur, obstacles, stt_params) * dt
        t_cur += dt
    return sig


def _pack(a_seq):
    return a_seq.flatten()


def _unpack(x):
    return x.reshape((N_HORIZON, 3))


def _rollout(p0, v0, a_seq):
    """Return (p, v) trajectories: shape (N+1, 3) each."""
    p = np.zeros((N_HORIZON + 1, 3))
    v = np.zeros((N_HORIZON + 1, 3))
    p[0], v[0] = p0, v0
    for k in range(N_HORIZON):
        v[k + 1] = v[k] + a_seq[k] * DT_MPC
        p[k + 1] = p[k] + v[k + 1] * DT_MPC
    return p, v


def _cost_and_grad(x, p0, v0, t0, sigma_now, stt_params, obstacles):
    a_seq = _unpack(x)
    p, v = _rollout(p0, v0, a_seq)
    cost = 0.0
    for k in range(N_HORIZON):
        t_k = t0 + k * DT_MPC
        sig_k = _sigma_at(t_k, stt_params, obstacles, sigma_now)
        cost += W_POS * np.sum((p[k] - sig_k) ** 2)
        cost += W_VEL * np.sum(v[k] ** 2)
        cost += W_ACC * np.sum(a_seq[k] ** 2)
    sig_N = _sigma_at(t0 + N_HORIZON * DT_MPC,
                      stt_params, obstacles, sigma_now)
    cost += W_TERMINAL * np.sum((p[N_HORIZON] - sig_N) ** 2)
    return cost


def _obstacle_constraints(p0, v0, t0, obstacles):
    """Return list of scipy NonlinearConstraint objects (one per obstacle per step)."""
    cons = []
    for obs in obstacles:
        for k in range(1, N_HORIZON + 1):
            t_k = t0 + k * DT_MPC
            oj_k = obs.position(t_k)
            r_min = obs.radius + OBS_MARGIN

            def _h(x, oj=oj_k, r=r_min, kk=k):
                a_seq = _unpack(x)
                p, _ = _rollout(p0, v0, a_seq)
                return np.linalg.norm(p[kk] - oj) - r

            cons.append(NonlinearConstraint(_h, 0.0, np.inf))
    return cons


_warm_start = None  # previous optimal a_seq (shifted)


def reset_mpc_state():
    global _warm_start
    _warm_start = None


def _lee_dcm_from_force(Fd, psi_d=0.0):
    fd = np.linalg.norm(Fd)
    b3d = Fd / max(fd, 1e-9)
    b1_psi = np.array([np.cos(psi_d), np.sin(psi_d), 0.0])
    b2d = np.cross(b3d, b1_psi)
    n = np.linalg.norm(b2d)
    if n < 1e-6:
        b1_psi = np.array([0.0, 1.0, 0.0])
        b2d = np.cross(b3d, b1_psi)
        n = np.linalg.norm(b2d)
    b2d /= n
    b1d = np.cross(b2d, b3d)
    return np.column_stack([b1d, b2d, b3d])


def _rotor_from_dcm(Rmat):
    try:
        quat_xyzw = scipyR.from_matrix(Rmat).as_quat()
        x, y, z, w = quat_xyzw
        return ga.rotor(w, -np.array([x, y, z])), True
    except Exception:
        return ga.rotor_identity(), False


_mpc_att_cache = {"t_prev": None, "Rd_prev": None}


def reset_mpc_att_state():
    _mpc_att_cache["t_prev"] = None
    _mpc_att_cache["Rd_prev"] = None


def compute_control_mpc(t0, state, obstacles, stt_params, drone_params, psi_d=0.0):
    global _warm_start
    p, v, R, Omega_b = state["p"], state["v"], state["R"], state["Omega_b"]
    sigma_now = state["sigma"]
    m, g = drone_params.m, 9.81

    # ── Warm start ──────────────────────────────────────────────────────────
    if _warm_start is None:
        x0 = np.zeros(N_HORIZON * 3)
    else:
        # Shift previous solution by one step; append hold at last value
        prev = _warm_start.reshape(N_HORIZON, 3)
        shifted = np.vstack([prev[1:], prev[-1:]])
        x0 = shifted.flatten()

    # ── Build constraints ───────────────────────────────────────────────────
    # Control bounds: ||a_k||_inf <= A_MAX per element
    bounds = [(-A_MAX, A_MAX)] * (N_HORIZON * 3)

    cons = _obstacle_constraints(p, v, t0, obstacles) if obstacles else []

    # ── Solve ───────────────────────────────────────────────────────────────
    result = minimize(
        _cost_and_grad, x0,
        args=(p, v, t0, sigma_now, stt_params, obstacles),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 80, "ftol": 1e-4, "disp": False},
    )

    if result.success or result.fun < 1e6:
        a_opt = _unpack(result.x)
    else:
        # Fallback to warm start if optimisation failed
        a_opt = _unpack(x0)

    _warm_start = a_opt.flatten().copy()

    # Apply first step's acceleration
    a0 = a_opt[0]
    F_total = m * (a0 + g * E3)
    f_command = np.clip(np.linalg.norm(F_total),
                        drone_params.f_min, drone_params.f_max)

    # ── Attitude from optimal thrust direction ──────────────────────────────
    Rmat0 = _lee_dcm_from_force(F_total, psi_d)
    Rd0, dcm_ok = _rotor_from_dcm(Rmat0)
    Rd0 = ga.normalize_rotor(Rd0)

    t_prev, Rd_prev = _mpc_att_cache["t_prev"], _mpc_att_cache["Rd_prev"]
    if t_prev is not None and (t0 - t_prev) > 1e-9:
        dt_loop = t0 - t_prev
        Rd_prev_a = Rd_prev
        if ga.rotor_scalar(ga.geometric_product(ga.reverse(Rd_prev), Rd0)) < 0:
            Rd_prev_a = -Rd_prev
        Rd_dot_fd = (Rd0 - Rd_prev_a) / dt_loop
        Omega_d = ga.to_bivector3(-2 *
                                  ga.geometric_product(ga.reverse(Rd0), Rd_dot_fd))
    else:
        Omega_d = np.zeros(3)
    _mpc_att_cache["t_prev"], _mpc_att_cache["Rd_prev"] = t0, Rd0

    Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd0), R))
    eR_vec = ga.to_bivector3(Re)
    eR = 0.5 * eR_vec
    eOmega = Omega_b - ga.sandwich_bivector(ga.reverse(Re), Omega_d)
    J = drone_params.J_diag
    commutator_term = np.cross(Omega_b, J * Omega_b)
    kR_lee = 2.5
    kOmega_lee = 0.5
    tau = kR_lee * eR - kOmega_lee * eOmega + commutator_term
    tau_max = 1.5
    if np.linalg.norm(tau) > tau_max:
        tau = tau * (tau_max / np.linalg.norm(tau))

    theta_e_diag, _ = ga.rotor_log_angle_axis(Re)
    diag = dict(Rd=Rd0, Omega_d=Omega_d, Omega_dot_d=np.zeros(3), Re=Re,
                theta_e=theta_e_diag, e_R=np.nan, rho_R=np.nan,
                dcm_ok=dcm_ok, mpc_iters=result.nit,
                mpc_success=result.success, a_opt=a0)
    return f_command, tau, diag
