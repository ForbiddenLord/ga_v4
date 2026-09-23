"""
cbf_controller.py — Control Barrier Function (CBF-QP) baseline.

Architecture:
  1. Compute nominal desired acceleration a_nom from a SIMPLE PD controller
     tracking σ(t) (the STT tube center). This is intentionally NOT the
     STT log-barrier law — the CBF adds obstacle avoidance on top of plain
     trajectory following, without the prescribed-time tube guarantee.
     This cleanly isolates the comparison: GA-STT provides tube invariance
     + obstacle avoidance + analytic attitude in closed form; CBF-QP only
     provides obstacle avoidance (no tube invariance, no analytic Ω_d).

  2. Solve a cvxpy QP to find the minimal δa from a_nom that satisfies
     second-order Extended CBF constraints for every active obstacle:
         h_j(p) = ‖p - o_j‖² - r_j²  ≥ 0
     Control constraint (ECBF, relative degree 2):
         2(p-o_j)ᵀ a  ≥  -2‖v-v_j‖²
                         -(α₁+α₂)·2(p-o_j)ᵀ(v-v_j)
                         -α₁α₂·(‖p-o_j‖²-r_j²)

  3. Recover thrust f = m‖a_safe + g e₃‖, set Rᵈ via classical (b3d,b1d,b2d)
     DCM, track Rᵈ with Lee PD + backward-FD Ω_d (same as baseline).
"""

import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation as scipyR

import ga3 as ga
import stt

E3 = np.array([0.0, 0.0, 1.0])

# ── CBF hyperparameters ────────────────────────────────────────────────────
ALPHA1 = 2.0    # first  ECBF class-K gain
ALPHA2 = 2.0    # second ECBF class-K gain
SLACK_WEIGHT = 1e4    # soft-constraint penalty (ensures QP always feasible)

# ── PD nominal tracking gains (σ(t) following) ────────────────────────────
K_P_NOM = 3.0   # proportional position gain  [1/s²]
K_D_NOM = 2.5   # velocity damping gain        [1/s]


# ── Attitude helpers (identical to baseline_controller) ───────────────────
def _lee_dcm(Fd, psi_d=0.0):
    b3d = Fd / max(np.linalg.norm(Fd), 1e-9)
    b1p = np.array([np.cos(psi_d), np.sin(psi_d), 0.0])
    b2d = np.cross(b3d, b1p)
    n = np.linalg.norm(b2d)
    if n < 1e-6:
        b2d = np.cross(b3d, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(b2d)
    b2d /= n
    return np.column_stack([np.cross(b2d, b3d), b2d, b3d])


def _rotor_from_dcm(Rmat):
    try:
        x, y, z, w = scipyR.from_matrix(Rmat).as_quat()
        return ga.rotor(w, -np.array([x, y, z])), True
    except Exception:
        return ga.rotor_identity(), False


_cache = {"t_prev": None, "Rd_prev": None}


def reset_cbf_state():
    _cache["t_prev"] = None
    _cache["Rd_prev"] = None


# ── CBF-QP core ────────────────────────────────────────────────────────────
def _cbf_qp(p, v, a_nom, obstacles, t):
    """Return (a_safe, qp_status, n_active_slack)."""
    active = [(obs, obs.position(t)) for obs in obstacles]
    if not active:
        return a_nom.copy(), "trivial", 0

    A_list, b_list = [], []
    for obs, oj in active:
        diff = p - oj
        rel_v = v - obs.vel
        d2 = float(np.dot(diff, diff))
        r2 = float(obs.radius ** 2)
        # coefficient of a in ECBF constraint: 2 diffᵀ a ≥ rhs
        lhs = 2.0 * diff
        rhs = (-2.0 * np.dot(rel_v, rel_v)
               - (ALPHA1 + ALPHA2) * 2.0 * np.dot(diff, rel_v)
               - ALPHA1 * ALPHA2 * (d2 - r2)
               - 2.0 * np.dot(diff, a_nom))   # a_nom part already in
        A_list.append(lhs)
        b_list.append(float(rhs))

    da = cp.Variable(3)
    slack = cp.Variable(len(active), nonneg=True)
    obj = cp.sum_squares(da) + SLACK_WEIGHT * cp.sum_squares(slack)
    cons = [A_list[j] @ da >= b_list[j] - slack[j]
            for j in range(len(active))]
    prob = cp.Problem(cp.Minimize(obj), cons)

    try:
        prob.solve(solver=cp.OSQP, warm_start=True,
                   eps_abs=1e-4, eps_rel=1e-4, max_iter=2000,
                   verbose=False, polish=False)
    except Exception:
        return a_nom.copy(), "failed", len(active)

    if prob.status in ("optimal", "optimal_inaccurate") and da.value is not None:
        n_sl = int(np.sum(slack.value > 1e-3)
                   ) if slack.value is not None else 0
        return a_nom + da.value, prob.status, n_sl
    return a_nom.copy(), prob.status or "failed", len(active)


# ── Public interface ───────────────────────────────────────────────────────
def compute_control_cbf(t0, state, obstacles, stt_params, drone_params, psi_d=0.0):
    """CBF-QP controller — signature matches all other controllers."""
    p, v, R, Omega_b = state["p"], state["v"], state["R"], state["Omega_b"]
    sigma_now = state["sigma"]
    m, g = drone_params.m, 9.81

    # ── Nominal: simple PD tracking σ(t) ──────────────────────────────────
    # σ_dot from the STT's own ODE (cheap: just calls sigma_value)
    sigma_dot = stt.sigma_value(sigma_now, t0, obstacles, stt_params)
    # Desired acceleration: PD toward σ plus feedforward σ_dot
    a_nom = (K_P_NOM * (sigma_now - p)
             + K_D_NOM * (sigma_dot - v))

    # ── CBF safety filter ──────────────────────────────────────────────────
    a_safe, qp_status, n_sl = _cbf_qp(p, v, a_nom, obstacles, t0)

    # ── Thrust + desired attitude ──────────────────────────────────────────
    F_total = m * (a_safe + g * E3)
    f_cmd = np.clip(np.linalg.norm(F_total),
                    drone_params.f_min, drone_params.f_max)
    Rd0, ok = _rotor_from_dcm(_lee_dcm(F_total, psi_d))
    Rd0 = ga.normalize_rotor(Rd0)

    # ── Backward-FD Ω_d (same as baseline) ────────────────────────────────
    t_prev, Rd_prev = _cache["t_prev"], _cache["Rd_prev"]
    if t_prev is not None and (t0 - t_prev) > 1e-9:
        dT = t0 - t_prev
        Rp = Rd_prev if ga.rotor_scalar(
            ga.geometric_product(ga.reverse(Rd_prev), Rd0)) >= 0 else -Rd_prev
        Omega_d = ga.to_bivector3(
            -2.0 * ga.geometric_product(ga.reverse(Rd0), (Rd0 - Rp) / dT))
    else:
        Omega_d = np.zeros(3)
    _cache["t_prev"], _cache["Rd_prev"] = t0, Rd0

    # ── Lee PD torque (same gains as baseline) ─────────────────────────────
    Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd0), R))
    eR = 0.5 * ga.to_bivector3(Re)
    eOmega = Omega_b - ga.sandwich_bivector(ga.reverse(Re), Omega_d)
    J = drone_params.J_diag
    tau = (2.5 * eR - 0.5 * eOmega + np.cross(Omega_b, J * Omega_b))
    tau_n = np.linalg.norm(tau)
    if tau_n > 1.5:
        tau = tau * (1.5 / tau_n)

    theta_e, _ = ga.rotor_log_angle_axis(Re)
    diag = dict(Rd=Rd0, Omega_d=Omega_d, Omega_dot_d=np.zeros(3), Re=Re,
                theta_e=theta_e, e_R=np.nan, rho_R=np.nan,
                dcm_ok=ok, qp_status=qp_status, n_cbf_slack=n_sl)
    return f_cmd, tau, diag
