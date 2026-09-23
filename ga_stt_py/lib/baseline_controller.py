"""
baseline_controller.py — classical comparison baseline.

Das et al.'s original real-time STT only
constrains a *translational* output -- it has no rotational tube and no
notion of a desired attitude at all. To make a quadrotor fly safely inside
that translational tube, a practitioner has to bolt on a separate
attitude controller. The standard way to do this (Lee et al. 2010) is:

  1. Build the desired body frame from the desired force vector via the
     classical (b3d, b1d, b2d) construction (DCM).
  2. Differentiate the resulting R_d(t) to get Omega_d. Because sigma(t)/
     rho_p(t) have the STT's switching obstacle-avoidance nonlinearity,
     there is no analytic flatness-based Omega_d available. The standard 
     fallback -- is to numerically differentiate R_d using a simple BACKWARD 
     difference between consecutive control-loop evaluations.
  3. Omega_dot_d (needed for the angular-acceleration feedforward term) is
     a SECOND derivative; numerically differentiating twice is noisy
     enough that most practical implementations simply drop that
     feedforward term (kOmega-only PD on the rate error).

To isolate the comparison to EXACTLY what the GA-STT letter claims an
advantage on (Sec. I: "approximation-free... no numerical differentiation
... avoids singularities" -- i.e. the ROTATIONAL chain), Stage 1
(translation, hence thrust command) reuses the SAME exact STT logic as the
GA-STT controller. Only how R_d/Omega_d are obtained from that differs.
"""

import numpy as np
from scipy.spatial.transform import Rotation as scipyR
import ga3 as ga
import ga_stt_controller as ctrl

E3 = np.array([0.0, 0.0, 1.0])

_cache = {"t_prev": None, "Rd_prev": None}


def reset_baseline_state():
    _cache["t_prev"] = None
    _cache["Rd_prev"] = None


def lee_dcm_from_force(Fd, psi_d=0.0):
    fd = np.linalg.norm(Fd)
    fd_safe = max(fd, 1e-9)
    b3d = Fd / fd_safe
    b1_psi = np.array([np.cos(psi_d), np.sin(psi_d), 0.0])
    b2d = np.cross(b3d, b1_psi)
    n = np.linalg.norm(b2d)
    if n < 1e-6:
        b1_psi = np.array([0.0, 1.0, 0.0])
        b2d = np.cross(b3d, b1_psi)
        n = np.linalg.norm(b2d)
    b2d = b2d / n
    b1d = np.cross(b2d, b3d)
    Rmat = np.column_stack([b1d, b2d, b3d])
    return Rmat, fd


def rotor_from_dcm(Rmat):
    """Convert rotation matrix to GA rotor via scipy. Returns (rotor, ok_flag).
    Returns (identity, False) if the matrix is numerically degenerate — this
    failure mode is exactly the singularity the GA-STT letter claims to avoid."""
    try:
        # scipy's from_matrix requires an orthonormal, det=+1 matrix; near-
        # singular Fd directions produce an ill-conditioned b3d/b2d/b1d frame
        # that can fail SVD — a known limitation of Euler/quaternion pipelines.
        quat_xyzw = scipyR.from_matrix(Rmat).as_quat()
        x, y, z, w = quat_xyzw
        return ga.rotor(w, -np.array([x, y, z])), True
    except Exception:
        return ga.rotor_identity(), False  # fallback: identity (hover attempt)


def compute_control_baseline(t0, state, obstacles, stt_params, drone_params, psi_d=0.0):
    p, v, R, Omega_b = state["p"], state["v"], state["R"], state["Omega_b"]
    sigma_now = state["sigma"]

    # Stage 1: EXACT same Fd as GA-STT (isolates the comparison to rotation)
    sjet = ctrl.stt.sigma_jet(sigma_now, t0, obstacles,
                              stt_params, ctrl.JET_ORDER_SIGMA)
    rjet = ctrl.stt.rho_p_jet(
        sjet, t0, obstacles, stt_params, ctrl.JET_ORDER_SIGMA)
    Fd_jet, fd_jet, that_d_jet = ctrl.resolve_translational_chain(
        t0, p, v, R, Omega_b, np.zeros(3), sjet, rjet, drone_params)
    Fd0 = Fd_jet[0]
    f_command = np.clip(np.linalg.norm(
        Fd0), drone_params.f_min, drone_params.f_max)

    Rmat0, _ = lee_dcm_from_force(Fd0, psi_d)
    Rd0, dcm_ok = rotor_from_dcm(Rmat0)
    Rd0 = ga.normalize_rotor(Rd0)

    # Stage R (baseline): backward finite-difference Omega_d using the
    # PREVIOUS control-loop's Rd.
    t_prev, Rd_prev = _cache["t_prev"], _cache["Rd_prev"]
    if t_prev is not None and (t0 - t_prev) > 1e-9:
        dt_loop = t0 - t_prev
        Rd_prev_aligned = Rd_prev
        if ga.rotor_scalar(ga.geometric_product(ga.reverse(Rd_prev), Rd0)) < 0:
            Rd_prev_aligned = -Rd_prev
        Rd_dot_fd = (Rd0 - Rd_prev_aligned) / dt_loop
        Omega_d = ga.to_bivector3(-2 *
                                  ga.geometric_product(ga.reverse(Rd0), Rd_dot_fd))
    else:
        Omega_d = np.zeros(3)
    Omega_dot_d = np.zeros(3)  # dropped: see module docstring

    _cache["t_prev"], _cache["Rd_prev"] = t0, Rd0

    Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd0), R))
    # (small-angle) vector part, classical Lee-style error
    eR_vec = ga.to_bivector3(Re)
    eR = 0.5 * eR_vec

    eOmega = Omega_b - ga.sandwich_bivector(ga.reverse(Re), Omega_d)
    J = drone_params.J_diag
    commutator_term = np.cross(Omega_b, J * Omega_b)
    kR_lee = 2.5      # N*m per rad (small-angle attitude proportional gain)
    kOmega_lee = 0.5  # N*m per (rad/s) (rate damping gain)
    tau = kR_lee * eR - kOmega_lee * eOmega + commutator_term
    tau_max = 1.5     # N*m, same actuator saturation limit as GA-STT controller
    tau_norm = np.linalg.norm(tau)
    if tau_norm > tau_max:
        tau = tau * (tau_max / tau_norm)

    theta_e_diag, _ = ga.rotor_log_angle_axis(Re)
    diag = dict(Rd=Rd0, Omega_d=Omega_d, Omega_dot_d=Omega_dot_d, Re=Re,
                theta_e=theta_e_diag, e_R=np.nan, rho_R=np.nan,
                dcm_ok=dcm_ok)
    return f_command, tau, diag
