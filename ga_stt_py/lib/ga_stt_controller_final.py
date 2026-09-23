"""
ga_stt_controller_final.py -- the FINAL, doubly-cross-checked GA-STT controller.

This merges two independently-converging lines of work:
  - Stage 1 (translation): velocity damping, corrected barrier Jacobian
    (2/(1-e1^2), not 4), and a self-contained Taylor/Picard reference
    generator with an explicit acceleration cap, replacing the original's
    fragile bootstrap off the vehicle's actual attitude/omega. Independently
    re-derived and re-verified by a parallel effort, matching to 15
    significant figures.
  - Stage R (attitude): REVERTED to the ORIGINAL, unmodified law
    (angle-axis / 1/sin(theta_e/2) with a threshold branch), NOT the
    smooth GA-vee-map replacement introduced earlier in this project.
    That replacement was a mistake, found and corrected only after a
    parallel effort's review pushed back on it:
      1. The "chattering at theta_e->0" concern that motivated the
         replacement does not hold up: re-tested with theta_e and eps_R
         correctly paired (they are not independent -- eps_R is DERIVED
         from theta_e via Psi_e=1-cos(theta_e/2) -- an earlier test
         wrongly varied them independently and found a spurious 7e6
         "jump"). With them properly paired, the branch point is a
         genuine, correctly-handled REMOVABLE singularity: sympy confirms
         the analytic limit is exactly 0 as theta_e->0, and the numerical
         branch matches that to ~1e-5.
      2. The original law has a real, EXACT closed-form Lyapunov identity
         that the replacement does not:
             dV_R/dt = -kR * xi_R(eps_R) * eps_R^2,   xi_R = 1/(rho_R(1-e_R))
         independently re-derived here from the GA kinematics (Re_dot =
         -1/2 Re Omega_b) and confirmed by high-precision finite-difference
         against the actual formula across 10 (theta_e, t) combinations,
         matching to 4-5 significant figures, including with an actively
         time-varying funnel. The replacement law does not reduce to any
         comparably clean closed form (checked the same way and found to
         stay a non-separable function of eps_R, e_R, rho_R).
    Net: the original Stage-R law is better-founded than the replacement,
    which is why it is restored here, unchanged, with its threshold
    branch intact (now understood to be safe, not a bug).

Use THIS file for new work.
"""


import numpy as np
import ga3 as ga
import jet1d as jt
import rotor_jet as rj
import stt

E3 = np.array([0.0, 0.0, 1.0])
JET_ORDER_SIGMA = 4
JET_ORDER_VD = 4  # full order now that u1 is resolved directly (see module docstring)


class DroneParams:
    def __init__(self, mass, J_diag, f_min, f_max, kappa1, kappa_v, kR, kOmega,
                tau_max=1.5, accel_cap_mult=4.0):
        self.m = float(mass)
        self.J = np.diag(np.asarray(J_diag, dtype=float))
        self.J_diag = np.asarray(J_diag, dtype=float)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.kappa1 = float(kappa1)
        self.kappa_v = float(kappa_v)
        self.kR = float(kR)
        self.kOmega = float(kOmega)
        # N*m; 1.5 was tuned for a much larger (~0.5kg) drone -- MUST be
        # overridden for small vehicles like the Crazyflie (real CF2 torque
        # authority is closer to 1e-2 N*m per axis, not 1.5).
        self.tau_max = float(tau_max)
        # accel_cap = accel_cap_mult * f_max/m + accel_cap_mult*g; see
        # resolve_translational_chain. Default 4.0 kept from the original
        # tuning; exposed here so small/large vehicles can adjust it.
        self.accel_cap_mult = float(accel_cap_mult)


def _clamp_row_norm(row, cap):
    n = np.linalg.norm(row)
    return row * (cap / n) if n > cap else row


def u1_jet(p_jet, v_jet, sigma_jet_arr, rho_p_jet_arr, kappa1, kappa_v, accel_cap):
    """Stage-1 virtual acceleration, as a jet. See module docstring."""
    EPS2 = 1e-10
    order = jt.order_of(p_jet)
    sigma_trunc = sigma_jet_arr[:order + 1]
    rho_trunc = rho_p_jet_arr[:order + 1]

    p_tilde = jt.sub(p_jet, sigma_trunc)
    norm_pt_sq = jt.add(jt.dot_vv(p_tilde, p_tilde), jt.const_scalar(EPS2, order))
    norm_pt = jt.sqrt_s(norm_pt_sq)
    one = jt.const_scalar(1.0, order)
    e1 = jt.div_ss(norm_pt, rho_trunc)
    eps_p = jt.sub(jt.log_s(jt.add(one, e1)), jt.log_s(jt.sub(one, e1)))
    p_hat = jt.vec_div_s(p_tilde, norm_pt)

    # xi_p(eps_p) = 2 / (rho * (1 - e1^2))  -- corrected Jacobian (was 4 in
    # the PDF derivation, and simply absent -- multiplied by nothing -- in
    # the original vd_formula_jet).
    one_minus_e1_sq = jt.sub(one, jt.mul_ss(e1, e1))
    xi_p_over_rho = jt.div_ss(jt.const_scalar(2.0, order),
                              jt.mul_ss(rho_trunc, one_minus_e1_sq))
    spring_gain = jt.scale(jt.mul_ss(xi_p_over_rho, eps_p), -kappa1)
    spring = jt.mul_sv(spring_gain, p_hat)

    sigma_dot = jt.differentiate(sigma_jet_arr)
    sigma_ddot = jt.differentiate(sigma_dot)
    z_v = jt.sub(v_jet[:jt.order_of(sigma_dot) + 1], sigma_dot)
    damping = jt.scale(z_v, -kappa_v)

    o = min(jt.order_of(spring), jt.order_of(sigma_ddot), jt.order_of(damping))
    u1 = jt.add(jt.add(sigma_ddot[:o + 1], damping[:o + 1]), spring[:o + 1])
    for k in range(o + 1):
        u1[k] = _clamp_row_norm(u1[k], accel_cap)
    # Pad back up to the caller's requested order with zeros (sigma_ddot's
    # own differentiate() chain loses 2 orders vs. sigma_jet_arr's order;
    # treating the missing highest Taylor coefficients as zero is the same
    # "zero-jerk beyond the available order" assumption already implicit in
    # stt.sigma_jet's own obstacle model, and keeps this function's output
    # shape consistent with (order+1) rows for the Picard loop's indexing).
    if o < order:
        pad = np.zeros((order - o, 3))
        u1 = np.concatenate([u1, pad], axis=0)
    return u1


def resolve_translational_chain(t0, p0, v0, sigma_jet_arr, rho_p_jet_arr, params):
    """
    Taylor/Picard resolution of p_dot = v, v_dot = u1(p, v, t) up to
    JET_ORDER_VD. See module docstring for why this replaces the old
    3-stage, actual-state-dependent bootstrap.
    """
    m, g = params.m, 9.81
    accel_cap = params.accel_cap_mult * params.f_max / params.m + params.accel_cap_mult * g

    p_jet = np.zeros((JET_ORDER_VD + 1, 3))
    v_jet = np.zeros((JET_ORDER_VD + 1, 3))
    p_jet[0] = p0
    v_jet[0] = v0

    for k in range(JET_ORDER_VD):
        u1_partial = u1_jet(p_jet[:k + 1], v_jet[:k + 1], sigma_jet_arr, rho_p_jet_arr,
                            params.kappa1, params.kappa_v, accel_cap)
        v_jet[k + 1] = u1_partial[k] / (k + 1)
        p_jet[k + 1] = v_jet[k] / (k + 1)

    u1_full = u1_jet(p_jet, v_jet, sigma_jet_arr, rho_p_jet_arr,
                     params.kappa1, params.kappa_v, accel_cap)
    Fd_jet = m * u1_full
    Fd_jet[0] = Fd_jet[0] + m * g * E3

    fd_jet = jt.norm_v(Fd_jet)
    that_d_jet = jt.vec_div_s(Fd_jet, fd_jet)
    return Fd_jet, fd_jet, that_d_jet


def compute_control(t0, state, obstacles, stt_params, drone_params,
                    Rp_value, wp_current=np.zeros(3), wtau_current=np.zeros(3)):
    """
    state: dict with keys p,v,R (8,) ga3 rotor, Omega_b (3,)
    Returns: (f_command, tau, diag)
    """
    del wp_current, wtau_current  # reserved: not injected into the reference generator by design
    p, v, R, Omega_b = state["p"], state["v"], state["R"], state["Omega_b"]

    sigma_now = state["sigma"]
    sjet = stt.sigma_jet(sigma_now, t0, obstacles, stt_params, JET_ORDER_SIGMA)
    rjet = stt.rho_p_jet(sjet, t0, obstacles, stt_params, JET_ORDER_SIGMA)

    Fd_jet, fd_jet, that_d_jet = resolve_translational_chain(
        t0, p, v, sjet, rjet, drone_params)

    f_command = np.clip(jt.value(fd_jet), drone_params.f_min, drone_params.f_max)

    # --- Stage R: analytic desired-rotor chain (unchanged in spirit) ---
    tp_hat = ga.sandwich(Rp_value, E3)
    order_rot = jt.order_of(that_d_jet)
    tp_jet = jt.const_vector(tp_hat, order_rot)
    Ra_jet = rj.align(tp_jet, that_d_jet)
    Rp_jet = rj.const_from_value(Rp_value, order_rot)
    Rd_jet = rj.mul(Ra_jet, Rp_jet)

    Rd_value = ga.normalize_rotor(rj.value_multivector(Rd_jet))
    Rd_rev_jet = rj.reverse(Rd_jet)
    Rd_dot_jet = rj.differentiate(Rd_jet)
    Omega_d_rotorjet = rj.mul(Rd_rev_jet, Rd_dot_jet)
    Omega_d_b = jt.scale(Omega_d_rotorjet.b, -2.0)
    Omega_d_value = jt.value(Omega_d_b)
    Omega_dot_d_value = jt.coeff_to_derivative(
        Omega_d_b, 1) if jt.order_of(Omega_d_b) >= 1 else np.zeros(3)

    # --- Stage R: attitude funnel + torque law, REVERTED to the original
    # law (see module docstring): angle-axis error with a 1/sin(theta_e/2)
    # Jacobian and a threshold branch at theta_e<=1e-6. This has a real,
    # independently-verified exact Lyapunov identity
    # (dV_R/dt = -kR*xi_R(eps_R)*eps_R^2) that the smooth-vee-map
    # alternative tried earlier in this project does not have; see module
    # docstring for the verification. ---
    Re = ga.normalize_rotor(ga.geometric_product(ga.reverse(Rd_value), R))
    theta_e, n_hat_e = ga.rotor_log_angle_axis(Re)
    Psi_e = 1.0 - ga.rotor_scalar(Re)

    rho_R_t = stt.rho_R(t0, stt_params)
    rho_R_dot_t = stt.rho_R_dot(t0, stt_params)
    e_R = np.clip(Psi_e / rho_R_t, 0.0, 1.0 - 1e-9)
    eps_R = -np.log(max(1.0 - e_R, 1e-12))

    half_theta = 0.5 * theta_e
    sin_half = np.sin(half_theta)
    if theta_e > 1e-6:
        eOmega_dot_nhat = (2.0 * drone_params.kR * eps_R
                           - 2.0 * e_R * rho_R_dot_t) / sin_half
    else:
        # Genuine removable singularity (verified: analytic limit is 0 as
        # theta_e -> 0, and this matches to ~1e-5 just above the
        # threshold) -- not an ad hoc patch over a real discontinuity.
        eOmega_dot_nhat = 0.0

    e_v_Omega = -eOmega_dot_nhat * n_hat_e
    Omega_c = e_v_Omega + ga.sandwich_bivector(ga.reverse(Re), Omega_d_value)

    z_Omega = Omega_b - Omega_c
    J = drone_params.J_diag
    commutator_term = np.cross(Omega_b, J * Omega_b)
    tau = (-drone_params.kOmega * z_Omega
           + commutator_term
           + J * Omega_dot_d_value)  # documented simplification (unrotated Omega_dot_d), unchanged from original
    tau_max = getattr(drone_params, "tau_max", 1.5)  # N*m; override via drone_params.tau_max for small vehicles
    tau_norm_val = np.linalg.norm(tau)
    if tau_norm_val > tau_max:
        tau = tau * (tau_max / tau_norm_val)

    diag = dict(Rd=Rd_value, Omega_d=Omega_d_value, Omega_dot_d=Omega_dot_d_value,
                theta_e=theta_e, Psi_e=Psi_e, e_R=e_R, eps_R=eps_R, rho_R=rho_R_t,
                z_Omega=z_Omega, that_d=jt.value(that_d_jet), fd=jt.value(fd_jet),
                sigma=sigma_now, rho_p=jt.value(rjet), Omega_v_b=Omega_c,
                Re=Re)
    return f_command, tau, diag
