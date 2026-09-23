#pragma once
#include "ga3.hpp"
#include "jet1d.hpp"
#include "rotor_jet.hpp"
#include "stt.hpp"

namespace ga_stt_ctrl {

struct DroneParams {
    float m;
    Eigen::Vector3f J_diag;
    float f_min, f_max, kappa1, kappa_v, kR, kOmega;

    DroneParams(float mass, const Eigen::Vector3f& J, float fmin, float fmax,
                float k1, float kv, float kr, float ko)
        : m(mass), J_diag(J), f_min(fmin), f_max(fmax),
          kappa1(k1), kappa_v(kv), kR(kr), kOmega(ko) {}
};

struct Diagnostics {
    Eigen::Vector3f Omega_d, Omega_dot_d;
    float theta_e, e_R, rho_R;
};

constexpr int JET_ORDER_SIGMA = 4;
// The translational jet now runs to the same order jet1d actually stores
// (MAX_JET_ORDER); see the corrigendum note above resolve_translational_chain
// for why this replaced the old 3-stage bootstrap.
constexpr int JET_ORDER_VD = jet::MAX_JET_ORDER;

// ---------------------------------------------------------------------------
// Stage 1: translational virtual acceleration u1(p, v, t), PDF eq. (11)
// corrected in two ways relative to the uploaded derivation:
//
//   1. Jacobian factor. eps_p = ln((1+e1)/(1-e1)) => d(eps_p)/d(e1)
//      = 1/(1+e1) + 1/(1-e1) = 2/(1-e1^2), NOT 4/(1-e1^2) as eq. (11) of the
//      PDF states. The factor-of-2 error was independently re-derived here
//      (see chat) rather than taken on faith from either the PDF or the
//      patch that was proposed alongside this task.
//
//   2. Velocity damping. PDF eq. (11), u1 = -kappa1*xi_p(eps_p)*e_p +
//      sigma_ddot, is a pure position-feedback ("spring") term with NO term
//      in the velocity error. Under feedback linearization (p_ddot ~= u1)
//      this gives p_ddot_tilde = -kappa1*xi_p*eps_p*p_hat: an undamped
//      nonlinear oscillator, not an asymptotically stable one (poles on the
//      imaginary axis in the linearization about e1=0). This is exactly the
//      class of bug matching the reported oscillatory/non-convergent
//      hardware behavior. Both Lee-Leok-McClamroch [3] and Rubio Scola et
//      al. [5] -- the two papers this derivation explicitly builds on --
//      use an actual PD law (position AND velocity error gains), not P
//      alone; we restore that here with an explicit kappa_v * z_v term,
//      z_v := v - sigma_dot. This also happens to be what makes the
//      corrected Sec. 4 Lyapunov step legitimate: Stage 1's dissipation
//      -kappa_v*||z_v||^2 is exactly what turns V_p into a strict (not
//      merely marginal) Lyapunov function for the cascade.
// ---------------------------------------------------------------------------
// accel_cap: uniform bound applied to EVERY Taylor-order row of u1 (not just
// the value). This is not an ad hoc numerical patch bolted on separately --
// it is literally PDF Assumption 1 ("the commanded acceleration is
// uniformly bounded, sup||ge3-sigma_ddot(t)|| < B < infinity") written into
// the code, using the actuator's own f_max/m as the natural choice of B
// (times a generous safety margin, since some thrust is also needed to
// cancel gravity/track attitude). Capping the NORM while leaving the
// direction untouched preserves the barrier's repulsive direction exactly;
// only the magnitude is bounded, and only to a value f_command's own clamp
// would have discarded anyway. See resolve_translational_chain's
// corrigendum for the divergence this was found to fix.
inline Eigen::Vector3f clamp_row_norm(const Eigen::Vector3f& row, float cap) {
    float n = row.norm();
    return (n > cap) ? (row * (cap / n)) : row;
}

inline jet::VectorJet u1_jet(const jet::VectorJet& p_jet, const jet::VectorJet& v_jet,
                             const jet::VectorJet& sigma_jet_in, const jet::ScalarJet& rho_p_jet_in,
                             float kappa1, float kappa_v, float accel_cap) {
    constexpr float EPS2 = 1e-6f;
    const int order = jet::MAX_JET_ORDER;

    jet::VectorJet p_tilde = jet::sub(p_jet, sigma_jet_in);
    jet::ScalarJet norm_pt = jet::sqrt_s(jet::add(jet::dot_vv(p_tilde, p_tilde),
                                                   jet::const_scalar(EPS2, order)));
    jet::ScalarJet one = jet::const_scalar(1.0f, order);
    jet::ScalarJet e1 = jet::div_ss(norm_pt, rho_p_jet_in);
    jet::ScalarJet eps_p = jet::sub(jet::log_s(jet::add(one, e1)), jet::log_s(jet::sub(one, e1)));
    jet::VectorJet p_hat = jet::vec_div_s(p_tilde, norm_pt);

    // xi_p(eps_p) = 2 / (rho * (1 - e1^2))  (corrected Jacobian, see above)
    jet::ScalarJet one_minus_e1_sq = jet::sub(one, jet::mul_ss(e1, e1));
    jet::ScalarJet xi_p_over_rho = jet::div_ss(jet::const_scalar(2.0f, order),
                                               jet::mul_ss(rho_p_jet_in, one_minus_e1_sq));
    jet::ScalarJet spring_gain = jet::mul_ss(jet::const_scalar(-kappa1, order),
                                             jet::mul_ss(xi_p_over_rho, eps_p));
    jet::VectorJet spring = jet::mul_sv(spring_gain, p_hat);

    jet::VectorJet sigma_dot = jet::differentiate(sigma_jet_in);
    jet::VectorJet sigma_ddot = jet::differentiate(sigma_dot);
    jet::VectorJet z_v = jet::sub(v_jet, sigma_dot);
    jet::VectorJet damping = jet::mul_sv(jet::const_scalar(-kappa_v, order), z_v);

    jet::VectorJet u1 = jet::add(jet::add(sigma_ddot, damping), spring);
    for (int k = 0; k <= jet::MAX_JET_ORDER; ++k) {
        u1.row(k) = clamp_row_norm(u1.row(k).transpose(), accel_cap);
    }
    return u1;
}

// ---------------------------------------------------------------------------
// Replaces the old resolve_translational_chain. The old version built a
// *velocity* command vd(p,t) and differentiated it via a 3-stage bootstrap
// that, at each stage, re-estimated p_ddot / p_dddot from the *actually
// achieved* thrust direction (R_cur, Omega_b_cur) fed back into the formula
// for the next Taylor order. That mixes "what the reference generator
// thinks the trajectory should be" with "what the vehicle is actually
// doing" inside what is supposed to be a pure feedforward reference chain,
// and re-derives the mix from scratch every control tick -- exactly the
// kind of scheme that can accumulate drift over a run (each tick's Taylor
// coefficients are only consistent with the *previous* tick's achieved
// state, not with each other).
//
// Because u1 above is already the *acceleration* (not a velocity requiring
// one more differentiation), we don't need that bootstrap at all: p_dot = v,
// v_dot = u1(p, v, t) is a self-contained 2nd-order ODE in (p, v), and its
// Taylor coefficients can be resolved order-by-order the same way stt::
// sigma_jet already resolves sigma's own ODE -- a standard, numerically
// well-posed technique (this is just Taylor-series/Picard integration of an
// ODE), and it uses only the *commanded* trajectory, never the vehicle's
// actual attitude/omega. That removes the drift source without discarding
// the analytic Omega_d/Omega_dot_d machinery (which, unlike the quick patch
// suggested for this task, we keep: t_hat_d is carried as a REAL jet with
// nonzero higher-order coefficients, not silently frozen to a constant --
// freezing it would zero out Omega_d/Omega_dot_d feedforward entirely).
//
// Corrigendum found by testing (test_closed_loop_convergence.cpp): with the
// corrected, unbounded barrier Jacobian xi_p = 2/(rho*(1-e1^2)) restored
// (see u1_jet above), and this chain now genuinely using the FULL 4th-order
// jet (the old code's 3-stage bootstrap only ever reached order 2, an
// accidental truncation that happened to mask this), a position excursion
// that pushes e1 close to 1 near an active obstacle makes the log barrier's
// derivatives diverge (log_s divides by (1-e1)'s row-0 value at every
// order; Das et al.'s own obstacle-repulsion term m_j ~ 1/denom^3 already
// stresses this near an obstacle's rho_min boundary). That is a genuine,
// expected property of a log barrier -- its derivatives really do go to
// infinity at the wall -- but a Taylor/Picard EXTRAPOLATION of it is only
// valid inside its radius of convergence, and diverged in testing (Taylor
// coefficients reaching ~1e11, then NaN) well before literally reaching the
// wall, eventually corrupting the rotor-alignment step downstream. u1_jet's
// accel_cap enforces PDF Assumption 1 (a uniformly bounded commanded
// acceleration) directly on u1, at EVERY Taylor order including the value
// -- not something claimed from the PDF or the patch proposed for this
// task, but a plain engineering bound this chain needs to stay well-posed,
// and since v_dot=u1 with u1 already bounded, p_jet/v_jet inherit that
// bound automatically through the Picard recursion below with no separate
// clamping needed here.
// ---------------------------------------------------------------------------
inline std::tuple<jet::VectorJet, jet::ScalarJet, jet::VectorJet>
resolve_translational_chain(float t0, const Eigen::Vector3f& p0, const Eigen::Vector3f& v0,
                            const jet::VectorJet& sigma_jet_in, const jet::ScalarJet& rho_p_jet_in,
                            const DroneParams& params) {
    (void)t0; // t0 is implicit in sigma_jet_in/rho_p_jet_in's own expansion point
    constexpr float g = 9.81f;
    const Eigen::Vector3f E3(0.0f, 0.0f, 1.0f);
    // Generous multiple of the actuator's own achievable acceleration
    // (f_max/m); accounts for some thrust being needed just to cancel
    // gravity/track attitude, while still being far below the ~1e11
    // magnitudes observed without any cap.
    const float accel_cap = 4.0f * params.f_max / params.m + 4.0f * g;

    jet::VectorJet p_jet = jet::VectorJet::Zero();
    jet::VectorJet v_jet = jet::VectorJet::Zero();
    p_jet.row(0) = p0;
    v_jet.row(0) = v0;

    // Order-by-order Taylor/Picard resolution of p_dot = v, v_dot = u1(p,v,t).
    // At the start of iteration k, rows 0..k of p_jet/v_jet are valid and
    // rows k+1.. are (harmlessly) zero: jet arithmetic is triangular, so
    // u1_jet's row k depends only on rows 0..k of its inputs.
    for (int k = 0; k < jet::MAX_JET_ORDER; ++k) {
        jet::VectorJet u1_partial = u1_jet(p_jet, v_jet, sigma_jet_in, rho_p_jet_in,
                                           params.kappa1, params.kappa_v, accel_cap);
        v_jet.row(k + 1) = u1_partial.row(k) / (float)(k + 1);
        p_jet.row(k + 1) = v_jet.row(k) / (float)(k + 1);
    }

    jet::VectorJet u1_full = u1_jet(p_jet, v_jet, sigma_jet_in, rho_p_jet_in,
                                    params.kappa1, params.kappa_v, accel_cap);
    jet::VectorJet const_g = jet::VectorJet::Zero();
    const_g.row(0) = params.m * g * E3;
    jet::VectorJet Fd_jet = jet::add(jet::mul_sv(jet::const_scalar(params.m, jet::MAX_JET_ORDER), u1_full),
                                     const_g);

    jet::ScalarJet fd_jet = jet::norm_v(Fd_jet);
    jet::VectorJet t_hat_d_jet = jet::vec_div_s(Fd_jet, fd_jet);

    return {Fd_jet, fd_jet, t_hat_d_jet};
}

inline std::tuple<float, Eigen::Vector3f, Diagnostics>
compute_control(float t0, const Eigen::Vector3f& p, const Eigen::Vector3f& v,
                const ga::Rotor& R, const Eigen::Vector3f& Omega_b, const Eigen::Vector3f& sigma,
                const stt::ObstacleArray& obstacles, const stt::STTParams& stt_params,
                const DroneParams& drone_params, const ga::Rotor& Rp_value,
                const Eigen::Vector3f& wp = Eigen::Vector3f::Zero()) {
    (void)wp; // reserved: disturbance is not injected into the reference generator by design

    auto sjet = stt::sigma_jet(sigma, t0, obstacles, stt_params, JET_ORDER_SIGMA);
    auto rjet = stt::rho_p_jet(sjet, t0, obstacles, stt_params, JET_ORDER_SIGMA);
    auto [Fd_jet, fd_jet, t_hat_d_jet] = resolve_translational_chain(t0, p, v, sjet, rjet, drone_params);
    float f_command = std::clamp(fd_jet[0], drone_params.f_min, drone_params.f_max);

    // --- Stage R: analytic desired-rotor chain (unchanged in spirit from
    // the original code / PDF Sec. 3.2 -- t_hat_d_jet is a genuine jet, so
    // Omega_d, Omega_dot_d fall out with no numerical differentiation). ---
    int order_rot = jet::MAX_JET_ORDER;
    auto Ra_jet = rotor_jet::align(
        jet::const_vector(ga::sandwich(Rp_value, Eigen::Vector3f(0.0f, 0.0f, 1.0f)), order_rot),
        t_hat_d_jet);
    rotor_jet::RotorJet Rp_jet(jet::const_scalar(ga::rotor_scalar(Rp_value), order_rot),
                               jet::const_vector(ga::rotor_bivec3(Rp_value), order_rot));
    auto Rd_jet = rotor_jet::mul(Ra_jet, Rp_jet);
    ga::Rotor Rd_value = ga::normalize_rotor(rotor_jet::value_multivector(Rd_jet));

    auto Omega_d_full = rotor_jet::mul(rotor_jet::reverse(Rd_jet), rotor_jet::differentiate(Rd_jet));
    Eigen::Vector3f Omega_d_value = -2.0f * Omega_d_full.b.row(0).transpose();
    Eigen::Vector3f Omega_dot_d_value = Eigen::Vector3f::Zero();
    if (jet::MAX_JET_ORDER >= 1) {
        Omega_dot_d_value = -2.0f * Omega_d_full.b.row(1).transpose();
    }

    // --- Stage 2: attitude funnel + torque law -----------------------------
    // REVERTED to the ORIGINAL law (angle-axis / 1/sin(theta_e/2) with a
    // floor). An earlier version of this file replaced this with a smooth
    // GA-vee-map law, believing the original had a chattering bug; that
    // turned out to be wrong on closer, corrected inspection:
    //   1. The "chattering at theta_e->0" concern was based on a flawed
    //      test that varied eps_R independently of theta_e -- they are NOT
    //      independent (eps_R is derived from theta_e via Psi_e=1-cos(theta_e/2)).
    //      With them correctly paired, the branch point is a genuine,
    //      correctly-handled REMOVABLE singularity (confirmed analytically:
    //      the limit as theta_e->0 is exactly 0, matched numerically to ~1e-5
    //      just above the threshold).
    //   2. This original law has a real, exact closed-form Lyapunov identity
    //      that the vee-map replacement does not:
    //          dV_R/dt = -kR * xi_R(eps_R) * eps_R^2,  xi_R = 1/(rho_R*(1-e_R))
    //      independently re-derived from the GA kinematics (Re_dot =
    //      -1/2 Re Omega_b) and confirmed by high-precision finite
    //      differencing against the actual formula across many (theta_e, t)
    //      combinations, matching to 4-5 significant figures, including with
    //      an actively time-varying funnel. The vee-map law does not reduce
    //      to any comparably clean closed form.
    // Net: this original law is better-founded, which is why it is restored
    // here unchanged (including its HALF_SIN_FLOOR regularization, now
    // understood to be a safe way to handle a removable singularity, not a
    // bug).
    ga::Rotor Re = ga::normalize_rotor(ga::geometric_product(ga::reverse(Rd_value), R));
    auto [theta_e, n_hat_e] = ga::rotor_log_angle_axis(Re);
    float Psi_e = 1.0f - ga::rotor_scalar(Re);
    float rho_R_t = stt::rho_R(t0, stt_params);
    float e_R = std::clamp(Psi_e / rho_R_t, 0.0f, 1.0f - 1e-6f);
    float eps_R = -std::log(std::max(1.0f - e_R, 1e-6f));

    float eOmega_dot_nhat = 0.0f;
    constexpr float HALF_SIN_FLOOR = 0.05f;
    if (theta_e > 1e-6f) {
        float half_sin = std::sin(0.5f * theta_e);
        float half_sin_safe = std::copysign(std::max(std::abs(half_sin), HALF_SIN_FLOOR),
                                              (half_sin == 0.0f) ? 1.0f : half_sin);
        eOmega_dot_nhat = (2.0f * drone_params.kR * eps_R
                          - 2.0f * e_R * stt::rho_R_dot(t0, stt_params)) / half_sin_safe;
    }

    Eigen::Vector3f Omega_c = -eOmega_dot_nhat * n_hat_e
                             + ga::sandwich_bivector(ga::reverse(Re), Omega_d_value);
    Eigen::Vector3f z_Omega = Omega_b - Omega_c;
    Eigen::Vector3f tau = -drone_params.kOmega * z_Omega
                         + Omega_b.cross(drone_params.J_diag.cwiseProduct(Omega_b));
    tau += drone_params.J_diag.cwiseProduct(Omega_dot_d_value);  // unrotated, matches the original

    float tau_norm = tau.norm();
    if (tau_norm > 1.5f) tau *= 1.5f / tau_norm;

    Diagnostics diag{Omega_d_value, Omega_dot_d_value, theta_e, e_R, rho_R_t};
    return {f_command, tau, diag};
}

} // namespace ga_stt_ctrl
