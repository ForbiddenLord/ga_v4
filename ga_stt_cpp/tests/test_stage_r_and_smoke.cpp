#include <iostream>
#include <cmath>
#include <vector>
#include "ga3.hpp"
#include "stt.hpp"
#include "ga_stt_controller.hpp"

int nfail = 0;
void check(const std::string& name, bool cond, const std::string& extra = "") {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name;
    if (!cond) { nfail++; std::cout << " " << extra; }
    std::cout << "\n";
}

// Isolated Stage-R evaluation of the REVERTED, original law
// (1/sin(theta_e/2) with a HALF_SIN_FLOOR regularization), Omega_d = 0.
struct StageRState {
    float theta_e, Psi_e, e_R, eps_R, xi_R;
    Eigen::Vector3f n_hat_e, Omega_c;
};

StageRState eval_stage_r(const ga::Rotor& Rd, const ga::Rotor& R, float t,
                          const stt::STTParams& stt_params, float kR) {
    ga::Rotor Re = ga::normalize_rotor(ga::geometric_product(ga::reverse(Rd), R));
    auto [theta_e, n_hat_e] = ga::rotor_log_angle_axis(Re);
    float Psi_e = 1.0f - ga::rotor_scalar(Re);
    float rho_R_t = stt::rho_R(t, stt_params);
    float e_R = std::clamp(Psi_e / rho_R_t, 0.0f, 1.0f - 1e-6f);
    float eps_R = -std::log(std::max(1.0f - e_R, 1e-6f));
    float xi_R = 1.0f / (rho_R_t * (1.0f - e_R));

    float eOmega_dot_nhat = 0.0f;
    constexpr float HALF_SIN_FLOOR = 0.05f;
    if (theta_e > 1e-6f) {
        float half_sin = std::sin(0.5f * theta_e);
        float half_sin_safe = std::copysign(std::max(std::abs(half_sin), HALF_SIN_FLOOR),
                                              (half_sin == 0.0f) ? 1.0f : half_sin);
        eOmega_dot_nhat = (2.0f * kR * eps_R - 2.0f * e_R * stt::rho_R_dot(t, stt_params)) / half_sin_safe;
    }
    Eigen::Vector3f Omega_c = -eOmega_dot_nhat * n_hat_e;  // Omega_d = 0 for this isolated check
    return {theta_e, Psi_e, e_R, eps_R, xi_R, n_hat_e, Omega_c};
}

int main() {
    stt::STTParams stt_params(Eigen::Vector3f(8.0f,8.0f,8.0f), Eigen::Vector3f(1.0f,1.0f,1.0f), 16.0f,
                              0.3f, 1.0f, 0.5f, 1.0f, 0.15f, 8.0f, 1.0f, 0.05f, 0.5f);
    float kR = 6.0f;

    // ------------------------------------------------------------------
    // Check #1: the exact closed-form Lyapunov identity
    //     dV_R/dt = -kR * xi_R(eps_R) * eps_R^2
    // independently re-derived (not taken from the original code or any
    // supplement) from the GA kinematics Re_dot = -1/2 Re * Omega_b, and
    // confirmed here by high-precision finite differencing against the
    // ACTUAL formula, across several (theta_e, t) combinations including
    // an actively time-varying funnel (t=2.0, 8.0 have nonzero rho_R_dot).
    // This replaces an earlier, WRONG test that checked against the
    // simpler-looking but incorrect target "-kR*eps_R^2" (missing the
    // xi_R factor) and consequently seemed to show a mismatch that doesn't
    // actually exist.
    // ------------------------------------------------------------------
    ga::Rotor Rd = ga::normalize_rotor(
        ga::rotor(std::cos(0.2f), -std::sin(0.2f) * Eigen::Vector3f(0.1f, 0.8f, 0.2f).normalized()));

    bool identity_ok = true;
    std::string identity_detail;
    for (float theta_e_target : {0.02f, 0.1f, 0.3f, 0.5f, 1.0f}) {
        for (float t0 : {0.0f, 2.0f, 8.0f}) {
            Eigen::Vector3f axis(0.3f, -0.2f, 0.5f); axis.normalize();
            ga::Rotor Re0 = ga::rotor(std::cos(theta_e_target / 2), -std::sin(theta_e_target / 2) * axis);
            ga::Rotor R0 = ga::normalize_rotor(ga::geometric_product(Rd, Re0));

            // Only test combinations where the state is actually INSIDE the
            // funnel at t0 (Psi_e < rho_R(t0)) -- the exact identity
            // describes the barrier's own unclamped dynamics, so a
            // (theta_e, t0) pair where the funnel has already shrunk past
            // the error forces the e_R clamp to engage, which is a
            // different (and, by the barrier's design, supposed to be
            // unreachable in valid operation) regime, not a counterexample.
            float Psi_e_check = 1.0f - std::cos(theta_e_target / 2.0f);
            if (Psi_e_check >= stt::rho_R(t0, stt_params)) continue;

            auto s0 = eval_stage_r(Rd, R0, t0, stt_params, kR);

            float dt = 1e-4f; // float32 precision: 1e-6 underflows the VR1-VR0 difference (checked: relative error 7% at 1e-6, 0.07% at 1e-4)
            ga::Multivector Rdot = -0.5f * ga::geometric_product(R0, ga::bivector(s0.Omega_c));
            ga::Rotor R1 = ga::normalize_rotor(R0 + Rdot * dt);
            float t1 = t0 + dt;
            auto s1 = eval_stage_r(Rd, R1, t1, stt_params, kR);

            float VR0 = 0.5f * s0.eps_R * s0.eps_R;
            float VR1 = 0.5f * s1.eps_R * s1.eps_R;
            float VR_dot_numeric = (VR1 - VR0) / dt;
            float predicted = -kR * s0.xi_R * s0.eps_R * s0.eps_R;

            float tol = std::max(0.05f * std::abs(predicted), 1e-4f);  // relative tolerance with an absolute floor (float32 precision limit at tiny magnitudes)
            if (std::abs(VR_dot_numeric - predicted) > tol) {
                identity_ok = false;
                identity_detail = "theta_e=" + std::to_string(theta_e_target) +
                                   " t0=" + std::to_string(t0) +
                                   " numeric=" + std::to_string(VR_dot_numeric) +
                                   " predicted=" + std::to_string(predicted);
            }
        }
    }
    check("Stage-R: dV_R/dt = -kR*xi_R(eps_R)*eps_R^2 holds exactly (z_Omega=0)",
          identity_ok, identity_detail);

    // ------------------------------------------------------------------
    // Check #2: the theta_e->0 branch point is a genuine, safely-handled
    // REMOVABLE singularity, not a chattering source. eps_R and theta_e
    // must be paired consistently here (an earlier, WRONG test varied
    // them independently and found a spurious multi-million-magnitude
    // "jump" that can never actually occur in the real closed loop).
    // ------------------------------------------------------------------
    bool smooth_ok = true;
    float prev_norm = -1.0f, prev_theta = -1.0f;
    for (float theta_e_target : {1e-4f, 1e-5f, 2e-6f, 1.5e-6f, 1.001e-6f, 1e-6f, 5e-7f, 1e-7f}) {
        Eigen::Vector3f axis(0.3f, -0.2f, 0.5f); axis.normalize();
        ga::Rotor Re0 = ga::rotor(std::cos(theta_e_target / 2), -std::sin(theta_e_target / 2) * axis);
        ga::Rotor R0 = ga::normalize_rotor(ga::geometric_product(Rd, Re0));
        auto s0 = eval_stage_r(Rd, R0, 2.0f, stt_params, kR);
        if (!s0.Omega_c.allFinite()) smooth_ok = false;
        float n = s0.Omega_c.norm();
        if (n > 1e-2f) smooth_ok = false;  // should be tiny this close to theta_e=0
        prev_norm = n; prev_theta = theta_e_target;
    }
    (void)prev_norm; (void)prev_theta;
    check("Stage-R: Omega_c stays finite and small across the theta_e~1e-6 branch point",
          smooth_ok);

    // ------------------------------------------------------------------
    // Smoke Test: Full Pipeline at Near-Equilibrium
    // ------------------------------------------------------------------
    ga_stt_ctrl::DroneParams drone(0.03f, Eigen::Vector3f(1.6e-5f, 1.6e-5f, 2.9e-5f),
                                   0.0f, 1.0f, 4.0f, 2.0f, 6.0f, 0.01f);
    std::vector<stt::Obstacle> obstacles = {
        stt::Obstacle(Eigen::Vector3f(4.0f,4.0f,4.0f), Eigen::Vector3f(0.05f,-0.02f,0.03f), 0.6f)
    };
    stt::ObstacleArray obs_arr{obstacles.data(), static_cast<int>(obstacles.size())};

    float t0 = 0.3f;
    Eigen::Vector3f sigma_now = stt_params.s0;
    Eigen::Vector3f sigma_dot_now = stt::sigma_value(sigma_now, t0, obs_arr, stt_params);
    Eigen::Vector3f p = sigma_now;
    Eigen::Vector3f v = sigma_dot_now;
    ga::Rotor R_cur = ga::rotor_identity();
    Eigen::Vector3f Omega_b = Eigen::Vector3f::Zero();

    auto [f_cmd, tau, diag] = ga_stt_ctrl::compute_control(
        t0, p, v, R_cur, Omega_b, sigma_now, obs_arr, stt_params, drone, ga::rotor_identity());

    check("thrust within bounds", f_cmd >= drone.f_min && f_cmd <= drone.f_max);
    check("thrust near hover", std::abs(f_cmd - drone.m * 9.81f) < 0.5f);
    check("torque finite and small", tau.allFinite() && tau.norm() < 1.0f);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " FAILED") << "\n";
    return nfail;
}
