#include <iostream>
#include <cmath>
#include <vector>
#include "ga3.hpp"
#include "jet1d.hpp"
#include "stt.hpp"
#include "ga_stt_controller.hpp"

int nfail = 0;
void check(const std::string& name, bool cond, const std::string& extra = "") {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name;
    if (!cond) { nfail++; std::cout << " " << extra; }
    std::cout << "\n";
}

// Evaluate the m-th derivative (m=0,1,2,...) of the Taylor polynomial stored
// in a vector jet, at absolute time t (t0 = expansion point).
Eigen::Vector3f eval_vec_jet_deriv(const jet::VectorJet& c, int m, float t, float t0) {
    float dt = t - t0;
    Eigen::Vector3f val = Eigen::Vector3f::Zero();
    for (int k = m; k < c.rows(); ++k) {
        float coeff = 1.0f;
        for (int i = 0; i < m; ++i) coeff *= (float)(k - i);
        float dt_pow = (k - m == 0) ? 1.0f : std::pow(dt, (float)(k - m));
        val += coeff * c.row(k).transpose() * dt_pow;
    }
    return val;
}

float eval_scalar_jet_deriv(const jet::ScalarJet& c, int m, float t, float t0) {
    float dt = t - t0;
    float val = 0.0f;
    for (int k = m; k < c.size(); ++k) {
        float coeff = 1.0f;
        for (int i = 0; i < m; ++i) coeff *= (float)(k - i);
        float dt_pow = (k - m == 0) ? 1.0f : std::pow(dt, (float)(k - m));
        val += coeff * c[k] * dt_pow;
    }
    return val;
}

int main() {
    stt::STTParams stt_params(Eigen::Vector3f(8,8,8), Eigen::Vector3f(1,1,1), 16.0f,
                              0.3f, 1.0f, 0.5f, 1.0f, 0.15f, 8.0f);
    std::vector<stt::Obstacle> obstacles_vec = {
        stt::Obstacle(Eigen::Vector3f(4,4,4), Eigen::Vector3f(0.05f,-0.02f,0.03f), 0.6f),
        stt::Obstacle(Eigen::Vector3f(3,5,4.5f), Eigen::Vector3f(-0.03f,0.04f,0.0f), 0.5f)
    };
    stt::ObstacleArray obstacles{obstacles_vec.data(), static_cast<int>(obstacles_vec.size())};

    // Gains chosen so the accel_cap safety clamp (see resolve_translational_chain)
    // is NOT triggered here -- that clamp's own behavior is exercised by
    // test_closed_loop_convergence; this test isolates and validates the
    // underlying (uncapped) Picard/Taylor ODE-propagation arithmetic itself,
    // which a capped finite-difference ground truth can't cleanly verify
    // (clamping a Taylor *coefficient* is not the same operation as
    // clamping the function's *value* before finite-differencing it).
    ga_stt_ctrl::DroneParams drone(0.03f, Eigen::Vector3f(1.6e-5f, 1.6e-5f, 2.9e-5f),
                                   0.0f, 1.0f, 1.0f, 1.0f, 6.0f, 0.01f);

    float t0 = 3.1f;
    // sigma0 placed with NO active obstacle nearby (well outside rho_max of
    // both obstacles): this isolates and validates the Picard/Taylor
    // ODE-propagation arithmetic on its own smooth, well-conditioned
    // terrain. Even far from the tube-boundary singularity, Das et al.'s
    // repulsion term is steep close to an obstacle regardless of kappa1
    // (verified: even kappa1=1 still saturated the accel_cap for a sigma0
    // near an active obstacle) -- that capped regime is intentionally
    // exercised instead by test_closed_loop_convergence, where it's the
    // right behavior, not a bug to chase here.
    Eigen::Vector3f sigma0(1.5f, 1.5f, 1.5f);
    Eigen::Vector3f p0 = sigma0 + Eigen::Vector3f(0.05f, -0.03f, 0.02f);
    Eigen::Vector3f v0(0.2f, -0.1f, 0.05f);

    auto sjet = stt::sigma_jet(sigma0, t0, obstacles, stt_params, ga_stt_ctrl::JET_ORDER_SIGMA);
    auto rjet = stt::rho_p_jet(sjet, t0, obstacles, stt_params, ga_stt_ctrl::JET_ORDER_SIGMA);

    // New signature: no R_cur/Omega_b_cur -- the reference generator is now
    // a pure function of (p, v, t, sigma-jet, rho_p-jet), see corrigendum
    // in ga_stt_controller.hpp.
    auto [Fd_jet, fd_jet, that_d_jet] = ga_stt_ctrl::resolve_translational_chain(
        t0, p0, v0, sjet, rjet, drone);

    constexpr float g = 9.81f;
    const Eigen::Vector3f E3(0.0f, 0.0f, 1.0f);
    float m = drone.m;

    // --- Ground truth: u1(t, p, v) evaluated directly against the SAME
    // Taylor-expanded sigma/rho_p polynomials the jet code uses (this tests
    // whether resolve_translational_chain's Picard/Taylor propagation of the
    // p_dot=v, v_dot=u1(p,v,t) ODE is correct -- not whether the STT
    // generator itself is accurate far from t0, which is a separate,
    // already-covered concern in test_stt_jet.cpp). ---
    // NOTE: explicit trailing return type is required here -- without it the
    // lambda's return type is deduced from the Eigen expression template
    // (CwiseBinaryOp chain), which lazily references this lambda's *local*
    // variables (sigma_ddot_t, z_v, p_hat, ...) rather than a materialized
    // Vector3f. Combined with the nested RK4 lambdas below this produced a
    // genuine stack-use-after-return (confirmed with
    // -fsanitize=address,undefined) and intermittently wrong values -- a
    // bug in this test harness, not in the production u1_jet/
    // resolve_translational_chain code, which materializes every
    // intermediate as jet::VectorJet/ScalarJet throughout.
    auto u1_value = [&](float t, const Eigen::Vector3f& p, const Eigen::Vector3f& v) -> Eigen::Vector3f {
        Eigen::Vector3f sigma_t = eval_vec_jet_deriv(sjet, 0, t, t0);
        Eigen::Vector3f sigma_dot_t = eval_vec_jet_deriv(sjet, 1, t, t0);
        Eigen::Vector3f sigma_ddot_t = eval_vec_jet_deriv(sjet, 2, t, t0);
        float rho_t = eval_scalar_jet_deriv(rjet, 0, t, t0);

        Eigen::Vector3f p_tilde = p - sigma_t;
        float norm_pt = std::sqrt(p_tilde.squaredNorm() + 1e-6f);
        float e1 = norm_pt / rho_t;
        float eps_p = std::log((1.0f + e1) / (1.0f - e1));
        Eigen::Vector3f p_hat = p_tilde / norm_pt;
        float xi_p_over_rho = 2.0f / (rho_t * (1.0f - e1 * e1));

        Eigen::Vector3f z_v = v - sigma_dot_t;
        Eigen::Vector3f u1 = sigma_ddot_t - drone.kappa_v * z_v - drone.kappa1 * xi_p_over_rho * eps_p * p_hat;
        // Match the production code's accel_cap exactly (see
        // resolve_translational_chain) so this ground truth is comparing
        // the same, well-posed function the jet is actually Taylor-
        // expanding, not an unbounded one.
        float accel_cap = 4.0f * drone.f_max / drone.m + 4.0f * 9.81f;
        float n = u1.norm();
        if (n > accel_cap) u1 *= (accel_cap / n);
        return u1;
    };

    auto Fd_value = [&](const Eigen::Vector3f& p, const Eigen::Vector3f& v, float t) -> Eigen::Vector3f {
        return m * (u1_value(t, p, v) + g * E3);
    };

    // RK4-integrate the SAME ODE (p_dot=v, v_dot=u1) that the jet chain is
    // implicitly Taylor-expanding, then finite-difference the resulting
    // Fd(t) trajectory as an independent ground truth for Fd_dot, Fd_ddot.
    auto rk4_traj = [&](float t_target) {
        int n = std::max(1, (int)(std::abs(t_target - t0) / 1e-4f));
        float dt = (t_target - t0) / n;
        Eigen::Vector3f p = p0, v = v0;
        float t = t0;
        for (int i = 0; i < n; ++i) {
            auto deriv = [&](const Eigen::Vector3f& pp, const Eigen::Vector3f& vv, float tt) {
                return std::make_pair(vv, u1_value(tt, pp, vv));
            };
            auto [k1p, k1v] = deriv(p, v, t);
            auto [k2p, k2v] = deriv(p + k1p * (dt / 2), v + k1v * (dt / 2), t + dt / 2);
            auto [k3p, k3v] = deriv(p + k2p * (dt / 2), v + k2v * (dt / 2), t + dt / 2);
            auto [k4p, k4v] = deriv(p + k3p * dt, v + k3v * dt, t + dt);
            p += (dt / 6.0f) * (k1p + 2 * k2p + 2 * k3p + k4p);
            v += (dt / 6.0f) * (k1v + 2 * k2v + 2 * k3v + k4v);
            t += dt;
        }
        return std::make_pair(p, v);
    };

    float h = 2e-3f;
    auto [p_plus, v_plus] = rk4_traj(t0 + h);
    auto [p_minus, v_minus] = rk4_traj(t0 - h);
    Eigen::Vector3f Fd0_direct = Fd_value(p0, v0, t0);
    Eigen::Vector3f Fd_plus = Fd_value(p_plus, v_plus, t0 + h);
    Eigen::Vector3f Fd_minus = Fd_value(p_minus, v_minus, t0 - h);
    Eigen::Vector3f Fd_dot_direct = (Fd_plus - Fd_minus) / (2.0f * h);
    Eigen::Vector3f Fd_ddot_direct = (Fd_plus - 2.0f * Fd0_direct + Fd_minus) / (h * h);

    check("Fd(t0) is finite and positive",
          std::isfinite(Fd_jet.row(0).norm()) && Fd_jet.row(0).norm() > 0.0f);
    check("Fd(t0) matches direct evaluation",
          (Fd_jet.row(0).transpose() - Fd0_direct).norm() < 1e-3f,
          "jet=" + std::to_string(Fd_jet.row(0).norm()) + " direct=" + std::to_string(Fd0_direct.norm()));
    check("Fd_dot(t0) is finite", std::isfinite(Fd_jet.row(1).norm()));
    check("Fd_dot(t0) matches RK4 finite-difference ground truth",
          (Fd_jet.row(1).transpose() - Fd_dot_direct).norm() < 5e-2f * std::max(1.0f, Fd_dot_direct.norm()),
          "jet=" + std::to_string(Fd_jet.row(1).norm()) + " fd=" + std::to_string(Fd_dot_direct.norm()));
    check("Fd_ddot(t0)/2 matches RK4 finite-difference ground truth",
          (2.0f * Fd_jet.row(2).transpose() - Fd_ddot_direct).norm() < 5e-1f * std::max(1.0f, Fd_ddot_direct.norm()),
          "jet=" + std::to_string(2.0f * Fd_jet.row(2).norm()) + " fd=" + std::to_string(Fd_ddot_direct.norm()));

    check("thrust direction is a unit vector",
          std::abs(that_d_jet.row(0).norm() - 1.0f) < 1e-3f);
    // Note: resolve_translational_chain itself does not clamp (compute_control
    // does, via f_command); here we just check the unclamped magnitude is sane.
    check("commanded thrust magnitude positive and finite",
          std::isfinite(fd_jet[0]) && fd_jet[0] > 0.0f);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " TEST(S) FAILED") << "\n";
    return nfail == 0 ? 0 : 1;
}
