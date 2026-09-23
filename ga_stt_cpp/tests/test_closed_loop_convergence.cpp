#include <iostream>
#include <vector>
#include <cmath>
#include "ga3.hpp"                 
#include "stt.hpp"                 
#include "ga_stt_controller.hpp"   

// Test-only helpers
namespace test_utils {
inline ga::Rotor exp_bivector(const Eigen::Vector3f& axis3, float theta) {
    float c = std::cos(theta / 2.0f);
    float s = std::sin(theta / 2.0f);
    return ga::rotor(c, -s * axis3);
}
}

int nfail = 0;
void check(const std::string& name, bool cond, const std::string& extra = "") {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name;
    if (!cond) { nfail++; std::cout << " " << extra; }
    std::cout << "\n";
}

struct QuadState {
    Eigen::Vector3f p, v, Omega_b;
    ga::Rotor R;
};

QuadState derivative(const QuadState& s, float f, const Eigen::Vector3f& tau,
                     const ga_stt_ctrl::DroneParams& drone) {
    constexpr float g = 9.81f;
    const Eigen::Vector3f E3(0.0f, 0.0f, 1.0f);
    QuadState ds;
    ds.p = s.v;
    ds.v = (1.0f / drone.m) * (-drone.m * g * E3 + f * ga::sandwich(s.R, E3));
    ds.R = -0.5f * ga::geometric_product(s.R, ga::bivector(s.Omega_b));
    ds.Omega_b = drone.J_diag.cwiseInverse().cwiseProduct(tau - s.Omega_b.cross(drone.J_diag.cwiseProduct(s.Omega_b)));
    return ds;
}

QuadState rk4_step(const QuadState& s, float dt, float f, const Eigen::Vector3f& tau,
                   const ga_stt_ctrl::DroneParams& drone) {
    auto k1 = derivative(s, f, tau, drone);
    QuadState s2 = s;
    s2.p += k1.p * (dt/2.0f); s2.v += k1.v * (dt/2.0f);
    s2.R = ga::normalize_rotor(s2.R + k1.R * (dt/2.0f)); s2.Omega_b += k1.Omega_b * (dt/2.0f);
    auto k2 = derivative(s2, f, tau, drone);
    
    QuadState s3 = s;
    s3.p += k2.p * (dt/2.0f); s3.v += k2.v * (dt/2.0f);
    s3.R = ga::normalize_rotor(s3.R + k2.R * (dt/2.0f)); s3.Omega_b += k2.Omega_b * (dt/2.0f);
    auto k3 = derivative(s3, f, tau, drone);
    
    QuadState s4 = s;
    s4.p += k3.p * dt; s4.v += k3.v * dt;
    s4.R = ga::normalize_rotor(s4.R + k3.R * dt); s4.Omega_b += k3.Omega_b * dt;
    auto k4 = derivative(s4, f, tau, drone);
    
    QuadState s_new = s;
    s_new.p += (dt/6.0f) * (k1.p + 2.0f*k2.p + 2.0f*k3.p + k4.p);
    s_new.v += (dt/6.0f) * (k1.v + 2.0f*k2.v + 2.0f*k3.v + k4.v);
    s_new.R = ga::normalize_rotor(s_new.R + (dt/6.0f) * (k1.R + 2.0f*k2.R + 2.0f*k3.R + k4.R));
    s_new.Omega_b += (dt/6.0f) * (k1.Omega_b + 2.0f*k2.Omega_b + 2.0f*k3.Omega_b + k4.Omega_b);
    return s_new;
}

int main() {
    // Note: float literals used for STTParams and DroneParams
    stt::STTParams stt_params(Eigen::Vector3f(8,8,8), Eigen::Vector3f(1,1,1), 16.0f,
                              0.15f, 1.0f, 0.5f, 1.0f, 0.15f, 8.0f, 1.0f, 0.05f, 0.5f);
    
    // Added kappa_v (translational velocity-damping gain, see corrigendum in
    // ga_stt_controller.hpp): without it this second-order position loop is
    // an undamped nonlinear oscillator, not an asymptotically stable one.
    // Added kappa_v (translational velocity-damping gain, see corrigendum in
    // ga_stt_controller.hpp): without it this second-order position loop is
    // an undamped nonlinear oscillator, not an asymptotically stable one.
    // kR retuned up from the original test's 3.0 to 8.0: with the sign
    ///frame-rotation fixes in the attitude law (see corrigendum), kR=3.0
    // still converges but overshoots before settling (theta_e rises from
    // 0.59 to 0.75 rad in the first second, only decaying after); kR=8.0
    // (kOmega raised slightly to 0.2 to match) gives clean, fast, monotonic
    // convergence (0.59 -> 0.02 rad within 1s) with the same test scenario.
    ga_stt_ctrl::DroneParams drone(0.5f, Eigen::Vector3f(3.0e-3f, 3.0e-3f, 5.5e-3f),
                                   0.0f, 12.0f, 1.5f, 2.0f, 8.0f, 0.2f);

    std::vector<stt::Obstacle> obstacles_vec = {
        stt::Obstacle(Eigen::Vector3f(4,4,4), Eigen::Vector3f(0.05f,-0.02f,0.03f), 0.6f),
        stt::Obstacle(Eigen::Vector3f(3,5,4.5f), Eigen::Vector3f(-0.03f,0.04f,0.0f), 0.5f)
    };
    stt::ObstacleArray obstacles{obstacles_vec.data(), static_cast<int>(obstacles_vec.size())};

    QuadState state;
    state.p = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
    state.v = Eigen::Vector3f::Zero();
    state.R = test_utils::exp_bivector(Eigen::Vector3f(0.3f, 0.5f, 0.1f).normalized(), 0.6f); 
    state.Omega_b = Eigen::Vector3f::Zero();

    Eigen::Vector3f sigma = stt_params.s0;
    float dt = 0.002f;
    int total_steps = 4000; // 8 seconds
    
    float theta_e_start = 0, theta_e_1s = 0, theta_e_4s = 0, theta_e_8s = 0;
    float max_ep = 0.0f;
    bool has_nan = false;

    for (int i = 0; i <= total_steps; ++i) {
        float t = i * dt;

        // Call controller (Updated namespace)
        auto [f_cmd, tau, diag] = ga_stt_ctrl::compute_control(
            t, state.p, state.v, state.R, state.Omega_b, sigma,
            obstacles, stt_params, drone, ga::rotor_identity());

        if (i == 0) theta_e_start = diag.theta_e;
        if (i == (int)(1.0f/dt)) theta_e_1s = diag.theta_e;
        if (i == (int)(4.0f/dt)) theta_e_4s = diag.theta_e;
        if (i == total_steps) theta_e_8s = diag.theta_e;

        if (!std::isfinite(f_cmd) || !tau.allFinite()) { has_nan = true; break; }

        // Calculate e_p
        float u = stt_params.u;
        float d_min = 1e9f;
        for(const auto& obs : obstacles_vec) {
            float dist = (sigma - obs.position(t)).norm();
            d_min = std::min(d_min, dist - obs.radius);
        }
        float rho_p = stt::rho_p_value(sigma, t, obstacles, stt_params);
        float ep = (state.p - sigma).norm() / rho_p;
        max_ep = std::max(max_ep, ep);

        // Advance sigma
        auto sjet = stt::sigma_jet(sigma, t, obstacles, stt_params, 1);
        Eigen::Vector3f sigma_dot = sjet.row(1).transpose();
        sigma += sigma_dot * dt;

        // Advance state
        state = rk4_step(state, dt, f_cmd, tau, drone);
    }

    std::cout << "theta_e: start=" << theta_e_start << ", t=1s=" << theta_e_1s
              << ", t=4s=" << theta_e_4s << ", t=8s=" << theta_e_8s << "\n";
    std::cout << "e_p: max over run = " << max_ep << "\n";

    // Relaxed tolerances for float
    check("attitude error decreases substantially within 1s", theta_e_1s < 0.5f * theta_e_start);
    check("attitude error stays bounded", theta_e_1s < 1.5f && theta_e_4s < 1.5f);
    check("attitude error eventually converges by t=8s", theta_e_8s < 0.05f);
    check("position stays strictly inside the translational tube (e_p < 1)", max_ep < 1.0f,
          "max_ep=" + std::to_string(max_ep));
    check("no NaN/divergence over the full 8s run", !has_nan);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " TEST(S) FAILED") << "\n";
    return nfail == 0 ? 0 : 1;
}