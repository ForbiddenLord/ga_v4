#pragma once
#include "jet1d.hpp"
#include <Eigen/Geometry>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

namespace stt {

// Constant-*acceleration* kinematic model (constant-velocity is the special
// case acc=0, which is why the old 3-arg constructor still works unchanged).
// Rationale: the controller needs sigma's Taylor jet up to 4th order to get
// Omega_d, Omega_dot_d analytically (PDF Sec. 3.2/Remark 1). With a
// constant-velocity obstacle model, obs_jets only ever has 2 nonzero rows,
// which silently truncates sigma's ODE-propagated jet at low order whenever
// an obstacle is "active" (inside rho_max). Feeding in a (possibly
// filtered/estimated) acceleration lets sigma_jet/rho_p_jet supply one more
// exact order before falling back to the zero-jerk assumption flagged in
// Remark 1 of the PDF.
struct Obstacle {
    Eigen::Vector3f pos0, vel, acc;
    float radius;
    Obstacle(const Eigen::Vector3f& p, const Eigen::Vector3f& v, float r)
        : pos0(p), vel(v), acc(Eigen::Vector3f::Zero()), radius(r) {}
    Obstacle(const Eigen::Vector3f& p, const Eigen::Vector3f& v, const Eigen::Vector3f& a, float r)
        : pos0(p), vel(v), acc(a), radius(r) {}
    Eigen::Vector3f position(float t) const { return pos0 + vel * t + 0.5f * acc * t * t; }
    Eigen::Vector3f velocity(float t) const { return vel + acc * t; }
};

struct STTParams {
    Eigen::Vector3f eta, s0;
    float tc, k1, k2, k3, rho_max, rho_min, u, rhoR0, rhoR_inf, kR;
    STTParams(const Eigen::Vector3f& eta_, const Eigen::Vector3f& s0_, float tc_,
              float k1_=0.3f, float k2_=1.0f, float k3_=0.5f, float rho_max_=1.0f,
              float rho_min_=0.15f, float u_=8.0f, float rhoR0_=1.2f,
              float rhoR_inf_=0.05f, float kR_=0.8f)
        : eta(eta_), s0(s0_), tc(tc_), k1(k1_), k2(k2_), k3(k3_), rho_max(rho_max_),
          rho_min(rho_min_), u(u_), rhoR0(rhoR0_), rhoR_inf(rhoR_inf_), kR(kR_) {}
};

constexpr int MAX_OBS = 2;

struct ObstacleArray {
    const Obstacle* data;
    int size;
};

inline jet::VectorJet vj_smooth(const jet::VectorJet& mj_jet) {
    Eigen::Vector3f m0 = mj_jet.row(0);
    Eigen::Vector3f m0n = m0 / (m0.norm() + 1e-6f);
    Eigen::Vector3f ref(0.0f, 0.0f, 1.0f);
    if (std::abs(m0n.z()) > 0.9f) ref = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

    jet::VectorJet ref_jet = jet::const_vector(ref, jet::MAX_JET_ORDER);
    jet::VectorJet raw = jet::cross_vv(mj_jet, ref_jet);
    jet::ScalarJet n = jet::norm_v(raw);

    jet::ScalarJet n_reg = n;
    n_reg[0] += 1e-6f;

    return jet::vec_div_s(raw, n_reg);
}

inline Eigen::Vector3f sigma_value(const Eigen::Vector3f& sigma, float t,
                                   const ObstacleArray& obstacles,
                                   const STTParams& params) {
    // Floor raised from 1e-3f (verified against a parallel delivery's
    // finding, cross-checked independently here): with a tight floor, the
    // prescribed-time gain k1*tc/(tc-t) grows very large near/at tc and,
    // since max(tc-t,floor) simply clamps once tc-t goes negative, STAYS
    // at that large value for all t>=tc -- a plausible root cause of
    // long-duration drift. Raising the floor (with no separate switched
    // law -- a two-law "hold" variant was checked and found to introduce
    // a discontinuity in sigma_dot at t=tc) keeps this exactly continuous
    // while bounding the gain to a sane value for all t, including
    // indefinitely past tc.
    float tc_minus_t = std::max(params.tc - t, std::max(0.5f, 0.02f * params.tc));
    float coef = params.k1 * (params.tc / tc_minus_t);
    Eigen::Vector3f sdot = coef * (params.eta - sigma);

    for (int j = 0; j < obstacles.size; ++j) {
        const auto& obs = obstacles.data[j];
        Eigen::Vector3f oj = obs.position(t);
        Eigen::Vector3f diff = sigma - oj;
        float dist = diff.norm();
        float dhat = dist - obs.radius;

        if (dhat > params.rho_max) continue;

        float denom = std::max(dist - (obs.radius + params.rho_min), 1e-6f);
        float denom3 = denom * denom * denom;
        Eigen::Vector3f mj = diff / denom3;

        Eigen::Vector3f m_hat = diff / std::max(dist, 1e-6f);
        Eigen::Vector3f ref(0.0f, 0.0f, 1.0f);
        if (std::abs(m_hat.z()) > 0.9f) ref = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

        Eigen::Vector3f vj_raw = m_hat.cross(ref);
        float vj_norm = vj_raw.norm();
        if (vj_norm < 1e-6f) {
            ref = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
            vj_raw = m_hat.cross(ref);
            vj_norm = vj_raw.norm();
        }
        Eigen::Vector3f vj = vj_raw / std::max(vj_norm, 1e-6f);

        float theta_j = (1.0f / std::max(dhat, 1e-6f)) - (1.0f / params.rho_max);
        sdot += (params.k2 * mj + params.k3 * vj) * theta_j;
    }
    return sdot;
}

inline float rho_p_value(const Eigen::Vector3f& sigma, float t,
                          const ObstacleArray& obstacles,
                          const STTParams& params) {
    if (obstacles.size == 0) return params.rho_max;
    float u = params.u;
    float sum_exp = 0.0f;

    for (int j = 0; j < obstacles.size; ++j) {
        const auto& obs = obstacles.data[j];
        Eigen::Vector3f oj = obs.position(t);
        float dist = (sigma - oj).norm();
        float dhat = dist - obs.radius;
        sum_exp += std::exp(-u * dhat);
    }

    float d = -std::log(sum_exp) / u;
    float inner = std::exp(-u * params.rho_max) + std::exp(-u * d);
    return -std::log(inner) / u;
}

inline jet::VectorJet sigma_jet(const Eigen::Vector3f& sigma0, float t0,
                                const ObstacleArray& obstacles,
                                const STTParams& params, int order) {
    jet::VectorJet c = jet::VectorJet::Zero();
    c.row(0) = sigma0;
    jet::ScalarJet t_jet = jet::const_scalar(t0, order);
    t_jet[1] = 1.0f;
    jet::VectorJet eta_jet = jet::const_vector(params.eta, order);

    std::array<std::pair<jet::VectorJet, jet::ScalarJet>, MAX_OBS> obs_jets;
    int num_obs = std::min(obstacles.size, MAX_OBS);

    for (int j = 0; j < num_obs; ++j) {
        const auto& obs = obstacles.data[j];
        // Taylor expansion of pos0 + vel*t0 + vel*dt + 0.5*acc*dt^2 about t0:
        // row0 = position(t0), row1 = velocity(t0), row2 = 0.5*acc (the k=2
        // Taylor coefficient), rows >=3 are 0 under the constant-acceleration
        // assumption (see Remark 1 of the derivation for the caveat this
        // still leaves at higher orders).
        jet::VectorJet oj = jet::const_vector(obs.position(t0), order);
        if (order >= 1) oj.row(1) = obs.velocity(t0);
        if (order >= 2) oj.row(2) = 0.5f * obs.acc;
        obs_jets[j] = {oj, jet::const_scalar(obs.radius, order)};
    }

    jet::ScalarJet tc_minus_t = jet::sub(jet::const_scalar(params.tc, order), t_jet);
    tc_minus_t[0] = std::max(tc_minus_t[0], std::max(0.5f, 0.02f * params.tc));
    jet::ScalarJet inv_tc = jet::div_ss(jet::const_scalar(params.tc, order), tc_minus_t);
    jet::ScalarJet coef = jet::mul_ss(inv_tc, jet::const_scalar(params.k1, order));

    for (int k = 0; k < order; ++k) {
        jet::VectorJet sigma_partial = c;
        jet::VectorJet term1 = jet::mul_sv(coef, jet::sub(eta_jet, sigma_partial));
        jet::VectorJet rhs = term1;

        for (int j = 0; j < num_obs; ++j) {
            const auto& [oj_jet, roj_jet] = obs_jets[j];
            jet::VectorJet diff_jet = jet::sub(sigma_partial, oj_jet);
            jet::ScalarJet dist_jet = jet::norm_v(diff_jet);

            if (dist_jet[0] - roj_jet[0] > params.rho_max) continue;

            jet::ScalarJet dhat_jet = jet::sub(dist_jet, roj_jet);
            jet::ScalarJet denom_jet = jet::sub(dist_jet, jet::add(roj_jet, jet::const_scalar(params.rho_min, order)));

            jet::ScalarJet denom3 = jet::mul_ss(jet::mul_ss(denom_jet, denom_jet), denom_jet);
            jet::VectorJet mj_jet = jet::vec_div_s(diff_jet, denom3);

            jet::VectorJet vj_jet = vj_smooth(mj_jet);

            jet::ScalarJet theta_jet = jet::sub(jet::div_ss(jet::const_scalar(1.0f, order), dhat_jet),
                                                jet::const_scalar(1.0f / params.rho_max, order));
            jet::VectorJet combo = jet::add(jet::mul_sv(jet::const_scalar(params.k2, order), mj_jet),
                                            jet::mul_sv(jet::const_scalar(params.k3, order), vj_jet));
            rhs = jet::add(rhs, jet::mul_sv(theta_jet, combo));
        }
        c.row(k + 1) = rhs.row(k) / (float)(k + 1);
    }
    return c;
}

inline jet::ScalarJet rho_p_jet(const jet::VectorJet& sigma_jet, float t0,
                                const ObstacleArray& obstacles,
                                const STTParams& params, int order) {
    if (obstacles.size == 0) return jet::const_scalar(params.rho_max, order);
    jet::ScalarJet sum_exp = jet::const_scalar(0.0f, order);

    int num_obs = std::min(obstacles.size, MAX_OBS);
    for (int j = 0; j < num_obs; ++j) {
        const auto& obs = obstacles.data[j];
        jet::VectorJet oj = jet::const_vector(obs.position(t0), order);
        if (order >= 1) oj.row(1) = obs.velocity(t0);
        if (order >= 2) oj.row(2) = 0.5f * obs.acc;
        jet::ScalarJet roj = jet::const_scalar(obs.radius, order);
        jet::ScalarJet dhat = jet::sub(jet::norm_v(jet::sub(sigma_jet, oj)), roj);
        sum_exp = jet::add(sum_exp, jet::exp_s(jet::mul_ss(jet::const_scalar(-params.u, order), dhat)));
    }
    jet::ScalarJet d_jet = jet::mul_ss(jet::const_scalar(-1.0f / params.u, order), jet::log_s(sum_exp));
    jet::ScalarJet inner = jet::add(jet::const_scalar(std::exp(-params.u * params.rho_max), order),
                                    jet::exp_s(jet::mul_ss(jet::const_scalar(-params.u, order), d_jet)));
    return jet::mul_ss(jet::const_scalar(-1.0f / params.u, order), jet::log_s(inner));
}

inline float rho_R(float t, const STTParams& p) {
    return (p.rhoR0 - p.rhoR_inf) * std::exp(-p.kR * t) + p.rhoR_inf;
}

inline float rho_R_dot(float t, const STTParams& p) {
    return -p.kR * (p.rhoR0 - p.rhoR_inf) * std::exp(-p.kR * t);
}

} // namespace stt