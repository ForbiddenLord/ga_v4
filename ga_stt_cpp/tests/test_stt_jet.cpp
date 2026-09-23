#include <iostream>
#include <cmath>
#include <vector>
#include "stt.hpp"

int nfail = 0;
void check(const std::string& name, bool cond) {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name << "\n";
    if (!cond) nfail++;
}

// Double-precision ground truth.
namespace ref {

struct ObstacleD {
    Eigen::Vector3d pos0, vel;
    double radius;
    Eigen::Vector3d position(double t) const { return pos0 + vel * t; }
};

struct ParamsD {
    Eigen::Vector3d eta, s0;
    double tc, k1, k2, k3, rho_max, rho_min, u;
};

inline Eigen::Vector3d sigma_value_d(const Eigen::Vector3d& sigma, double t,
                                      const std::vector<ObstacleD>& obstacles,
                                      const ParamsD& params) {
    double tc_minus_t = std::max(params.tc - t, 1e-3);
    double coef = params.k1 * (params.tc / tc_minus_t);
    Eigen::Vector3d sdot = coef * (params.eta - sigma);

    for (const auto& obs : obstacles) {
        Eigen::Vector3d oj = obs.position(t);
        Eigen::Vector3d diff = sigma - oj;
        double dist = diff.norm();
        double dhat = dist - obs.radius;
        if (dhat > params.rho_max) continue;

        double denom = std::max(dist - (obs.radius + params.rho_min), 1e-9);
        double denom3 = denom * denom * denom;
        Eigen::Vector3d mj = diff / denom3;

        Eigen::Vector3d m_hat = diff / std::max(dist, 1e-9);
        Eigen::Vector3d ref_axis(0.0, 0.0, 1.0);
        if (std::abs(m_hat.z()) > 0.9) ref_axis = Eigen::Vector3d(1.0, 0.0, 0.0);

        Eigen::Vector3d vj_raw = m_hat.cross(ref_axis);
        double vj_norm = vj_raw.norm();
        if (vj_norm < 1e-9) {
            ref_axis = Eigen::Vector3d(0.0, 1.0, 0.0);
            vj_raw = m_hat.cross(ref_axis);
            vj_norm = vj_raw.norm();
        }
        Eigen::Vector3d vj = vj_raw / std::max(vj_norm, 1e-9);

        double theta_j = (1.0 / std::max(dhat, 1e-9)) - (1.0 / params.rho_max);
        sdot += (params.k2 * mj + params.k3 * vj) * theta_j;
    }
    return sdot;
}

inline Eigen::Vector3d rk4_integrate(const Eigen::Vector3d& s0, double t0, double t_end,
                                      const std::vector<ObstacleD>& obs, const ParamsD& p) {
    double t = t0;
    Eigen::Vector3d s = s0;
    double dt = 1e-6;
    int steps = std::max(1, (int)std::round(std::abs(t_end - t0) / dt));
    dt = (t_end - t0) / steps;
    auto f = [&](double tt, const Eigen::Vector3d& ss) { return sigma_value_d(ss, tt, obs, p); };
    for (int i = 0; i < steps; ++i) {
        auto k1 = f(t, s);
        auto k2 = f(t + dt/2.0, s + dt/2.0 * k1);
        auto k3 = f(t + dt/2.0, s + dt/2.0 * k2);
        auto k4 = f(t + dt, s + dt * k3);
        s += (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        t += dt;
    }
    return s;
}

inline double rho_p_value_d(const Eigen::Vector3d& sigma, double t,
                             const std::vector<ObstacleD>& obstacles, const ParamsD& params) {
    if (obstacles.empty()) return params.rho_max;
    double u = params.u;
    double sum_exp = 0.0;
    for (const auto& obs : obstacles) {
        Eigen::Vector3d oj = obs.position(t);
        double dist = (sigma - oj).norm();
        double dhat = dist - obs.radius;
        sum_exp += std::exp(-u * dhat);
    }
    double d = -std::log(sum_exp) / u;
    double inner = std::exp(-u * params.rho_max) + std::exp(-u * d);
    return -std::log(inner) / u;
}

} // namespace ref

int main() {
    stt::STTParams params(Eigen::Vector3f(8.0f,8.0f,8.0f), Eigen::Vector3f(1.0f,1.0f,1.0f), 16.0f,
                          0.3f, 1.0f, 0.5f, 1.0f, 0.15f, 8.0f);
    std::vector<stt::Obstacle> obstacles = {
        stt::Obstacle(Eigen::Vector3f(4.0f,4.0f,4.0f), Eigen::Vector3f(0.05f,-0.02f,0.03f), 0.6f),
        stt::Obstacle(Eigen::Vector3f(3.0f,5.0f,4.5f), Eigen::Vector3f(-0.03f,0.04f,0.0f), 0.5f)
    };
    stt::ObstacleArray obs_arr{obstacles.data(), static_cast<int>(obstacles.size())};

    ref::ParamsD paramsD{Eigen::Vector3d(8.0,8.0,8.0), Eigen::Vector3d(1.0,1.0,1.0), 16.0,
                          0.3, 1.0, 0.5, 1.0, 0.15, 8.0};
    std::vector<ref::ObstacleD> obstaclesD = {
        {Eigen::Vector3d(4.0,4.0,4.0), Eigen::Vector3d(0.05,-0.02,0.03), 0.6},
        {Eigen::Vector3d(3.0,5.0,4.5), Eigen::Vector3d(-0.03,0.04,0.0), 0.5}
    };

    double t0d = 6.37;
    float t0 = (float)t0d;

    Eigen::Vector3d sigma0d = ref::rk4_integrate(paramsD.s0, 0.0, t0d, obstaclesD, paramsD);
    Eigen::Vector3f sigma0 = sigma0d.cast<float>();

    int order = 3;
    auto sjet = stt::sigma_jet(sigma0, t0, obs_arr, params, order);

    auto sigma_of_t = [&](double t) { return ref::rk4_integrate(paramsD.s0, 0.0, t, obstaclesD, paramsD); };

    double h = 1e-4;
    auto s0_t = sigma_of_t(t0d);
    auto s_p1 = sigma_of_t(t0d+h), s_m1 = sigma_of_t(t0d-h);
    auto s_p2 = sigma_of_t(t0d+2.0*h), s_m2 = sigma_of_t(t0d-2.0*h);

    Eigen::Vector3d d1_fd = (-s_p2 + 8.0*s_p1 - 8.0*s_m1 + s_m2) / (12.0*h);
    Eigen::Vector3d d2_fd = (-s_p2 + 16.0*s_p1 - 30.0*s0_t + 16.0*s_m1 - s_m2) / (12.0*h*h);
    Eigen::Vector3d d3_fd = (s_p2 - 2.0*s_p1 + 2.0*s_m1 - s_m2) / (2.0*h*h*h);

    Eigen::Vector3d d1_jet = sjet.row(1).transpose().cast<double>();
    Eigen::Vector3d d2_jet = 2.0 * sjet.row(2).transpose().cast<double>();
    Eigen::Vector3d d3_jet = 6.0 * sjet.row(3).transpose().cast<double>();

    check("sigma(t0) matches", (sjet.row(0).transpose().cast<double>() - sigma0d).norm() < 1e-4);
    check("sigma_dot matches FD", (d1_jet - d1_fd).norm() < 1e-3);
    check("sigma_ddot matches FD", (d2_jet - d2_fd).norm() < 1e-2);
    check("sigma_dddot matches FD", (d3_jet - d3_fd).norm() < 5e-2);

    auto rjet = stt::rho_p_jet(sjet, t0, obs_arr, params, order);
    double rp_d1 = (double)rjet[1];

    auto rho_eval = [&](double t) { return ref::rho_p_value_d(sigma_of_t(t), t, obstaclesD, paramsD); };
    double rp_d1_fd = (rho_eval(t0d+h) - rho_eval(t0d-h)) / (2.0*h);
    check("rho_p_dot matches FD", std::abs(rp_d1 - rp_d1_fd) < 1e-2);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " FAILED") << "\n";
    return nfail;
}