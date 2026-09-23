#include <iostream>
#include <cmath>
#include "ga3.hpp"
#include "jet1d.hpp"
#include "rotor_jet.hpp"

int nfail = 0;
void check(const std::string& name, bool cond) {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name << "\n";
    if (!cond) nfail++;
}

// Float side: builds the inputs fed to the jet code under test
namespace test_utils {
inline ga::Rotor align_rotor(const Eigen::Vector3f& t_p, const Eigen::Vector3f& t_d) {
    float dot = t_d.dot(t_p);
    Eigen::Vector3f cross_tp = t_d.cross(t_p);
    ga::Multivector N = ga::Multivector::Zero();
    N[0] = 1.0f + dot;
    N[6] = cross_tp[0]; N[5] = -cross_tp[1]; N[3] = cross_tp[2];
    float D = std::sqrt(std::max(2.0f * (1.0f + dot), 1e-6f));
    return N / D;
}
}

Eigen::Vector3f td_raw(float t) {
    Eigen::Vector3f v(std::sin(t) + 0.3f*std::sin(3.0f*t), std::cos(t), 1.5f + 0.2f*std::cos(2.0f*t));
    return v.normalized();
}

Eigen::Vector3f tp_raw(float t) {
    Eigen::Vector3f v(0.1f*t, 0.0f, 1.0f);
    return v.normalized();
}

ga::Rotor Rd_direct(float t) {
    return test_utils::align_rotor(tp_raw(t), td_raw(t));
}

// Double side: independent ground truth.
namespace ref {

using MultivectorD = Eigen::Matrix<double, 8, 1>;
using RotorD = MultivectorD;

inline MultivectorD geometric_product_d(const MultivectorD& A, const MultivectorD& B) {
    MultivectorD out = MultivectorD::Zero();
    for (int a = 0; a < 8; ++a) {
        if (A[a] == 0.0) continue;
        for (int b = 0; b < 8; ++b) {
            if (B[b] == 0.0) continue;
            out[ga::STRUCT_IDX[a][b]] += ga::STRUCT_SIGN[a][b] * A[a] * B[b];
        }
    }
    return out;
}
inline RotorD reverse_d(const RotorD& R) {
    RotorD rev = R;
    rev[3] = -R[3]; rev[5] = -R[5]; rev[6] = -R[6]; rev[7] = -R[7];
    return rev;
}
inline Eigen::Vector3d to_bivector3_d(const MultivectorD& M) {
    return Eigen::Vector3d(M[6], -M[5], M[3]);
}

inline Eigen::Vector3d td_raw_d(double t) {
    Eigen::Vector3d v(std::sin(t) + 0.3*std::sin(3.0*t), std::cos(t), 1.5 + 0.2*std::cos(2.0*t));
    return v.normalized();
}
inline Eigen::Vector3d tp_raw_d(double t) {
    Eigen::Vector3d v(0.1*t, 0.0, 1.0);
    return v.normalized();
}
inline RotorD align_rotor_d(const Eigen::Vector3d& tp, const Eigen::Vector3d& td) {
    double dot = td.dot(tp);
    Eigen::Vector3d cross_tp = td.cross(tp);
    MultivectorD N = MultivectorD::Zero();
    N[0] = 1.0 + dot;
    N[6] = cross_tp[0]; N[5] = -cross_tp[1]; N[3] = cross_tp[2];
    double D = std::sqrt(std::max(2.0*(1.0+dot), 1e-12));
    return N / D;
}
inline RotorD Rd_direct_d(double t) { return align_rotor_d(tp_raw_d(t), td_raw_d(t)); }

inline Eigen::Vector3d Omega_d(double t, double h) {
    RotorD Rp = Rd_direct_d(t + h/2.0);
    RotorD Rm = Rd_direct_d(t - h/2.0);
    RotorD Rdot = (Rp - Rm) / h;
    RotorD Rval = Rd_direct_d(t);
    MultivectorD Om = -2.0 * geometric_product_d(reverse_d(Rval), Rdot);
    return to_bivector3_d(Om);
}

} // namespace ref

int main() {
    float t0 = 1.234f;
    int order = 3;
    float h = 1e-3f;

    auto build_vec_jet = [&](auto fn, float t0, int order) {
        jet::VectorJet c = jet::VectorJet::Zero();
        c.row(0) = fn(t0);
        if (order >= 1) c.row(1) = (fn(t0+h) - fn(t0-h)) / (2.0f*h);
        if (order >= 2) c.row(2) = (fn(t0+h) - 2.0f*fn(t0) + fn(t0-h)) / (h*h);
        if (order >= 3) c.row(3) = (fn(t0+2.0f*h) - 2.0f*fn(t0+h) + 2.0f*fn(t0-h) - fn(t0-2.0f*h)) / (2.0f*h*h*h);
        c.row(1) /= 1.0f; c.row(2) /= 2.0f; c.row(3) /= 6.0f;
        return c;
    };

    auto td_jet = build_vec_jet(td_raw, t0, order);
    auto tp_jet = build_vec_jet(tp_raw, t0, order);
    auto Ra_jet = rotor_jet::align(tp_jet, td_jet);

    ga::Rotor Ra0 = ga::rotor(Ra_jet.s[0], Ra_jet.b.row(0).transpose());
    ga::Rotor Rd0_direct = Rd_direct(t0);

    if ((Ra0 - Rd0_direct).norm() > 1e-5f) {
        check("Ra_jet value matches direct", (Ra0 + Rd0_direct).norm() < 1e-5f);
    } else {
        check("Ra_jet value matches direct", true);
    }

    rotor_jet::RotorJet Rd_jet = Ra_jet;
    auto Rd_rev = rotor_jet::reverse(Rd_jet);
    auto Rd_dot = rotor_jet::differentiate(Rd_jet);
    auto Omega_d_full = rotor_jet::mul(Rd_rev, Rd_dot);

    Eigen::Vector3f Omega_d_value = -2.0f * Omega_d_full.b.row(0).transpose();
    Eigen::Vector3f Omega_dot_d_value = -2.0f * Omega_d_full.b.row(1).transpose();

    double t0d = (double)t0;
    Eigen::Vector3d Om_fd = ref::Omega_d(t0d, 1e-5);
    Eigen::Vector3d Om_dot_fd = (ref::Omega_d(t0d + 1e-4, 1e-5) - ref::Omega_d(t0d - 1e-4, 1e-5)) / (2.0*1e-4);

    check("Omega_d matches FD ground truth", (Omega_d_value.cast<double>() - Om_fd).norm() < 1e-3);
    check("Omega_dot_d matches FD ground truth", (Omega_dot_d_value.cast<double>() - Om_dot_fd).norm() < 5e-2);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " FAILED") << "\n";
    return nfail;
}