#include <iostream>
#include <cmath>
#include <random>
#include "ga3.hpp"

namespace test_utils {
inline ga::Rotor exp_bivector(const Eigen::Vector3f& axis3, float theta) {
    float c = std::cos(theta / 2.0f);
    float s = std::sin(theta / 2.0f);
    return ga::rotor(c, -s * axis3);
}

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

int nfail = 0;
void check(const std::string& name, bool cond) {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name << "\n";
    if (!cond) nfail++;
}

int main() {
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    constexpr float TOL = 1e-5f;

    ga::Multivector pos_one = ga::Multivector::Zero();
    pos_one[0] = 1.0f;

    ga::Multivector e23 = ga::bivector(Eigen::Vector3f(1.0f,0.0f,0.0f));
    ga::Multivector e31 = ga::bivector(Eigen::Vector3f(0.0f,1.0f,0.0f));
    ga::Multivector e12 = ga::bivector(Eigen::Vector3f(0.0f,0.0f,1.0f));

    check("e23^2 == -1", (ga::geometric_product(e23, e23) + pos_one).norm() < TOL);
    check("e31^2 == -1", (ga::geometric_product(e31, e31) + pos_one).norm() < TOL);
    check("e12^2 == -1", (ga::geometric_product(e12, e12) + pos_one).norm() < TOL);

    ga::Multivector A, B, C;
    for(int i=0; i<8; ++i) { A[i]=dist(rng); B[i]=dist(rng); C[i]=dist(rng); }
    auto lhs = ga::geometric_product(ga::geometric_product(A, B), C);
    auto rhs = ga::geometric_product(A, ga::geometric_product(B, C));
    check("Associativity (AB)C == A(BC)", (lhs - rhs).norm() < TOL);

    Eigen::Vector3f b(dist(rng), dist(rng), dist(rng));
    check("bivector roundtrip", (ga::to_bivector3(ga::bivector(b)) - b).norm() < TOL);

    Eigen::Vector3f b1(dist(rng), dist(rng), dist(rng));
    Eigen::Vector3f b2(dist(rng), dist(rng), dist(rng));
    auto comm = ga::to_bivector3(0.5f * (ga::geometric_product(ga::bivector(b1), ga::bivector(b2)) -
                                         ga::geometric_product(ga::bivector(b2), ga::bivector(b1))));
    check("bivector commutator == cross product", (comm + b1.cross(b2)).norm() < TOL);

    Eigen::Vector3f axis(dist(rng), dist(rng), dist(rng));
    axis.normalize();
    float theta = 1.234f;
    auto R = test_utils::exp_bivector(axis, theta);
    auto RRrev = ga::geometric_product(R, ga::reverse(R));
    check("R Rrev == 1", (RRrev - pos_one).norm() < TOL); 

    float max_err = 0.0f;
    for (int i = 0; i < 100; ++i) {
        Eigen::Vector3f ax(dist(rng), dist(rng), dist(rng)); ax.normalize();
        float th = dist(rng);
        auto Rr = test_utils::exp_bivector(ax, th);
        Eigen::Vector3f v_test(dist(rng), dist(rng), dist(rng));
        auto v_ga = ga::sandwich(Rr, v_test);
        Eigen::Vector3f v_rod = v_test * std::cos(th) + ax.cross(v_test) * std::sin(th) + ax * (ax.dot(v_test) * (1.0f - std::cos(th)));
        max_err = std::max(max_err, (v_ga - v_rod).norm());
    }
    check("sandwich matches Rodrigues rotation", max_err < 1e-5f);

    Eigen::Vector3f tp(dist(rng), dist(rng), dist(rng)); tp.normalize();
    Eigen::Vector3f td(dist(rng), dist(rng), dist(rng)); td.normalize();
    auto Ra = test_utils::align_rotor(tp, td);

    check("align_rotor aligns tp -> td", (ga::sandwich(Ra, tp) - td).norm() < 3e-4f);

    auto [th_rec, ax_rec] = ga::rotor_log_angle_axis(R);
    check("rotor_log recovers theta", std::abs(th_rec - theta) < 1e-5f);
    check("rotor_log recovers axis", (ax_rec - axis).norm() < 1e-5f);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " FAILED") << "\n";
    return nfail;
}