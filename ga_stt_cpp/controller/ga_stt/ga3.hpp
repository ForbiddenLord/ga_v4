#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <algorithm>

namespace ga {

#if defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE inline
#endif

using Multivector = Eigen::Matrix<float, 8, 1>;
using Vector3 = Eigen::Vector3f;
using Rotor = Eigen::Matrix<float, 8, 1>;

constexpr int blade_mul_sign(int a, int b) {
    int seq[6] = {}, len = 0;
    for (int i = 0; i < 3; ++i) if ((a >> i) & 1) seq[len++] = i;
    for (int i = 0; i < 3; ++i) if ((b >> i) & 1) seq[len++] = i;
    int sign = 1, i = 0;
    while (i < len - 1) {
        if (seq[i] > seq[i + 1]) {
            int tmp = seq[i]; seq[i] = seq[i + 1]; seq[i + 1] = tmp;
            sign *= -1;
            if (i > 0) { --i; continue; }
        } else if (seq[i] == seq[i + 1]) {
            for (int k = i; k < len - 2; ++k) seq[k] = seq[k + 2];
            len -= 2; if (i > 0) --i; continue;
        }
        ++i;
    }
    return sign;
}

constexpr auto make_struct_idx() {
    std::array<std::array<int, 8>, 8> idx{};
    for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) idx[a][b] = a ^ b;
    return idx;
}

constexpr auto make_struct_sign() {
    std::array<std::array<int, 8>, 8> sign{};
    for (int a = 0; a < 8; ++a) for (int b = 0; b < 8; ++b) sign[a][b] = blade_mul_sign(a, b);
    return sign;
}

constexpr auto STRUCT_IDX = make_struct_idx();
constexpr auto STRUCT_SIGN = make_struct_sign();

FORCE_INLINE Multivector geometric_product(const Multivector& A, const Multivector& B) {
    Multivector out = Multivector::Zero();
    #pragma GCC unroll 8
    for (int a = 0; a < 8; ++a) {
        if (A[a] == 0.0f) continue;
        #pragma GCC unroll 8
        for (int b = 0; b < 8; ++b) {
            if (B[b] == 0.0f) continue;
            out[STRUCT_IDX[a][b]] += STRUCT_SIGN[a][b] * A[a] * B[b];
        }
    }
    return out;
}

FORCE_INLINE Rotor reverse(const Rotor& R) {
    Rotor rev = R;
    rev[3] = -R[3]; rev[5] = -R[5]; rev[6] = -R[6]; rev[7] = -R[7];
    return rev;
}

inline Multivector bivector(const Vector3& b) {
    Multivector B = Multivector::Zero();
    B[6] = b[0]; B[5] = -b[1]; B[3] = b[2];
    return B;
}

inline Rotor rotor(float scalar, const Vector3& biv3) {
    Rotor R = bivector(biv3); R[0] = scalar; return R;
}

inline Vector3 to_bivector3(const Multivector& M) {
    return Vector3(M[6], -M[5], M[3]);
}

inline float rotor_scalar(const Rotor& R) { return R[0]; }
inline Vector3 rotor_bivec3(const Rotor& R) { return to_bivector3(R); }

inline Rotor rotor_identity() {
    Rotor R = Rotor::Zero(); R[0] = 1.0f; return R;
}

inline float norm(const Multivector& M) {
    return std::sqrt(std::max(geometric_product(M, reverse(M))[0], 0.0f));
}

inline Rotor normalize_rotor(const Rotor& R) {
    float n = norm(R); if (n < 1e-6f) n = 1.0f; return R / n;
}

FORCE_INLINE Vector3 sandwich(const Rotor& R, const Vector3& v) {
    float s = R[0];
    Vector3 b(-R[6], R[5], -R[3]);
    Vector3 b_cross_v = b.cross(v);
    return v + (2.0f * s) * b_cross_v + 2.0f * b.cross(b_cross_v);
}

FORCE_INLINE Vector3 sandwich_derivative(const Rotor& R, const Vector3& omega_b3, const Vector3& v3) {
    Vector3 v_prime = sandwich(R, v3);
    Vector3 omega_s = sandwich(R, omega_b3);
    return omega_s.cross(v_prime);
}

FORCE_INLINE Vector3 sandwich_bivector(const Rotor& X, const Vector3& biv3) {
    return sandwich(X, biv3);
}

inline std::pair<float, Vector3> rotor_log_angle_axis(const Rotor& Re) {
    float s0 = std::clamp(rotor_scalar(Re), -1.0f, 1.0f);
    Vector3 biv = rotor_bivec3(Re);
    float biv_norm = biv.norm();
    float theta = 2.0f * std::acos(s0);
    Vector3 n_hat = -biv / (biv_norm + 1e-6f);
    return {theta, n_hat};
}

} // namespace ga