#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>

namespace jet {

constexpr int MAX_JET_ORDER = 4;
using ScalarJet = Eigen::Matrix<float, MAX_JET_ORDER + 1, 1>;
using VectorJet = Eigen::Matrix<float, MAX_JET_ORDER + 1, 3>;

inline int order_of(const ScalarJet&) { return MAX_JET_ORDER; }
inline int order_of(const VectorJet&) { return MAX_JET_ORDER; }

inline ScalarJet const_scalar(float value, int /*order*/) {
    ScalarJet c = ScalarJet::Zero();
    c[0] = value;
    return c;
}

inline VectorJet const_vector(const Eigen::Vector3f& value, int /*order*/) {
    VectorJet c = VectorJet::Zero();
    c.row(0) = value;
    return c;
}

inline ScalarJet add(const ScalarJet& a, const ScalarJet& b) { return a + b; }
inline VectorJet add(const VectorJet& a, const VectorJet& b) { return a + b; }
inline ScalarJet sub(const ScalarJet& a, const ScalarJet& b) { return a - b; }
inline VectorJet sub(const VectorJet& a, const VectorJet& b) { return a - b; }

inline ScalarJet mul_ss(const ScalarJet& a, const ScalarJet& b) {
    ScalarJet out = ScalarJet::Zero();
    #pragma GCC unroll 8
    for (int k = 0; k <= MAX_JET_ORDER; ++k) {
        #pragma GCC unroll 8
        for (int i = 0; i <= k; ++i) out[k] += a[i] * b[k - i];
    }
    return out;
}

inline VectorJet mul_sv(const ScalarJet& s, const VectorJet& v) {
    VectorJet out = VectorJet::Zero();
    #pragma GCC unroll 8
    for (int k = 0; k <= MAX_JET_ORDER; ++k) {
        #pragma GCC unroll 8
        for (int i = 0; i <= k; ++i) out.row(k) += s[i] * v.row(k - i);
    }
    return out;
}

inline ScalarJet div_ss(const ScalarJet& a, const ScalarJet& b) {
    ScalarJet h = ScalarJet::Zero();
    h[0] = a[0] / b[0];
    #pragma GCC unroll 8
    for (int k = 1; k <= MAX_JET_ORDER; ++k) {
        float s = a[k];
        #pragma GCC unroll 8
        for (int i = 0; i < k; ++i) s -= h[i] * b[k - i];
        h[k] = s / b[0];
    }
    return h;
}

inline ScalarJet sqrt_s(const ScalarJet& a) {
    ScalarJet h = ScalarJet::Zero();
    h[0] = std::sqrt(std::max(a[0], 0.0f));
    #pragma GCC unroll 8
    for (int k = 1; k <= MAX_JET_ORDER; ++k) {
        float inner = 0.0f;
        #pragma GCC unroll 8
        for (int i = 1; i < k; ++i) inner += h[i] * h[k - i];
        h[k] = (a[k] - inner) / (2.0f * h[0]);
    }
    return h;
}

inline ScalarJet exp_s(const ScalarJet& a) {
    ScalarJet h = ScalarJet::Zero();
    h[0] = std::exp(a[0]);
    #pragma GCC unroll 8
    for (int k = 1; k <= MAX_JET_ORDER; ++k) {
        float sum = 0.0f;
        #pragma GCC unroll 8
        for (int i = 1; i <= k; ++i) sum += (float)i * a[i] * h[k - i];
        h[k] = sum / (float)k;
    }
    return h;
}

inline ScalarJet log_s(const ScalarJet& a) {
    ScalarJet h = ScalarJet::Zero();
    h[0] = std::log(a[0]);
    #pragma GCC unroll 8
    for (int k = 1; k <= MAX_JET_ORDER; ++k) {
        float inner = 0.0f;
        #pragma GCC unroll 8
        for (int i = 1; i < k; ++i) inner += (float)i * h[i] * a[k - i];
        h[k] = (a[k] - inner / (float)k) / a[0];
    }
    return h;
}

inline ScalarJet dot_vv(const VectorJet& u, const VectorJet& v) {
    ScalarJet out = ScalarJet::Zero();
    #pragma GCC unroll 8
    for (int k = 0; k <= MAX_JET_ORDER; ++k) {
        #pragma GCC unroll 8
        for (int i = 0; i <= k; ++i) out[k] += u.row(i).dot(v.row(k - i));
    }
    return out;
}

inline VectorJet cross_vv(const VectorJet& u, const VectorJet& v) {
    VectorJet out = VectorJet::Zero();
    #pragma GCC unroll 8
    for (int k = 0; k <= MAX_JET_ORDER; ++k) {
        #pragma GCC unroll 8
        for (int i = 0; i <= k; ++i) out.row(k) += u.row(i).cross(v.row(k - i));
    }
    return out;
}

inline ScalarJet norm_v(const VectorJet& v) { return sqrt_s(dot_vv(v, v)); }

inline VectorJet vec_div_s(const VectorJet& v, const ScalarJet& s) {
    VectorJet out = VectorJet::Zero();
    #pragma GCC unroll 8
    for (int ax = 0; ax < 3; ++ax) {
        out.col(ax) = div_ss(v.col(ax), s);
    }
    return out;
}

inline ScalarJet differentiate(const ScalarJet& c) {
    ScalarJet out = ScalarJet::Zero();
    #pragma GCC unroll 8
    for (int k = 0; k < MAX_JET_ORDER; ++k) out[k] = (float)(k + 1) * c[k + 1];
    return out;
}

inline VectorJet differentiate(const VectorJet& c) {
    VectorJet out = VectorJet::Zero();
    #pragma GCC unroll 8
    for (int k = 0; k < MAX_JET_ORDER; ++k) out.row(k) = (float)(k + 1) * c.row(k + 1);
    return out;
}

} // namespace jet