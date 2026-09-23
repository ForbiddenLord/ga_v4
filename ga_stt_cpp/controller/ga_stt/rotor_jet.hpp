#pragma once
#include "ga3.hpp"
#include "jet1d.hpp"

namespace rotor_jet {

struct RotorJet {
    jet::ScalarJet s;
    jet::VectorJet b;
    RotorJet() : s(jet::ScalarJet::Zero()), b(jet::VectorJet::Zero()) {}
    RotorJet(const jet::ScalarJet& s_, const jet::VectorJet& b_) : s(s_), b(b_) {}
    int order() const { return jet::order_of(s); }
};

inline RotorJet mul(const RotorJet& R1, const RotorJet& R2) {
    jet::ScalarJet s = jet::sub(jet::mul_ss(R1.s, R2.s), jet::dot_vv(R1.b, R2.b));
    jet::VectorJet b = jet::sub(jet::add(jet::mul_sv(R1.s, R2.b), jet::mul_sv(R2.s, R1.b)), jet::cross_vv(R1.b, R2.b));
    return RotorJet(s, b);
}

inline RotorJet reverse(const RotorJet& R) { return RotorJet(R.s, -R.b); }

inline RotorJet differentiate(const RotorJet& R) {
    return RotorJet(jet::differentiate(R.s), jet::differentiate(R.b));
}

inline RotorJet align(const jet::VectorJet& tp, const jet::VectorJet& td) {
    jet::ScalarJet dot_jet = jet::dot_vv(td, tp);
    jet::VectorJet cross_jet = jet::cross_vv(td, tp);
    int order = jet::MAX_JET_ORDER;
    jet::ScalarJet N_s = jet::add(jet::const_scalar(1.0f, order), dot_jet);
    jet::VectorJet N_b = cross_jet;
    jet::ScalarJet D = jet::sqrt_s(jet::mul_ss(jet::const_scalar(2.0f, order), jet::add(jet::const_scalar(1.0f, order), dot_jet)));
    return RotorJet(jet::div_ss(N_s, D), jet::vec_div_s(N_b, D));
}

inline ga::Rotor value_multivector(const RotorJet& R) {
    return ga::rotor(R.s[0], R.b.row(0).transpose());
}

} // namespace rotor_jet