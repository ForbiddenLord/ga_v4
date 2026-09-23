#include <iostream>
#include <cmath>
#include "jet1d.hpp"

int nfail = 0;
void check(const std::string& name, bool cond) {
    std::cout << "[" << (cond ? "PASS" : "FAIL") << "] " << name << "\n";
    if (!cond) nfail++;
}

jet::ScalarJet sin_jet(float omega, float t0, int order) {
    jet::ScalarJet c = jet::ScalarJet::Zero();
    for (int k = 0; k <= order; ++k) {
        c[k] = std::pow(omega, k) * std::sin(omega * t0 + k * (float)M_PI / 2.0f);
        float fact = 1.0f; for(int i=1; i<=k; ++i) fact *= i;
        c[k] /= fact;
    }
    return c;
}

int main() {
    float t0 = 0.83f;
    int N = 4;
    auto sin_part = sin_jet(2.0f, t0, N);

    jet::ScalarJet sqrt_arg = jet::const_scalar(t0 + 3.0f, N);
    sqrt_arg[1] = 1.0f;
    auto sqrt_part = jet::sqrt_s(sqrt_arg);

    jet::ScalarJet denom = jet::const_scalar(1.0f + t0*t0, N);
    denom[1] = 2.0f * t0;
    denom[2] = 1.0f;

    auto num = jet::mul_ss(sin_part, sqrt_part);
    auto f_jet = jet::div_ss(num, denom);

    // --- Ground truth ---
    auto f_eval = [](double t) { return std::sin(2.0*t) * std::sqrt(t+3.0) / (1.0 + t*t); };
    double t0d = (double)t0;

    for (int k = 0; k <= N; ++k) {
        double true_deriv = 0.0;
        double h = 1e-4;
        if (k == 0) true_deriv = f_eval(t0d);
        else if (k == 1) true_deriv = (-f_eval(t0d+2.0*h) + 8.0*f_eval(t0d+h) - 8.0*f_eval(t0d-h) + f_eval(t0d-2.0*h)) / (12.0*h);
        else if (k == 2) true_deriv = (-f_eval(t0d+2.0*h) + 16.0*f_eval(t0d+h) - 30.0*f_eval(t0d) + 16.0*f_eval(t0d-h) - f_eval(t0d-2.0*h)) / (12.0*h*h);
        else {
            double hh = 1e-3;
            if (k == 3) true_deriv = (f_eval(t0d+2.0*hh) - 2.0*f_eval(t0d+hh) + 2.0*f_eval(t0d-hh) - f_eval(t0d-2.0*hh)) / (2.0*hh*hh*hh);
            else       true_deriv = (f_eval(t0d+2.0*hh) - 4.0*f_eval(t0d+hh) + 6.0*f_eval(t0d) - 4.0*f_eval(t0d-hh) + f_eval(t0d-2.0*hh)) / (hh*hh*hh*hh);
        }
        float fact = 1.0f; for(int i=1; i<=k; ++i) fact *= i;
        double comp_deriv = (double)(fact * f_jet[k]);

        check("f(t) deriv k=" + std::to_string(k), std::abs(comp_deriv - true_deriv) < 1e-3 * (std::abs(true_deriv) + 1.0));
    }

    jet::ScalarJet exp_arg = jet::const_scalar(0.5f * t0, N);
    exp_arg[1] = 0.5f;
    auto ej = jet::exp_s(exp_arg);
    check("exp(0.5t) val", std::abs(ej[0] - std::exp(0.5f*t0)) < 1e-5f);
    check("exp(0.5t) d1", std::abs(ej[1] - 0.5f*std::exp(0.5f*t0)) < 1e-5f);

    jet::ScalarJet log_arg = jet::const_scalar(2.0f + t0, N);
    log_arg[1] = 1.0f;
    auto lj = jet::log_s(log_arg);
    check("log(2+t) val", std::abs(lj[0] - std::log(2.0f+t0)) < 1e-5f);
    check("log(2+t) d1", std::abs(lj[1] - 1.0f/(2.0f+t0)) < 1e-5f);

    std::cout << "\n" << (nfail == 0 ? "ALL TESTS PASSED" : std::to_string(nfail) + " FAILED") << "\n";
    return nfail;
}