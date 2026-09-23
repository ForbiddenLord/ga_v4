// controller_ga_stt.cpp -- GA-STT controller wrapper.

#include "controller_ga_stt.h"

#include "ga3.hpp"
#include "jet1d.hpp"
#include "rotor_jet.hpp"
#include "stt.hpp"
#include "ga_stt_controller.hpp"

extern "C" {
#include "log.h"
#include "param.h"
#include "usec_time.h"
}

namespace {

constexpr float DRONE_MASS_KG = 0.0282f;
const Eigen::Vector3f DRONE_J_DIAG(21.06e-6f, 21.14e-6f, 35.41e-6f);
constexpr float F_MIN_N = 0.0f;
constexpr float F_MAX_N = 0.351f;
const Eigen::Vector3f TAU_MAX_VEC(7.70e-3f, 7.70e-3f, 2.10e-3f);

static float param_kappa1 = 1.5f;
static float param_kappa_v = 2.0f;  // new: translational velocity-error damping gain
static float param_kr = 3.0f;
static float param_komega = 0.005f;
static float param_eta_x = 0.0f;
static float param_eta_y = 0.0f;
static float param_eta_z = 0.5f;
static float param_rho_max = 0.5f;

PARAM_GROUP_START(ga_stt)
PARAM_ADD(PARAM_FLOAT, kappa1, &param_kappa1)
PARAM_ADD(PARAM_FLOAT, kappa_v, &param_kappa_v)
PARAM_ADD(PARAM_FLOAT, kr, &param_kr)
PARAM_ADD(PARAM_FLOAT, komega, &param_komega)
PARAM_ADD(PARAM_FLOAT, eta_x, &param_eta_x)
PARAM_ADD(PARAM_FLOAT, eta_y, &param_eta_y)
PARAM_ADD(PARAM_FLOAT, eta_z, &param_eta_z)
PARAM_ADD(PARAM_FLOAT, rho_max, &param_rho_max)
PARAM_GROUP_STOP(ga_stt)

ga_stt_ctrl::DroneParams DRONE_PARAMS(
    DRONE_MASS_KG, DRONE_J_DIAG, F_MIN_N, F_MAX_N,
    param_kappa1, param_kappa_v, param_kr, param_komega
);
stt::ObstacleArray EMPTY_OBSTACLES{nullptr, 0};

stt::STTParams STT_PARAMS(
    Eigen::Vector3f(param_eta_x, param_eta_y, param_eta_z),
    Eigen::Vector3f(0.0f, 0.0f, 0.015f),
    20.0f,                              // tc
    0.15f, 1.0f, 0.5f,                  // k1, k2, k3
    param_rho_max,
    0.05f,                              // rho_min
    8.0f,                               // u
    0.5f, 0.05f, 0.5f                   // rhoR0, rhoR_inf, kR
);

// ---- Persistent state across calls ----
Eigen::Vector3f g_sigma = Eigen::Vector3f::Zero();
ga::Rotor g_Rd_hover = ga::rotor_identity();
double g_t_start = -1.0;
bool g_latched = false;

constexpr double MAIN_LOOP_HZ = 1000.0;
constexpr float MAIN_LOOP_DT_F = 1.0f / 1000.0f;
constexpr float DEG2RAD_F = 0.017453292519943295f;

uint32_t g_log_call_count = 0;
float g_log_f_cmd = 0.0f;
float g_log_tau_x = 0.0f, g_log_tau_y = 0.0f, g_log_tau_z = 0.0f;
float g_log_sigma_z = 0.0f;
float g_log_theta_e = 0.0f;
float g_log_e_R = 0.0f;
uint32_t g_log_nan_guard_count = 0;
uint32_t g_log_exec_us = 0;
uint8_t param_relatch = 0;

// -- Decimation: full compute_control() runs every Nth tick (effective
// rate = 1000/N Hz); output is cached and reused on skipped ticks.
static uint8_t param_decimation = 4;

double g_cached_f_cmd = 0.0;
Eigen::Vector3f g_cached_tau = Eigen::Vector3f::Zero();
ga_stt_ctrl::Diagnostics g_cached_diag{
    Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 0.0f, 0.0f, 0.0f};
uint32_t g_decim_counter = 0;

}  // namespace

LOG_GROUP_START(ga_stt)
LOG_ADD(LOG_UINT32, calls, &g_log_call_count)
LOG_ADD(LOG_FLOAT, f_cmd, &g_log_f_cmd)
LOG_ADD(LOG_FLOAT, tau_x, &g_log_tau_x)
LOG_ADD(LOG_FLOAT, tau_y, &g_log_tau_y)
LOG_ADD(LOG_FLOAT, tau_z, &g_log_tau_z)
LOG_ADD(LOG_FLOAT, sigma_z, &g_log_sigma_z)
LOG_ADD(LOG_FLOAT, theta_e, &g_log_theta_e)
LOG_ADD(LOG_FLOAT, e_R, &g_log_e_R)
LOG_ADD(LOG_UINT32, nan_guards, &g_log_nan_guard_count)
LOG_ADD(LOG_UINT32, exec_us, &g_log_exec_us)
LOG_GROUP_STOP(ga_stt)

PARAM_GROUP_START(ga_stt_ctl)
PARAM_ADD(PARAM_UINT8, relatch, &param_relatch)
PARAM_ADD(PARAM_UINT8, decimation, &param_decimation)
PARAM_GROUP_STOP(ga_stt_ctl)

void controllerGaSttInit(void) {
    g_Rd_hover = ga::rotor_identity();
    g_sigma.setZero();
    g_t_start = -1.0;
    g_latched = false;
    g_log_call_count = 0;
    g_log_nan_guard_count = 0;
    g_decim_counter = 0;
}

bool controllerGaSttTest(void) {
    return true;
}

void controllerGaStt(control_t *control, const setpoint_t *setpoint,
                      const sensorData_t *sensors, const state_t *state,
                      const stabilizerStep_t stabilizerStep) {
    (void)setpoint;

    uint64_t t_enter_us = usecTimestamp();

    const double t_now = (double)stabilizerStep / MAIN_LOOP_HZ;
    if (g_t_start < 0.0) g_t_start = t_now;
    const double t_stt = t_now - g_t_start;
    const float t_stt_f = (float)t_stt;

    const Eigen::Vector3f p(state->position.x, state->position.y, state->position.z);
    const Eigen::Vector3f v(state->velocity.x, state->velocity.y, state->velocity.z);

    const Eigen::Vector3f omega_b = Eigen::Vector3f(sensors->gyro.x, sensors->gyro.y,
                                                     sensors->gyro.z) * DEG2RAD_F;

    const ga::Rotor R = ga::rotor(
        (float)state->attitudeQuaternion.w,
        -Eigen::Vector3f(state->attitudeQuaternion.x,
                         state->attitudeQuaternion.y,
                         state->attitudeQuaternion.z));

    if (!g_latched || param_relatch) {
        g_sigma = p;
        g_Rd_hover = R;
        g_t_start = t_now;
        g_latched = true;
        param_relatch = 0;
    }

    // -- Run GA-STT, decimated --
    const uint32_t decim = (param_decimation > 0) ? param_decimation : 1;
    const bool do_full_compute = (g_decim_counter % decim == 0);
    g_decim_counter++;

    float f_cmd;
    Eigen::Vector3f tau;
    ga_stt_ctrl::Diagnostics diag;

    if (do_full_compute) {
        auto result = ga_stt_ctrl::compute_control(
            t_stt_f, p, v, R, omega_b, g_sigma,
            EMPTY_OBSTACLES, STT_PARAMS, DRONE_PARAMS, g_Rd_hover);
        f_cmd = std::get<0>(result);
        tau   = std::get<1>(result);
        diag  = std::get<2>(result);
        g_cached_f_cmd = f_cmd;
        g_cached_tau   = tau;
        g_cached_diag  = diag;
    } else {
        f_cmd = g_cached_f_cmd;
        tau   = g_cached_tau;
        diag  = g_cached_diag;
    }

    // -- Integrate the STT tube reference every tick regardless of decimation --
    const Eigen::Vector3f sigma_dot = stt::sigma_value(g_sigma, t_stt_f, EMPTY_OBSTACLES, STT_PARAMS);
    g_sigma += sigma_dot * MAIN_LOOP_DT_F;

    Eigen::Vector3f tau_clamped = tau;
    for (int i = 0; i < 3; ++i) {
        if (tau_clamped[i] > TAU_MAX_VEC[i]) tau_clamped[i] = TAU_MAX_VEC[i];
        if (tau_clamped[i] < -TAU_MAX_VEC[i]) tau_clamped[i] = -TAU_MAX_VEC[i];
    }

    bool finite = std::isfinite(f_cmd) && tau_clamped.allFinite();
    if (!finite) {
        f_cmd = 0.0f;
        tau_clamped.setZero();
        g_log_nan_guard_count++;
    }

    control->thrustSi = f_cmd;
    control->torqueX = tau_clamped[0];
    control->torqueY = tau_clamped[1];
    control->torqueZ = tau_clamped[2];
    control->controlMode = controlModeForceTorque;

    g_log_call_count++;
    g_log_f_cmd = f_cmd;
    g_log_tau_x = tau_clamped[0];
    g_log_tau_y = tau_clamped[1];
    g_log_tau_z = tau_clamped[2];
    g_log_sigma_z = g_sigma[2];
    g_log_theta_e = diag.theta_e;
    g_log_e_R = diag.e_R;

    g_log_exec_us = (uint32_t)(usecTimestamp() - t_enter_us);
}