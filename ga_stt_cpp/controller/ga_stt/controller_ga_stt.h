/**
 * controller_ga_stt.h -- C-linkage interface for the GA-STT controller.
 *
 * This header is included by BOTH controller.c (compiled as plain C) and
 * controller_ga_stt.cpp (compiled as C++, where the actual GA-STT math
 * lives). The extern "C" block gives the three entry points C linkage so
 * controller.c can call them via ordinary function declarations, with no
 * knowledge of Eigen, namespaces, or anything else C++-specific.
 *
 * NOTE: `stabilizerStep_t` is a plain uint32_t tick counter (confirmed from
 * stabilizer_types.h + Bitcraze docs), incrementing at RATE_MAIN_LOOP =
 * 1000 Hz. controllerGaStt() is called every tick of that loop.
 */
#pragma once

#include "stabilizer_types.h"
#include "controller.h"

#ifdef __cplusplus
extern "C" {
#endif

void controllerGaSttInit(void);
bool controllerGaSttTest(void);
void controllerGaStt(control_t *control, const setpoint_t *setpoint,
                      const sensorData_t *sensors, const state_t *state,
                      const stabilizerStep_t stabilizerStep);

#ifdef __cplusplus
}
#endif
