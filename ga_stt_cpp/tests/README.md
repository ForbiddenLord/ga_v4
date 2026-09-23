# GA-STT Controller — Test Suite

## Prerequisites

- CMake ≥ 3.14
- A C++17 compiler (GCC or Clang)
- Eigen3 ≥ 3.4

On Debian/Ubuntu, install Eigen3 with:

```bash
sudo apt-get install libeigen3-dev
```

## Building and running

From `ga_stt_cpp/tests/`:

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ctest --output-on-failure
```

Each test also builds as a standalone executable (e.g. `./test_ga3`) if you
want to run one in isolation.

## Test descriptions

| Test | Covers |
|---|---|
| `test_ga3.cpp` | Core geometric algebra: blade multiplication rules, rotor properties, sandwich product vs. Rodrigues' rotation formula, rotor logarithm. |
| `test_jet1d.cpp` | Scalar Taylor-jet arithmetic (`add`, `mul`, `div`, `sqrt`, `exp`, `log`) against closed-form and finite-difference derivatives, up to 4th order. |
| `test_rotor_jet.cpp` | Jet-valued rotor algebra: aligning a thrust axis to a target direction and propagating angular velocity `Ω` and `Ω̇` through the jet chain. |
| `test_stt_jet.cpp` | STT trajectory generator: `σ(t)` and its derivatives under obstacle avoidance, and the shrinking safety-tube radius `ρ_p(t)`. |
| `test_stage_r_and_smoke.cpp` | Attitude-tracking (rotational) error dynamics smoke test. |
| `test_controller_translational.cpp` | Translational control chain (desired force/thrust from position error). |
| `test_closed_loop_convergence.cpp` | Full closed-loop controller convergence over a simulated trajectory. |