cmake_minimum_required(VERSION 3.14)
project(ga_stt_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED)

# Embedded-safety and performance flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
add_compile_definitions(EIGEN_NO_MALLOC EIGEN_STACK_ALLOCATION_ONLY)

# Point to the GA-STT headers based on your folder structure
include_directories(${CMAKE_SOURCE_DIR}/../controller/ga_stt)
include_directories(${EIGEN3_INCLUDE_DIR})

# Helper macro for tests
macro(add_ga_stt_test TEST_NAME TEST_SOURCE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} Eigen3::Eigen)
endmacro()

# --- Tests ---
enable_testing()

# Core Math Primitives
add_ga_stt_test(test_ga3 test_ga3.cpp)
add_ga_stt_test(test_jet1d test_jet1d.cpp)
add_ga_stt_test(test_rotor_jet test_rotor_jet.cpp)

# STT & Controller Logic
add_ga_stt_test(test_stt_jet test_stt_jet.cpp)
add_ga_stt_test(test_stage_r_and_smoke test_stage_r_and_smoke.cpp)
add_ga_stt_test(test_controller_translational test_controller_translational.cpp)
add_ga_stt_test(test_closed_loop_convergence test_closed_loop_convergence.cpp)

add_test(NAME test_ga3 COMMAND test_ga3)
add_test(NAME test_jet1d COMMAND test_jet1d)
add_test(NAME test_rotor_jet COMMAND test_rotor_jet)
add_test(NAME test_stt_jet COMMAND test_stt_jet)
add_test(NAME test_stage_r_and_smoke COMMAND test_stage_r_and_smoke)
add_test(NAME test_controller_translational COMMAND test_controller_translational)
add_test(NAME test_closed_loop_convergence COMMAND test_closed_loop_convergence)