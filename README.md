# GA-STT: Geometric Algebra-Based Spatiotemporal Tubes
This repository provides a singularity-free, no numerical differentiation, closed-form attitude tracking control framework for underactuated UAVs operating in dynamic, cluttered environments. By unifying 3D Geometric Algebra (GA) with Spatiotemporal Tubes (STT), GA-STT guarantees **Temporal Reach-Avoid-Stay (T-RAS)** specifications with formal Input-to-State Stability (ISS) bounds.

### Quickstart
 
```bash
git clone <this-repo-url> ga-stt && cd ga-stt/ga_stt_py
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
for f in tests/test_*.py; do python3 "$f" || echo "FAILED: $f"; done   # unit tests
python3 scenarios/run_scenario2.py                                     # reproduces Fig. 1
```
 
See [`ga_stt_py/README.md`](ga_stt_py/README.md) for the full test/scenario list, and
[`ga_stt_cpp/controller/README.md`](ga_stt_cpp/controller/README.md) for the Crazyflie
firmware build.


### Repository Structure

| | Purpose | Language |
|---|---|---|
| [`ga_stt_cpp/`](ga_stt_cpp/) | The header-only C++ implementation using Eigen. Ported to single precision for the Crazyflie firmware | C++ |
| [`ga_stt_py/`](ga_stt_py/) | Reference implementation, unit tests, and the scripts that generate the paper's tables and figures | Python |

Start with
`ga_stt_cpp/` if you want hardware deployment;  
`ga_stt_py/` if you want to understand or extend the algorithm.

### Affiliation
 
This work is carried out under the supervision of
 
> **Pushpak Jagtap**  
> Assistant Professor  
> Department of Cyber-Physical Systems  
> Indian Institute of Science, Bangalore


### Citation

The manuscript is still in preparation:

```bibtex
@article{bhat_gastt,
  title   = {Geometric Algebra-Based Spatiotemporal Tubes (GA-STT) for Safe,
             Prescribed-Time Trajectory Tracking of Underactuated UAVs},
  author  = {Bhat, Kumar S. and TODO_add_coauthors},
  journal = {TODO_venue},
  year    = {TODO},
  note    = {manuscript}
}
```
