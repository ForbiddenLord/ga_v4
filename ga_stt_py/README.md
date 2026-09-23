# GA-STT — Python Validation Suite

This is the codebase the paper's claims and figures were produced from.

### Initialize
Python == 3.10.12
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Directory layout

```
ga_stt_py/
├── lib/           # GA-STT algorithm + baselines it's compared against
├── tests/         # unit + integration tests
├── scenarios/     # scripts that produced the paper's tables/figures
├── out/           # generated figures/tables
└── requirements.txt
```

### `lib/`

| Module | Role |
|---|---|
| `ga3.py` | 3-D Geometric Algebra (Cl(3,0)) engine: geometric product, rotors, sandwich rotation. |
| `jet1d.py` | Order-N Taylor-jet arithmetic (exact automatic differentiation) for scalars and R³ vectors. |
| `rotor_jet.py` | Taylor jets of GA rotors — how `Ω_d`, `Ω̇_d` are obtained analytically, with no finite differences. |
| `stt.py` | Spatiotemporal tube dynamics: tube centre `σ(t)`, radius `ρ_p(t)`, and their exact Taylor jets. |
| `ga_stt_controller.py` | The cascaded GA-STT controller (Stage 1 / Stage R / Stage 2). |
| `dynamics.py` | Rigid-body equations of motion + RK4 integrator, for closed-loop simulation. |
| `simulate.py` | Closed-loop simulation harness shared by the test/scenario scripts. |
| `baseline_controller.py` | Classical Lee et al. DCM + backward-finite-difference `Ω_d` baseline — isolates the paper's rotational-chain claim. |
| `cbf_controller.py` | Control Barrier Function (CBF-QP) safety-filter baseline. |
| `mpc_controller.py` | Receding-horizon MPC baseline. |

### Benchmark environment
 
The compute-time figures in Table I / Fig. 2 (and the `ctrl_ms_*` columns of
`out/scaling_study.csv`) were produced on:
 
| | |
|---|---|
| OS | Ubuntu 22.04 |
| CPU | Intel Core i5, 13th Gen |
| RAM | 16 GB |
| Threading | Default library behaviour — **no manual thread pinning** |

### `tests/`

Run using:

```bash
for f in tests/test_*.py; do python3 "$f" || echo "FAILED: $f"; done
```

| Test | Covers |
|---|---|
| `test_ga3.py` | GA primitives against an independent reference (`scipy.spatial.transform.Rotation`). |
| `test_jet1d.py` | Jet arithmetic against exact `sympy` derivatives. |
| `test_rotor_jet.py` | Jet-valued rotor chain (`Ω_d`, `Ω̇_d`) against finite differences of a directly-constructed rotor. |
| `test_stt_jet.py` | Tube dynamics jets against a high-accuracy `solve_ivp` ground truth. |
| `test_stage_r_and_smoke.py` | Closed-form Stage-R Lyapunov rate (eq. in paper) reproduced exactly; smoke-tests `compute_control`. |
| `test_controller_translational.py` | Stage-1 translational chain (`F_d`, `Ḟ_d`) against finite-difference ground truth. |
| `test_closed_loop_convergence.py` | Full closed-loop convergence under real rigid-body dynamics. |
| `test_noise_sensitivity.py` | Sensitivity of the translational Taylor-jet chain to measurement noise on `Ω_b`/`v`. |

### `scenarios/`

| Script | Generates |
|---|---|
| `run_scaling_study.py` | Table I / Fig. 2 — obstacle-count scaling study (GA-STT vs. Baseline vs. CBF-QP, tube-safety rate and compute time). |
| `run_scenario2.py` | Fig. 1 — 180° flip recovery, isolating the rotational singularity-free claim. |

```bash
python3 scenarios/run_scenario2.py
python3 scenarios/run_scaling_study.py --trials 25
```

`run_scaling_study.py` checkpoints after every `(obstacle count, condition)`
block to a temp `.pkl`, so `Ctrl-C` + rerun resumes rather than restarting.

Trials are seeded (fixed master seed): rerunning reproduces the same 25
trials and the same CSV, not a fresh random draw each time.