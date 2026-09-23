# Tuning & MuJoCo Validation Guide (Crazyflie 2, real model)

This covers everything in `ga_stt_py/mujoco_validation/` added after the
Stage-R reversion (see the top-level `DELIVERY_STATUS.md` for why). It
assumes Ubuntu 22.04.5 LTS with Python 3.10+.

## 1. Install

```bash
# System Python is fine; a venv is optional but recommended
python3 -m venv ~/.venvs/gastt && source ~/.venvs/gastt/bin/activate

pip install mujoco numpy matplotlib
```

That's the only new dependency beyond what the rest of this delivery
already needs (`numpy`). `mujoco` (the official DeepMind Python bindings,
`pip install mujoco`) bundles the MuJoCo engine itself — no separate
MuJoCo install, license, or `LD_LIBRARY_PATH` setup is needed; this is the
modern (post-2022, Apache-2.0) MuJoCo, not the old paid/licensed version.

Verify it works:
```bash
python3 -c "import mujoco; print(mujoco.__version__)"
```
Should print `3.x.x`.

## 2. What's already in this delivery vs. what you don't need to fetch

`ga_stt_py/mujoco_validation/crazyflie_menagerie/` is a copy of
`bitcraze_crazyflie_2/` from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(MIT license, see the `LICENSE` file inside that folder), taken as-is —
same mesh, same `mass=0.027`, `diaginertia=(2.3951e-5, 2.3951e-5,
3.2347e-5)` from `cf2.xml`. You do not need to clone the menagerie
yourself; it's already here. If you want a newer version or a different
airframe from the same repo:
```bash
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
# then point --xml-path (where scripts take it) or copy the folder in,
# matching cf_scene.xml's <include file="cf2.xml"/>
```

## 3. Files

| File | What it is |
|---|---|
| `ga_stt_py/lib/ga_stt_controller_final.py` | The controller to use: Stage-1 fixed (velocity damping, corrected Jacobian, Picard reference generator with acceleration cap), Stage-R **reverted** to the original, proven-exact law. See its module docstring. |
| `crazyflie_menagerie/` | The real Crazyflie 2 MJCF (unmodified), copied from mujoco_menagerie. |
| `crazyflie_menagerie/cf_scene.xml` | Wraps `cf2.xml` with a floor, obstacle/target visual markers, and the settings the validation scripts expect. |
| `run_cf2_validation.py` | Single-scenario closed-loop run (what was run in the delivery session — see its printed output for the result). Run this first. |
| `tune_gains.py` | CLI-driven gain grid-search tool (mass, inertia, thrust/torque limits, obstacle, scenario, and target response speed/damping all as flags). Re-run this whenever you change the vehicle, obstacle scenario, or actuator limits. |
| `monte_carlo_cf2.py` | Batch/Monte Carlo sweep over randomized obstacle fields — **not run in the delivery session**, run this yourself (see below). |

## 4. Run the single validation (reproduces the delivery-session result)

```bash
cd ga_stt_py/mujoco_validation
python3 run_cf2_validation.py
```

Expected output (from the delivery session): no divergence, `theta_e`
converges from 0.49 rad to ~5e-4 rad within 1 s, `max e_p` stays around
0.13 (well inside the tube, which is violated at `e_p>=1`), and thrust
only briefly touches its 0.60 N ceiling for ~0.05% of the run during the
initial recovery transient (not sustained saturation). `cf2_validation_result.png`
(already included) is a 3-panel plot of `theta_e`, `e_p`, and thrust vs.
time from that run — regenerate it by adding the plotting snippet at the
bottom of `run_cf2_validation.py`'s module or just re-running the inline
version:
```bash
python3 -c "
from run_cf2_validation import run
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
res = run(verbose_every=0)
fig, axs = plt.subplots(3, 1, figsize=(8,9), sharex=True)
axs[0].plot(res['t'], res['theta_e']); axs[0].set_yscale('log'); axs[0].set_ylabel('theta_e')
axs[1].plot(res['t'], res['e_p']); axs[1].axhline(1.0, color='r', ls='--'); axs[1].set_ylabel('e_p')
axs[2].plot(res['t'], res['thrust']); axs[2].axhline(0.60, color='r', ls='--'); axs[2].set_ylabel('thrust (N)')
plt.tight_layout(); plt.savefig('cf2_validation_result.png', dpi=130)
"
```

To *watch* it (interactive viewer, not headless) instead of just getting
numbers, swap `mujoco.mj_step` calls for MuJoCo's `viewer` module:
```python
import mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # inside the existing step loop, after mj_step(model, data):
    viewer.sync()
    import time; time.sleep(dt)  # real-time playback
```
(Needs a display; skip this on a headless server.)

## 5. Re-tuning gains (if you change the vehicle or scenario)

`tune_gains.py` is a CLI-driven grid search over `(kappa1, kappa_v, kR,
kOmega)`, using a fast custom Euler integrator (not MuJoCo — for speed;
it always tells you to confirm the winner against the real MuJoCo run
afterward, which is what `run_cf2_validation.py` is for). Run with
`--help` for the full option list; the methodology (the
`omega_n`/`zeta` linearization it uses to pick a sensible starting point
for `kappa1`/`kappa_v` before grid-searching around it) is explained in
its module docstring, not repeated here.

Example, retuning for a different vehicle and a tighter tube:
```bash
python3 tune_gains.py \
    --mass 0.05 --inertia 3e-5 3e-5 5e-5 \
    --f-max 1.0 --tau-max 0.02 \
    --target-omega-n 5.0 --target-zeta 0.9 \
    --obstacle 0.6 0.6 0.5 0.15 \
    --start 0.0 0.0 0.4 --target 1.2 1.2 1.0 --tc 8.0 --rho-max 0.25
```

Default search is 16 combinations (2 values per gain); pass
`--grid-factors 0.5 1.0 2.0` for a finer 81-combination search around the
same guess, or `--grid-factors 1.0` to just test the linearized guess
itself with no search at all (1 combination — the fastest possible
sanity check). It prints a ready-to-copy `DroneParams(...)` line for the
winner. Confirm that in the real MuJoCo sim before trusting it — same
warning the script itself prints.

`f_max=0.60` N and `tau_max=0.012` N·m are **assumptions**, not measured
or taken from the MJCF (see `run_cf2_validation.py`'s docstring for why).
If you have real thrust-stand or system-ID numbers for your specific
Crazyflie (stock vs. thrust-upgrade motors, battery, prop wear all
matter), use those instead and re-tune.

## 6. Running the Monte Carlo sweep yourself

```bash
cd ga_stt_py/mujoco_validation
python3 monte_carlo_cf2.py --n-obstacles 5  --trials 20 --out results_n5.csv
python3 monte_carlo_cf2.py --n-obstacles 15 --trials 20 --out results_n15.csv
python3 monte_carlo_cf2.py --n-obstacles 25 --trials 20 --out results_n25.csv
```

Each trial is a fresh MuJoCo model load + an 8 s closed-loop sim
(4000 steps at `dt=0.002`); expect roughly 15-20 s/trial on a modern
laptop CPU (a 20-trial sweep: a few minutes). A smoke test with
`--trials 2` at `--n-obstacles 5` took 34.6 s total in the delivery
session's environment (100% tube-safe); a single `--trials 1
--n-obstacles 25` trial took 4.9 s and **did** register a tube violation
(`max_ep=0.968`, `tube_safe=False`) — both outcomes are expected and
exactly the kind of result this sweep exists to characterize at scale;
neither is a bug.

Each row of the output CSV has: `seed, n_obstacles, diverged, max_ep,
tube_safe, theta_e_final, frac_saturated, start`. Columns worth
aggregating across trials, per `n_obstacles`:
- **`tube_safe` fraction** — the headline metric (compare against the
 parallel effort's own scaling study, which reported 100% at low density
 dropping to 75% at `n_obstacles=25` for their controller variants).
- **`frac_saturated`** — mean/max across trials. A parallel effort
 attributed the 25-obstacle tube violations to genuine actuator
 saturation; this column is the direct way to check that against the
 corrected controller (do violated trials also show high
 `frac_saturated`, or do they violate the tube for some other reason?
 the mechanism-level argument in the top-level status doc predicts high
 `frac_saturated` should now be much rarer than before, since the
 Picard+cap Stage-1 fix specifically targets the runaway-feedforward
 mechanism that a garbage `Omega_dot_d` term produces near a close
 obstacle — but that prediction hasn't been checked against this exact
 harness's numbers yet; that's what running this sweep will tell you).

A minimal aggregation snippet:
```python
import pandas as pd
df = pd.read_csv("results_n25.csv")
print(df["tube_safe"].mean(), df["frac_saturated"].mean())
print(df[~df["tube_safe"]][["seed", "max_ep", "frac_saturated"]])  # look at the failures specifically
```

## 7. Known limitations of this validation

- Obstacles are visual markers + inputs to the STT reference generator
 only; MuJoCo's own collision engine is not involved (`contype="0"
 conaffinity="0"` on the obstacle geoms) — a tube violation here means
 the *planned* safety margin was breached, not that MuJoCo detected an
 actual collision with the mesh.
- `f_max` and `tau_max` are literature-based assumptions, not
 hardware-measured (Section 5 above and `run_cf2_validation.py`'s
 docstring).
- No sensor noise, no motor/ESC dynamics, no communication delay — same
 limitation as the rest of this delivery's simulation work, noted in the
 top-level `DELIVERY_STATUS.md`.
- `monte_carlo_cf2.py`'s obstacle generator is a simple rejection sampler
 (uniform in a box, resampled if too close to start/target); it is not
 the same generator the parallel effort used for their own scaling
 study, so the two studies' numbers are not a like-for-like comparison
 out of the box even at the same `n_obstacles` — useful for tracking
 this controller's own behavior as density increases, less so for a
 precise head-to-head against their reported percentages.

## 8. Motor/ESC dynamics, sensor noise, comm delay, and the Holybro X500

Not covered above because it was added later. Short version:

- `hardware_realism.py` adds a quadrotor motor mixer, first-order
  motor/ESC lag, sensor noise, and a comm-delay ring buffer.
- `run_x500_validation.py` / `run_x500_validation_realistic.py` do for
  a Holybro X500 (built from a PX4 Gazebo SDF, `x500/x500.xml`) exactly
  what `run_cf2_validation.py` / `run_cf2_validation_realistic.py` do for
  the Crazyflie 2: an ideal-actuator baseline, and a version with motor
  lag + sensor noise + comm delay added.
- **Read `TIME_SCALE_SEPARATION_ANALYSIS.md` before assuming the
  `_realistic` scripts' gains or tube schedules transfer to your own
  scenario.** Short version of that document: the ideal-actuator gains
  do not survive realistic motor lag, a specific gain-derivation method
  is needed (and its source got one formula backwards, corrected there),
  and even with corrected gains, the *tube schedule itself* had to be
  relaxed before a comfortably stable operating point existed for either
  vehicle. This is a real, load-bearing finding, not a footnote.
- Run them the same way as the others: `python3 run_x500_validation.py`,
  `python3 run_x500_validation_realistic.py`, etc.

## 9. Rigorous κ1/κ_v bounds tied to tube shrink rate and t_c

`tune_gains.py` now accepts `--mu-max` (your scenario's worst-case tube
shrink rate `max(-rho_dot/rho)`) and `--tc-margin-N`. When `--mu-max` is
nonzero, the script checks its omega_n/zeta guess against two additional,
independently-derived-and-verified bounds before grid-searching:
`kappa_v > mu_max` and a `t_c`-settling-time lower bound on `kappa1`,
raising the guess if either is violated. See
`STAGE1_GAIN_TUNING_ANALYSIS.md` for the full derivation and verification
against the exact nonlinear Stage-1 dynamics (not just the linearization)
-- including a worked example of a *different* proposed tuning method
(from the same parallel effort behind `TIME_SCALE_SEPARATION_ANALYSIS.md`)
that was checked and found to produce provably unstable gains from its
own worked example.

```bash
python3 tune_gains.py --mu-max 0.8 --tc 10.0 --rho-min 0.15 ...
```

If you don't have an estimate of `mu_max` for your scenario, leave it at
0 (the default) -- the script behaves exactly as before.
