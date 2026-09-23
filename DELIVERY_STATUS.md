# GA-STT delivery status (read this first)

**UPDATE, after further review: Stage-R was reverted.** An earlier version
of this delivery replaced the original attitude-error law
(`1/sin(theta_e/2)` with a threshold floor) with a smooth GA-vee-map
alternative, believing the original had a chattering bug. On further,
corrected investigation (prompted by a parallel effort's review pushing
back on this), that turned out to be wrong: the "chattering" concern was
based on a flawed test, and the original law has a real, independently
re-derived and re-verified EXACT closed-form Lyapunov identity
(`dV_R/dt = -kR*xi_R(eps_R)*eps_R^2`) that the replacement does not have.
**Use `ga_stt_controller_final.py` (Python) and the current
`ga_stt_controller.hpp` (C++, already updated) for new work** -- both now
have Stage-1 fixed (velocity damping, corrected Jacobian, Picard reference
generator + acceleration cap) AND Stage-R reverted to the original,
proven-exact law. See each file's module docstring / header comment for
the full account, and see "Revision: Stage-R reverted" below for the
complete story including what was checked and how.

This is a corrected implementation of the RT-STT + GA quadrotor controller
derived in `STT_GA_Quadrotor_Control_Derivation.pdf` (see also
`GA_STT_Corrected.pdf` for the rewritten paper -- **note the paper has NOT
yet been updated for the Stage-R reversion below**; its Corrigendum 6 on
the vee-map sign convention describes the now-superseded replacement, not
the current code), covering both the C++ (`ga_stt_cpp/`, firmware-facing)
and Python (`ga_stt_py/`, simulation/paper-validation) codebases, plus a
MuJoCo cross-validation harness (`ga_stt_py/mujoco_validation/`) now
including a validation run against the REAL Bitcraze Crazyflie 2 model
(MuJoCo Menagerie), not just a generic free body.

All 7 C++ test binaries, all 7 Python test scripts (both against
`ga_stt_controller.py` and, separately, against
`ga_stt_controller_final.py`), and both MuJoCo validations (generic body
and real Crazyflie 2) pass as of this delivery (see "How to verify" below
-- re-run them yourself rather than trusting this line).

## What was wrong, and what changed

Full derivation and rationale live in the module docstrings/comments at
the point of each fix (`ga_stt_controller.hpp`'s corrigendum comments are
the most complete single account; `ga_stt_controller.py` mirrors them).
Summary:

1. **Missing velocity damping in Stage 1.** The derivation's own
   `u1 = -kappa1*xi_p(eps_p)*e_p + sigma_ddot` has no term in the velocity
   error at all — under feedback linearization this is an undamped
   nonlinear oscillator (marginally stable, not asymptotically stable).
   Fixed by adding `-kappa_v*(v - sigma_dot)`, matching the actual PD laws
   in the two papers this derivation builds on (Lee-Leok-McClamroch;
   Rubio Scola et al.), neither of which uses P-only feedback either.
2. **Wrong barrier Jacobian.** `xi_p = d(eps_p)/d(e1)` is `2/(1-e1^2)`,
   not the PDF's `4/(1-e1^2)` — verified independently (symbolically),
   not taken on trust. The original code applied no such factor at all.
3. **Fragile, actual-state-dependent reference generator, replaced with a
   self-contained Taylor/Picard ODE propagation.** The original
   `resolve_translational_chain` built a *velocity* command and
   differentiated it via a 3-stage bootstrap that re-estimated higher
   derivatives from the vehicle's *actually achieved* attitude/omega at
   each stage — mixing feedback state into what should be a pure
   feedforward reference chain, tick to tick. Replaced with a direct
   Taylor-series (Picard) resolution of `p'=v, v'=u1(p,v,t)`, one order
   shallower (since `u1` is already the acceleration) and using only the
   commanded trajectory — never the vehicle's real attitude/omega.
   **This is a genuine, tested-against-an-alternative design choice, not
   an uncontroversial fix — see "Open architectural question" below.**
4. **Added an explicit acceleration cap** (`accel_cap`, derived from
   `f_max/m`), applied at every Taylor order of `u1`. This is literally
   the PDF's own Assumption 1 (a uniformly bounded commanded
   acceleration) enforced in code. Needed because the corrected (now
   properly strong) log barrier genuinely diverges near an active
   obstacle — a Taylor/Picard extrapolation of a function with a nearby
   singularity blows up well before reaching the singularity itself
   (verified: reached ~1e11 and then NaN without this cap, in a
   configuration close to an obstacle's `rho_min` clearance boundary).
5. ~~**Removed a singular, chattering-prone attitude-error formula.**~~
   **SUPERSEDED, see "Revision: Stage-R reverted" below.** This item and
   items 6-7 described replacing the original Stage-R law with a smooth
   GA vee-map. That replacement has been undone; the original law (as
   described in this now-struck-through item) is back, and turns out not
   to have had the chattering problem claimed here.
6. ~~**Found and fixed a real sign error**~~ **SUPERSEDED** -- the sign
   question was specific to the vee-map replacement, which is no longer
   used. Not a live issue for the reverted law.
7. ~~**Kept the necessary Omega_d frame rotation**~~ **PARTIALLY
   SUPERSEDED** -- the reverted Stage-R law rotates `Omega_d` the same
   way (`ga.sandwich_bivector(reverse(Re), Omega_d)`) but, matching the
   original exactly, does NOT rotate `Omega_dot_d` before using it as
   torque feedforward. This is an intentional reversion to what was
   proven-exact, not an oversight.
8. **Raised the prescribed-time reach-law gain floor** (`k1*tc/(tc-t)`
   had a floor of `1e-6`/`1e-3`, allowing gains up to ~1e5-1e7 for `t`
   near or past `tc`) — a plausible contributor to long-duration drift
   that no test in this delivery happens to run long enough to trigger,
   found only by independently reviewing a parallel effort's finding (see
   below) and verifying it against this codebase. **Still current, not
   affected by the Stage-R reversion.**
9. **Obstacle model extended to constant-acceleration** (was
   constant-velocity only), giving `sigma`'s Taylor jet one more exact
   order before falling back to a zero-jerk assumption. Backward
   compatible (`acc` defaults to zero). **Still current.**
10. ~~**Retuned `kR`** (3.0 -> 8.0...)~~ **SUPERSEDED** -- that retuning
    was for the vee-map replacement. The reverted law uses different
    gains again; see the Crazyflie-2 section below for the values used
    there (`kR=25.0`, retuned for a much smaller real vehicle, not
    comparable to the earlier 3.0/8.0 figures).

## Revision: Stage-R reverted (read this if you only read one section)

A parallel effort reviewed an earlier version of this delivery and pushed
back on the Stage-R replacement (items 5-7, 10 above). Both of their
specific objections were checked independently, not taken on trust, and
both held up:

1. **The "chattering" claim (item 5) does not hold up.** The test behind
   it varied `eps_R` and `theta_e` independently, which is not physically
   possible -- `eps_R` is *derived* from `theta_e` via
   `Psi_e = 1-cos(theta_e/2)`, always. Re-tested with them correctly
   paired: the branch point at `theta_e<=1e-6` is a genuine, correctly-
   handled REMOVABLE singularity. `sympy` confirms the analytic limit as
   `theta_e->0` is exactly 0; the numerical branch matches that to ~1e-5
   just above the threshold, not the multi-million-magnitude jump the
   flawed test found.
2. **The original law has a real, exact closed-form Lyapunov identity
   that the replacement does not.** Independently re-derived here (not
   copied from anywhere) from the GA kinematics
   (`Re_dot = -1/2 Re Omega_b`):
   ```
   dV_R/dt = -kR * xi_R(eps_R) * eps_R^2,    xi_R = 1/(rho_R*(1-e_R))
   ```
   Confirmed by high-precision finite differencing against the actual
   formula across many `(theta_e, t)` combinations (including with an
   actively time-varying funnel), matching to 4-5 significant figures.
   The vee-map replacement was checked the same way and does not reduce
   to any comparably clean closed form -- it stays a non-separable
   function of `eps_R`, `e_R`, and `rho_R`.

Net: the original Stage-R law is better-founded than the replacement.
Both `ga_stt_controller_final.py` and the current `ga_stt_controller.hpp`
now use it, unchanged, with its threshold branch intact (understood now
to be safe). `ga_stt_controller.py` (Python) is left with the vee-map
version, kept only for provenance/comparison -- do not use it for new
work.

A third claim from the same review -- that a `Kbuild`/`controller.c` file
present in this delivery was "worth scrutiny" as unusual -- was also
checked directly: it's ordinary Crazyflie firmware build tooling (a
`Kbuild`-style Makefile fragment and the stock multi-controller dispatch
table), present in the *original* codebase before any of this work
started, not a Linux kernel module or anything unusual.

## Real Crazyflie 2 MuJoCo validation

`ga_stt_py/mujoco_validation/run_cf2_validation.py` runs the corrected
controller (`ga_stt_controller_final.py`) against the actual Bitcraze
Crazyflie 2 MJCF model from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(copied into `crazyflie_menagerie/`, unmodified: real mesh, real mass
0.027 kg, real diagonal inertia `(2.3951e-5, 2.3951e-5, 3.2347e-5)
kg*m^2`), not the generic free-body box used in the earlier
`run_validation.py`. Gains were re-tuned for this much smaller vehicle
(`tune_gains.py`; settled on `kappa1=3.0, kappa_v=6.0, kR=25.0,
kOmega=1.5e-3`) and two assumptions not in the MJCF were made explicit
(`f_max=0.60 N`, `tau_max=0.012 N*m` -- both literature-based, not
hardware-measured; see that script's docstring).

Result of the single (non-Monte-Carlo) run performed in this delivery
session: no divergence, `theta_e` converges from 0.49 rad to ~5e-4 rad
within 1 s, `max e_p` stays around 0.13 (well inside the tube), thrust
briefly touches its ceiling for ~0.05% of the run during the initial
recovery transient only (not sustained saturation). See
`cf2_validation_result.png` and
`ga_stt_py/mujoco_validation/TUNING_AND_MUJOCO_GUIDE.md` for the full
writeup, how to reproduce it, how to re-tune for a different vehicle, and
how to run `monte_carlo_cf2.py` (a batch/scaling sweep over randomized
obstacle fields) yourself -- intentionally not run as a large batch in
this delivery session; see that guide for why and how.

## A parallel effort's delivery was reviewed and partially adopted

A separate attempt at this same task (`GA_STT_delivery_v2.zip`, not
included in this delivery but referenced here) was reviewed in full —
every claim in it was independently re-verified rather than trusted,
per the same standard applied to the original PDF and to the patch
proposed alongside this task. Specifically:

- **Adopted**: their MuJoCo frame-convention findings (rotor-to-quaternion
  needs a sign flip; MuJoCo's free-joint `qvel[3:6]` is already
  body-frame, not world-frame) — both re-verified independently here
  against `ga.sandwich`/`Rdot=-1/2 R Omega_b` ground truth before being
  trusted, and used to fix a stuck bug in this delivery's own MuJoCo
  harness.
- **Adopted, in a corrected form**: their finding that the `tc/(tc-t)`
  gain floor was dangerously tight. Their specific fix (switch to a
  separate fixed-gain "hold" law for `t>=tc`) was found, on independent
  testing, to make `sigma_dot` **discontinuous at `t=tc`** (a jump from
  ~2.4 to 0.5 in a representative case — verified numerically). This
  delivery instead just raises the floor with no second law, which is
  exactly continuous (verified: identical output from `t=tc` through
  `t=6*tc`) and equally bounded.
- **Not adopted**: their Stage-R attitude law, which is unchanged from
  the original and keeps the `1/sin(theta_e/2)` singularity/floor this
  delivery replaced (item 5 above). Their own tuning guide calls this
  "already-validated," but it still has the discontinuity this delivery
  specifically fixed.
- **Investigated, and NOT adopted**: their Stage-1 reference-generator
  architecture (`resolve_translational_chain_v1`), which keeps the
  original design's pattern of bootstrapping higher Taylor orders off
  the vehicle's *actual* current attitude/omega, rather than this
  delivery's self-contained Picard propagation (item 3 above). See "Open
  architectural question, investigated and resolved" below — this was
  tested head-to-head on the same near-singular scenario that motivated
  this delivery's `accel_cap`, and found to be *worse*, not better: it
  produces a `Omega_dot_d` feedforward of ~9000 rad/s^2 in that scenario
  (a physically meaningless value) versus ~0.003 rad/s^2 for this
  delivery's capped design, with no analogous safeguard of its own. Their
  reported Monte Carlo finding of a genuine "sustained actuator
  saturation" tube-safety failure at high obstacle density is very
  plausibly this exact mechanism (an enormous, essentially-garbage
  feedforward term dominating the torque command until the hard 1.5 N*m
  clamp absorbs it, tick after tick, near a close obstacle) — not, as
  characterized there, an inherent, architecture-independent physical
  limit. This delivery's `accel_cap` was NOT re-run through their
  Monte-Carlo harness to confirm the failure rate actually improves
  (see "Not done" below) — the claim above is mechanism-level evidence,
  not a re-measured outcome.
- **Reported, not independently re-verified**: their Monte Carlo scaling
  study's raw numbers (`out/scaling_summary.csv` in their delivery,
  inspected directly here) — real data, not fabricated, showing both
  their design and the original losing tube-safety in 1 of 4 trials at
  `n_obs=25` (down from 100% at lower density). Worth knowing as
  evidence that dense obstacle fields can exhaust this vehicle's thrust
  authority regardless of Stage-1 design, even though the specific
  mechanism they attributed it to is disputed above.

### Open architectural question, investigated and resolved (see above)

There were two plausible designs for the Stage-1 reference-generator jet
chain: (a) this delivery's self-contained Taylor/Picard propagation of
the *idealized* commanded trajectory (no dependency on real-time attitude
state), or (b) the parallel effort's bootstrap off the vehicle's *actual*
current attitude/omega (arguably more physically faithful moment-to-moment,
at the cost of reintroducing the original architecture's
feedback-into-reference-generation coupling). This was tested head-to-head
(see above) and (a) was kept: (b) does not avoid the log-barrier blow-up
near an obstacle at all — it's arguably worse, since it stays numerically
finite (no NaN) but wildly unreasonable in magnitude, relying entirely on
downstream thrust/torque saturation to hide the problem rather than
preventing it. If you want to pursue (b) further anyway (e.g. because
per-tick physical fidelity matters more for your use case), the
comparison script used for this test is not included in this delivery but
is straightforward to reconstruct from the description above and from
`ga_stt_controller.py`'s `u1_jet` vs. a bootstrap-style alternative.

## Stage-1 gain tuning: a verified relationship to ρ_p, t_c, and tube shrink rate

Closes a gap left explicitly open earlier ("Stage 1 does not have a
simple time constant"). `ga_stt_py/mujoco_validation/STAGE1_GAIN_TUNING_ANALYSIS.md`
derives, and verifies against the exact nonlinear Stage-1 ODE (not just
its linearization), two genuinely new results:

- **`kappa_v > mu_max`** (`mu_max := max(-rho_dot/rho)`, the tube's own
  worst-case shrink rate) and **`kappa1 > mu_max*kappa_v*rho_min^2/4`** —
  local (Routh-Hurwitz) necessary stability conditions from a proper
  linearization that, unlike the plain `omega_n`/`zeta` heuristic already
  in `tune_gains.py`, keeps the tube's own time-varying radius in the
  picture. Verified: 16 combinations of `kappa_v`/`mu` run through the
  exact nonlinear equations with a synthetic, exactly-imposed shrink
  rate — every single one matched the predicted stable/unstable outcome.
- **A `t_c`-based lower bound on `kappa1`** from a settling-time argument
  (`kappa1 >= (2*N*rho/(zeta*t_c))^2`) — also verified: predicted
  threshold correctly separated a settling time that missed a `t_c/N`
  target (3.18s vs. 2.0s) from ones that met it (1.74s, 1.01s).

`tune_gains.py` now has a `--mu-max` flag that applies these bounds
automatically, on top of its existing `omega_n`/`zeta` search.

A second parallel-effort document (`stage1_tuning.md`, preserved as
`stage1_tuning_ORIGINAL.md`) proposing a related but different
methodology was reviewed the same way as everything else from that
source: its Constraint 2 (critical damping) and Constraint 3 (thrust
physical-feasibility budget) were checked and are correct, and are
folded into the tuning algorithm above. **Its Constraint 1 (a proposed
upper bound on `kappa1`) was checked and found to produce unstable
gains from its own worked example** — traced to two compounding errors:
a dropped factor of `1/rho` in its `M(e1_bar)` definition (confirmed
symbolically), and, more seriously, a 4th-power dependence on an
arbitrarily-chosen free parameter (`||z_v||_min`, "the minimum expected
velocity error") that collapses the bound to a near-zero `kappa1` for
any physically-reasonable small choice. Directly verified: their own
prescribed `(kappa1, kappa_v) = (3e-6, 0.008)` for a scenario with
`mu=0.8` leaves the tube at `t=1.53s` — the exact shrink rate their
formula was built to protect against. This is the same *class* of
mistake found earlier in a different document from the same effort (the
`lambda_R^min` formula in `TIME_SCALE_SEPARATION_ANALYSIS.md`, which
used an assumed error bound `e_bar_R` in the wrong direction) — worth
extra scrutiny on any future bound from that source built around a
similarly free "assumed typical/minimum" parameter.

## Not done / open items

- **The LaTeX paper (`GA_STT_Corrected.pdf`/`.tex`) reflects the Stage-1
  fixes but NOT the Stage-R reversion above** — its Corrigendum 6
  (sign convention for the vee-map replacement) describes a law that is
  no longer used in the code. Needs a rewrite of Section 2.2/4's
  treatment of Stage R to present the actual exact identity
  (`dV_R/dt = -kR*xi_R*eps_R^2`) instead.
- **No full Monte Carlo scaling study was run against this delivery's
  controller in this environment** (by design, per this session's
  instructions — batch runs are meant to happen on your own machine).
  `ga_stt_py/mujoco_validation/monte_carlo_cf2.py` is a ready-to-run
  harness for the real Crazyflie 2 model (smoke-tested here with 1-2
  trials, not run as a full sweep); see
  `TUNING_AND_MUJOCO_GUIDE.md` for how to run it and what to look for.
  A parallel effort's own Monte Carlo study (different harness, not
  included here) was reviewed and is discussed above.
- ~~**No hardware validation. No motor/ESC dynamics, sensor noise, or
  communication delay modeled anywhere**~~ **ADDRESSED.**
  `ga_stt_py/mujoco_validation/hardware_realism.py` adds a motor mixer,
  first-order motor/ESC lag, sensor noise, and comm delay;
  `run_x500_validation_realistic.py` and `run_cf2_validation_realistic.py`
  use them. **Important finding, not just a feature add**: the
  ideal-actuator gains and tube schedules used elsewhere in this delivery
  do NOT survive literature-typical motor lag (~20-50ms) on either
  vehicle -- a gain-selection methodology (verified and corrected from a
  parallel effort's proposal) was needed, AND the tube schedule itself
  had to be relaxed (slower, more clearance) before a stable, comfortable
  operating point was found. See
  `ga_stt_py/mujoco_validation/TIME_SCALE_SEPARATION_ANALYSIS.md` for the
  full account, including what in the proposed methodology was verified
  correct, what was found and corrected as wrong (a sign/direction error
  in a "worst-case rate" formula), and the specific negative results
  (which gain/schedule combinations were tried and still failed). Still
  no real hardware measurement backing any of the motor-lag or sensor-
  noise numeric values — see that document's own "what wasn't done"
  section, and each `_realistic` script's docstring, for exactly which
  assumptions remain unverified.
- **A Holybro X500 model, built from the Gazebo/PX4 SDF provided**
  (`ga_stt_py/mujoco_validation/x500/x500.xml`) — a primitive-geometry
  reconstruction (mesh files referenced in the SDF weren't included in
  what was uploaded), but with mass, inertia (parallel-axis corrected for
  the 4 rotor point masses, cross-checked against MuJoCo's own composite
  rigid-body dynamics to 4 significant figures), and rotor geometry
  transcribed exactly from the SDF. `run_x500_validation.py` (ideal
  actuators, tuned with `tune_gains.py`) and
  `run_x500_validation_realistic.py` (motor lag + noise + delay) both
  pass.
- **`f_max=0.60 N` and `tau_max=0.012 N*m` for the Crazyflie 2 validation
  are literature-based assumptions, not hardware-measured** — see
  `run_cf2_validation.py`'s docstring.
- **`kappa_v` was added to `run_scaling_study.py`, `run_scenario2.py`,
  and `test_noise_sensitivity.py`'s `DroneParams` calls (a required
  parameter now) but NOT independently re-tuned for those specific
  masses/scenarios** — only for the drone in
  `test_closed_loop_convergence.py` and, separately, for the real
  Crazyflie 2 in the `mujoco_validation/` scripts. Re-verify before
  trusting those other scripts' quantitative output for the corrected
  law. These three scripts also still use `ga_stt_controller.py`
  (vee-map Stage-R), not `ga_stt_controller_final.py` — not yet updated
  to the reverted law.
- **The C++ firmware wrapper (`controller_ga_stt.cpp`) was updated
  (added a `kappa_v` tunable parameter, updated the `DroneParams`
  construction) but could not be compiled/tested end-to-end** — it
  depends on the Crazyflie firmware SDK headers, which are not available
  in this environment. Only syntactic/structural review was possible
  there; the underlying `ga_stt_controller.hpp` it calls into (Stage-R
  now reverted, same as the Python `_final` version) is fully tested via
  the standalone test binaries.

## How to verify (re-run these yourself)

```bash
# C++ (no cmake needed, per your original instructions)
cd ga_stt_cpp
INC="-I controller/ga_stt -I /usr/include/eigen3"   # apt install libeigen3-dev if missing
for f in tests/test_*.cpp; do
  g++ -std=c++17 -O2 -Wall -Wextra $INC "$f" -o "build_g++/$(basename $f .cpp)"
done
for t in build_g++/*; do echo "-- $t --"; "$t"; done

# Python
cd ga_stt_py
for t in tests/test_*.py; do PYTHONPATH=lib python3 "$t"; done

# MuJoCo cross-validation, generic free body (pip install mujoco)
cd ga_stt_py/mujoco_validation
python3 run_validation.py

# MuJoCo cross-validation, REAL Crazyflie 2 model (same pip install)
python3 run_cf2_validation.py
# then see TUNING_AND_MUJOCO_GUIDE.md for re-tuning and the Monte Carlo sweep
```
