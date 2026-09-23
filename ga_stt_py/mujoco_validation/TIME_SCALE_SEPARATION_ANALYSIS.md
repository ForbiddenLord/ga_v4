# Motor/ESC dynamics, sensor noise, comm delay, and a time-scale-separation
# tuning methodology: what was verified, corrected, and found

This documents work done in response to two things: (1) a direct question
about whether motor/ESC dynamics, sensor noise, and communication delay
had been tested in MuJoCo (they had not, until now), and (2) a
`convergence_based_tuning_method.md` write-up from a parallel effort
proposing a time-scale-separation methodology to pick gains that survive
motor lag, with an explicit request to verify rather than trust it.

## What's new

- `hardware_realism.py`: a reusable module providing a quadrotor motor
  mixer, first-order motor/ESC lag, sensor noise injection, and a
  communication-delay ring buffer. Used by all four scripts below.
- `run_x500_validation_realistic.py`, `run_cf2_validation_realistic.py`:
  closed-loop MuJoCo validation of both vehicles WITH these three effects
  active, as opposed to the existing `run_x500_validation.py` /
  `run_cf2_validation.py`, which remain the ideal-actuator baselines.
- This document, and the gain-selection method behind the `_realistic`
  scripts' `KAPPA1, KAPPA_V, KR, KOMEGA` constants.

## The uploaded methodology: what was verified, what was corrected

`convergence_based_tuning_method.md` claims two exact closed-form
Lyapunov decay rates and proposes deriving gains from them for a
time-scale-separated cascade `tau_motor << tau_Omega << tau_R << tau_Stage1`.

**Stage Omega's rate, `lambda_Omega = 2*kappa_Omega/lambda_max(J)`:**
confirmed correct. This is a standard Rayleigh-quotient comparison-lemma
argument applied to the EXACT identity `dV_Omega/dt = -kappa_Omega*||z_Omega||^2`
(Section 2.3/Step C of the corrected derivation, unchanged since the
original PDF and re-verified multiple times earlier in this project).

**Stage R's rate, `lambda_R(t) = 2*kappa_R/(rho_R(t)*(1-e_R(t)))`:**
confirmed correct -- and gratifyingly, this is EXACTLY the identity
`dV_R/dt = -kR*xi_R(eps_R)*eps_R^2` independently re-derived (from the GA
kinematics directly, not from this document) during the Stage-R-reversion
work earlier in this project. Two independent derivations of the same
result is good evidence it's right.

**Their `lambda_R^min` formula -- found to be WRONG, corrected here.**
They propose `lambda_R^min = 2*kappa_R / (rho_R,max*(1-e_bar_R))` as a
worst-case (slowest-possible) guaranteed rate, where `e_bar_R` is a
design ceiling on `e_R`. This is backwards: `(1-e_R)` DECREASES as `e_R`
increases, so bounding `e_R <= e_bar_R` gives `(1-e_R) >= (1-e_bar_R)`, a
LOWER bound on `(1-e_R)`, not an upper one -- but the denominator
`rho_R(t)*(1-e_R(t))` needs an UPPER bound to get a true worst-case
(minimum) rate, since `lambda_R` is smallest when the denominator is
LARGEST. The true supremum of `(1-e_R(t))` is 1 (approached whenever
tracking is good, `e_R -> 0`), which can happen at any time, including
near `t=0` when `rho_R(t)` is also at its largest (`rho_R,0`). Using
`e_bar_R` in this formula therefore UNDERESTIMATES the true worst-case
denominator and OVERESTIMATES the guaranteed minimum rate -- i.e., their
formula is optimistic, not conservative. Verified numerically: for
`rho_R0=1.0, kappa_R=1.111, e_bar_R=0.5`, their formula gives
`lambda_R^min=4.444`; the rigorous bound (using denominator `rho_R,0 * 1`,
no `e_bar_R` term) gives `lambda_R^min=2.222` -- exactly half.

**Corrected design formula used here:**
```
kappa_R = lambda_R_target * rho_R,0 / 2      (no e_bar_R discount)
```
This is what `run_x500_validation_realistic.py` and
`run_cf2_validation_realistic.py` actually use.

## The gains actually used, and how they were derived

For each vehicle: pick a literature-typical `tau_motor`, then apply a 3x
separation factor at each stage (the document's own heuristic, flagged
there as "a thing to be verified" -- not independently re-derived or
justified further here beyond confirming it empirically works when
combined with the correction above and the tube relaxation below):

```
tau_Omega = 3 * tau_motor         -> kappa_Omega = (1/tau_Omega) * lambda_max(J) / 2
tau_R     = 3 * tau_Omega         -> kappa_R     = (1/tau_R) * rho_R,0 / 2   (corrected formula)
```

| | X500 | CF2 |
|---|---|---|
| `tau_motor` (assumption) | 0.05 s | 0.02 s |
| `tau_Omega` -> `kappa_Omega` | 0.15 s -> 0.1463 | 0.06 s -> 0.00027 |
| `tau_R` -> `kappa_R` | 0.45 s -> 1.111 | 0.18 s -> 2.778 |
| `kappa1`, `kappa_v` | 0.08, 0.849 | 0.2, 2.68 |

`kappa1`/`kappa_v` were NOT derived from a formula (Stage 1 has no clean
time constant, as the document itself acknowledges) -- they were found by
sweeping, described next.

## The important finding the uploaded document doesn't address, found by testing

The document's own summary table lists Stage 1 as having "no clean
exponential rate" and suggests, in one sentence, "you can make the
translational gains small enough that the translational loop is slow."
Testing this directly (both vehicles, both the theoretically-motivated
`kappa_Omega`/`kappa_R` AND several `kappa1` sweeps) found this advice
incomplete in a way that matters:

- **Making Stage 1 too fast** reproduces the original oscillatory
  instability (motor lag interacting with an aggressive position loop --
  `theta_e` growing with each oscillation, e.g. 0.02 -> 0.78 -> 1.14 ->
  1.42 -> 2.53 rad across a handful of cycles, confirmed on the X500 with
  ZERO initial attitude error, ruling out "just a bad initial transient"
  as the explanation).
- **Making Stage 1 too slow** causes a DIFFERENT failure: the vehicle
  falls behind the STT tube's own prescribed-time schedule (`tc`) and
  overshoots/oscillates in position space badly enough to leave the tube
  (`e_p > 1`) -- confirmed on the X500 with `kappa1=0.05`: attitude
  tracking stays fine (`theta_e` under 0.26 rad throughout) while
  position error grows steadily to `e_p=0.96` by `t=1s`, then oscillates
  and eventually exceeds 1 at `t=2.94s`. A completely different
  mechanism from the fast-gain case, but equally fatal.
- **For the two vehicles' ORIGINAL (ideal-actuator-tuned) tube
  schedules** (X500: `rho_max=0.8, tc=10.0`; CF2: `rho_max=0.35, tc=8.0`)
  under literature-typical motor lag, NO `kappa1` value found -- fast,
  slow, or in between, with the theoretically-correct `kappa_Omega`/
  `kappa_R` held fixed -- kept the vehicle inside the tube for the full
  run. This was checked with a reasonably fine sweep, not just a couple
  of points.
- **Relaxing the tube schedule** (X500: `rho_max=1.2, tc=15.0`; CF2:
  `rho_max=0.6, tc=14.0` -- both still real, flyable missions, just less
  aggressive) DOES restore a comfortable margin (X500: max `e_p=0.86`
  with `kappa1=0.08`, tighter schedules tested down to `rho_max=1.0,
  tc=12.0` still diverged; CF2: max `e_p=0.23` with `kappa1=0.2`). These
  are the schedules and gains the `_realistic` scripts actually use.

**The missing constraint**: Stage 1 needs to be simultaneously (a) slow
enough relative to Stage R (the document's constraint) and (b) fast
enough relative to the tube's own schedule `tc` (not addressed by the
document at all). These two constraints define a window that may or may
not be non-empty depending on how tight the tube is relative to how slow
motor lag forces the inner loops to be. For the two ORIGINAL aggressive
schedules used elsewhere in this delivery, under the specific
`tau_motor` assumptions used here, that window was empirically found to
be empty (or at least, not found by search); for a relaxed schedule, it
is not.

## What this means for the rest of this delivery

`run_x500_validation.py` and `run_cf2_validation.py` (the ideal-actuator
baselines) are unaffected by any of this -- they don't model motor lag,
so the time-scale-separation constraint doesn't apply to them, and their
existing gains and tube schedules remain valid for what they test.

The `_realistic` scripts use DIFFERENT gains AND a different (relaxed)
tube schedule than their ideal-actuator counterparts, for the reasons
above. This is a genuine, reproducible finding about this controller
architecture's sensitivity to actuator lag on an aggressive mission
profile, not an artifact of a specific implementation bug (the mixer
round-tripped exactly in isolated testing before any of this; the
divergence mechanism was traced in both its fast-gain oscillatory form
and its slow-gain tube-overshoot form).

## What wasn't done

- The `3x` separation factor is untested as a design choice beyond "it
  happened to work once corrected and combined with a relaxed tube" --
  no attempt was made to find the actual minimum separation factor that
  works, or to check whether e.g. `2x` also works or `4x` is needed for
  more aggressive missions.
- No attempt was made to find the boundary of the feasible
  `(rho_max, tc)` window more precisely than the handful of points
  tested above (X500: `rho_max=1.0, tc=12` fails, `rho_max=1.2, tc=15`
  succeeds -- the actual boundary is somewhere between and was not
  searched for further).
- Sensor noise and comm delay were tested TOGETHER with motor lag in the
  final `_realistic` scripts, but the earlier isolation testing (which
  found motor lag alone, even with noise and delay OFF, was sufficient
  to cause the original divergence) means their individual marginal
  contribution to the margin, if any, was not separately quantified.
- No hardware measurement backs any of `tau_motor`, `k_m`, the per-motor
  thrust cap multiplier, or the sensor noise standard deviations for
  either vehicle -- all are literature-typical order-of-magnitude
  choices, as stated in each script's own docstring.
