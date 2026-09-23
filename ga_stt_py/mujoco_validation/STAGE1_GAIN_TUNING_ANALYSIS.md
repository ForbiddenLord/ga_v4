# Tuning κ₁ and κ_v: a verified relationship to ρ_p, t_c, and STT tube motion

This establishes a rigorous, empirically-verified method for choosing the
Stage-1 gains, closing a gap explicitly left open earlier in this project
("Stage 1 has no clean exponential rate... its convergence depends on the
tube geometry and obstacle proximity"). It also reviews a second
`stage1_tuning.md` write-up from a parallel effort — verified rather than
trusted, per usual practice here, and **found to contain a real bug that
makes its central formula produce unstable gains**, demonstrated by
running its own prescribed numbers through the exact closed-loop dynamics
it was meant to protect.

Everything below was checked against the **exact** nonlinear Stage-1 ODE
(not just the linearization used to derive it), using a synthetic,
independently-imposed tube shrink-rate `μ(t) := -ρ̇(t)/ρ(t)` so each claim
could be tested in isolation from any particular STT obstacle scenario.

## 1. Setting: the exact Stage-1 dynamics, and where they leave off

Recall the exact identity (established earlier in this project,
`GA_STT_Corrected.pdf` Section 4):
```
dV1/dt = -κ_v ||z_v||²  -  κ1 · ε_p · ξ_p(ε_p) · e1 · (ρ̇/ρ)
```
with `V1 = ½κ1 ε_p² + ½||z_v||²`. The first term is always dissipative.
The second is a genuine, unavoidable destabilizing contribution whenever
the tube is shrinking (`ρ̇<0`, e.g. near an obstacle) — and critically,
**it does not vanish as `z_v→0`**: even a perfectly-tracking vehicle
(`z_v=0`) sees its normalized error `e1=|e_p|/ρ` grow automatically if
`ρ` itself shrinks under it. No choice of `κ_v` alone fixes this; that's
exactly why a naive "just add damping" argument is insufficient, and why
the earlier writeup correctly flagged this as having no clean rate.

## 2. The missing piece: a local (near-tube-center) linear stability condition

Linearizing near `e1→0` (`ε_p≈2e1`, `ξ_p≈2/ρ`) and reducing to the radial
scalar `s := z_v·p̂` (velocity error along the position-error direction)
gives a genuinely new, clean 2-state linear system with `μ(t)` and `ρ(t)`
entering explicitly:
```
ė1 = μ·e1 + s/ρ
ṡ  = -(4κ1/ρ)·e1 - κ_v·s
```
(Derived and cross-checked symbolically with `sympy`.) This is an
ordinary 2×2 linear time-varying system; freezing `μ, ρ` locally and
applying the Routh-Hurwitz criterion to the resulting constant matrix
gives two necessary conditions for local asymptotic stability:

```
Trace condition:     κ_v > μ                     (μ := -ρ̇/ρ, the tube's own shrink rate)
Determinant condition: κ1 > μ·κ_v·ρ² / 4
```

**The trace condition is the headline result**: κ_v must exceed the
tube's own relative shrink rate. This has a direct physical reading —
`μ` is itself a destabilizing self-feedback rate on `e1` (a tube
shrinking at rate `μ` grows `e1` at that same rate even with zero
tracking error), so the velocity-error damping `κ_v` must out-pace it.

**Verified against the exact nonlinear ODE** (not the linearization):
16 combinations of `κ_v ∈ {0.5,1.0,1.5,2.0}` and `μ ∈ {0.3,0.8,1.2,2.0}`
(ρ₀=0.8, κ1=5 fixed well above the determinant threshold) were run
through the full nonlinear equations with a synthetic
`ρ(t)=ρ₀e^{-μt}` (exact shrink rate, decoupled from any specific
obstacle scenario). **Every single case matched the prediction**: all 9
cases with `κ_v > μ` stayed inside the tube (or reached a bounded
`max e1 < 1`); all 7 cases with `κ_v ≤ μ` left the tube (`e1→1`), with
time-to-violation shrinking as the margin `κ_v - μ` shrank (e.g.
`κ_v=0.5,μ=0.3`: survives 5.38s before the horizon; `κ_v=2.0,μ=2.0`:
violates almost immediately, t=0.91s) — exactly the qualitative behavior
a marginal-stability boundary should produce.

## 3. Relevance to t_c: a lower bound on κ1 from settling time

`κ_v > μ` and the determinant condition are necessary for local
stability, but say nothing about whether the loop is fast enough to
matter within the mission's actual duration. The STT's own prescribed
time `t_c` sets a natural requirement: the position loop should settle
well before `t_c`, or reaching the target set at `t_c` is a
technicality, not a meaningful guarantee.

Using the same linearization's natural frequency `ω_n = 2√κ1/ρ` and a
standard 2%-settling-time estimate for a near-critically-damped 2nd order
system, `t_settle ≈ 4/(ζω_n)`, requiring `t_settle ≤ t_c/N` for a design
margin `N` (e.g. `N=5`, i.e. want to settle by 1/5 of the mission
duration) gives:

```
κ1 ≥ (2·N·ρ / (ζ·t_c))²
```

**Verified**: for `ρ=0.8, t_c=10, N=5, ζ=1` (predicted `κ1_min=0.64`),
running the exact nonlinear ODE at `0.3×`, `1.0×`, and `3.0×` this value
gave settling times of 3.18s (exceeds the 2.0s target — FAILS), 1.74s,
and 1.01s (both meet the target) — a clean, correctly-predicted
transition right around the derived threshold.

## 4. Critical damping: the κ_v–κ1 relationship for a chosen response shape

Standard for a 2nd-order system, and consistent between this analysis,
the earlier `tune_gains.py` heuristic, and the parallel effort's writeup
(this part of their document is correct and not in dispute):
```
ω_n = 2√κ1/ρ,   ζ = κ_v·ρ / (4√κ1)   =>   κ_v = 4ζ√κ1/ρ
```
Pick `ζ≈0.9-1.0` (critically to slightly underdamped: fast without
overshoot) once `κ1` is chosen from Sections 2-3.

## 5. Physical feasibility: thrust budget (adopted from the parallel effort, verified correct)

The parallel effort's Constraint 3 was checked and found to be correctly
derived, unlike their Constraint 1 (Section 6). From
`u1 = σ̈ - κ_v z_v - κ1 ξ_p(ε_p) ε_p p̂` and the triangle inequality:
```
||u1|| ≤ ||σ̈||_max + κ_v ||z_v||_max + κ1 · K_bar(ē1),   K_bar(ē1) := ε_p(ē1)·ξ_p(ē1)
```
(`ē1` a chosen worst-case normalized error, e.g. 0.8). Checked
`K_bar(ē1) = (1/ρ)·ln((1+ē1)/(1-ē1))·2/(1-ē1²)` **is** dimensionally the
true per-unit-κ1 magnitude of the spring term (verified symbolically:
equals `ε_p(ē1)ξ_p(ē1)` exactly, using the correct `ξ_p=2/(ρ(1-e1²))`,
unlike their Constraint-1 quantity below). Requiring
`||u1|| ≤ f_max/m + g` gives an upper feasibility bound:
```
κ_v·||z_v||_max + κ1·K_bar(ē1)  ≤  f_max/m + g - ||σ̈||_max
```
If the gains from Sections 2-4 violate this, either relax `ē1` (accept
closer approach to the tube wall before saturating) or accept a longer
`t_c` (Section 3's bound allows smaller κ1).

## 6. What was checked and rejected from the parallel effort's Constraint 1

Their Constraint 1 proposes an *upper* bound on κ1 (opposite direction
from Section 3's lower bound) derived from requiring `dV1/dt<0` whenever
`||z_v|| > ||z_v||_min`, a free design parameter they call "the minimum
expected velocity error." Two problems, both confirmed:

**A dropped factor of `1/ρ`.** Their `M(ē1) := ε_p(ē1)·[2ē1/(1-ē1²)]`
implicitly uses `ξ_p = 2/(1-e1²)` — missing the `1/ρ` that the real
`ξ_p = 2/(ρ(1-e1²))` has. Checked symbolically: the ratio of the true
cross-term coefficient to their `M(ē1)` is exactly `1/ρ`, not 1.

**A much larger, dominant error: hypersensitivity to their own free
parameter.** Their final formula is
`κ1_max = 16||z_v||⁴_min / (M(ē1)²γ_max²ρ²)` — a **fourth power** of a
value (`||z_v||_min`) the document asks the user to pick as "the minimum
expected velocity error" without further guidance. Picking a
physically-reasonable-sounding small value (0.05 m/s) with `ē1=0.8,
γ_max=0.8, ρ=0.8` gives `κ1_max ≈ 3×10⁻⁶` and (via their own Constraint
2) `κ_v ≈ 0.008` — **which is itself below the μ=0.8 trace-stability
threshold from Section 2** (`κ_v` must exceed `μ`; 0.008 ≪ 0.8).

**Directly verified this fails**: running their own prescribed
`(κ1, κ_v) = (3×10⁻⁶, 0.008)` through the exact nonlinear ODE at exactly
`μ=0.8` — the shrink rate their formula was built to protect against —
the tube is violated at `t=1.53s`. Their methodology, applied as
written, produces gains that fail the scenario it claims to guarantee
safety for. This is not a edge case or misapplication; it's their
worked example's own numbers.

Their Constraint 2 (critical damping) and Constraint 3 (thrust budget)
are correct and retained (Sections 4-5); Constraint 1 is not adopted.
(This is the same *class* of error found earlier in a different document
from the same effort — a "worst-case" formula built around a
free/assumed parameter that, taken to its natural extreme, breaks the
guarantee rather than tightening it. There, it was `ē_R` used in the
wrong direction; here, it's `||z_v||_min` entering as an unconstrained
4th power. Worth extra scrutiny on any future bound from the same
source that depends on a similarly free "assumed minimum/typical" value.)

## 7. Complete tuning algorithm

1. **Determine μ_max** for your scenario: the maximum relative tube
   shrink rate `-ρ̇/ρ` the STT generator can produce for your obstacle
   field (numerically, by simulating `stt.sigma_jet`/`rho_p_jet` over
   the mission and taking the max of `-ρ̇/ρ`, or conservatively from the
   STT construction's own worst-case obstacle-approach geometry — not
   derived in closed form here; numerical evaluation for your specific
   scenario is the practical route).
2. **Trace condition (Section 2)**: pick `κ_v > μ_max`, with margin
   (e.g. `κ_v ≥ 1.5·μ_max`).
3. **t_c lower bound (Section 3)**: compute
   `κ1_min_tc = (2·N·ρ_min/(ζ·t_c))²` (use `ρ_min`, the smallest tube
   radius your mission reaches, for the most conservative bound; `N≈5`,
   `ζ≈0.9`).
4. **Determinant condition (Section 2)**: check
   `κ1 > μ_max·κ_v·ρ_min²/4` at your chosen `κ_v`; raise `κ1` to satisfy
   this if `κ1_min_tc` doesn't already.
5. **Critical damping (Section 4)**: given the final `κ1`, set
   `κ_v = 4ζ√κ1/ρ` (recheck this still satisfies step 2's `κ_v > μ_max`;
   iterate if not — a genuinely infeasible combination, needing a
   relaxed mission, is a real possible outcome, not a bug, as found
   directly in the motor-lag investigation earlier in this project).
6. **Physical feasibility (Section 5)**: check the thrust budget
   inequality; relax `ē1` or `t_c` if it fails.
7. **If motor/ESC dynamics are modeled** (see
   `TIME_SCALE_SEPARATION_ANALYSIS.md`), separately confirm the
   resulting `κ1` is slow enough relative to the derived `κ_Ω, κ_R` from
   that document's (corrected) method — an independent, additional upper
   bound not captured by the steps above.

## 8. What wasn't done

- `μ_max` was tested here only against a synthetic, exactly-imposed
  exponential shrink rate, not computed end-to-end from a real STT
  obstacle field's `rho_p_jet` output and fed through this algorithm on
  an actual scenario. Doing so (numerically maximizing `-ρ̇/ρ` over a
  real mission) would close the loop; not done in this session.
- The Routh-Hurwitz conditions are LOCAL (linearized near `e1→0`); no
  attempt was made to extend them to a global nonlinear guarantee
  (e.g. via a proper ISS argument valid for all `e1∈[0,1)`), and the
  parallel effort's attempt at a more global bound (Constraint 1) was
  specifically the one found flawed.
- The settling-time constant "4" (2%-settling-time rule) and margin
  factor `N` are standard control-engineering conventions, not
  re-derived from the STT paper's own guarantees; different choices
  shift the numeric threshold but not the functional form.
