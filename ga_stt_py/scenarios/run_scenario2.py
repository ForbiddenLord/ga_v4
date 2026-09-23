"""
run_scenario2.py — Scenario 2: Aggressive flip recovery from θ_e ≈ 180°.

Test design: position p is locked to σ(t) throughout, isolating the
rotational dynamics.  This is the standard approach and avoids conflating
translational divergence (the drone can't translate upward when inverted) 
with the rotational claim.

The 'singularity-free' result: GA-STT evaluates rotor_log_angle_axis and
sin(θ_e/2) at exactly π without any special-casing; the baseline DCM
construction hits a numerical singularity when the thrust direction aligns
with the DCM's degenerate axis (visible as the DCM failure count).
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import pickle
import time
import matplotlib
matplotlib.use("Agg")        # static plot
import matplotlib.pyplot as plt

import ga3 as ga
import stt
import ga_stt_controller as ctrl
import baseline_controller as bctrl

# ── Scenario parameters ────────────────────────────────────────────────────
DRONE = ctrl.DroneParams(
    mass=0.5, J_diag=[3.0e-3, 3.0e-3, 5.5e-3],
    # kappa_v added (translational velocity-damping gain, required by the
    # corrected Stage-1 law -- see ga_stt_controller.py docstring). Not
    # independently re-tuned for this scenario; re-verify before trusting
    # this figure for the corrected law.
    f_min=0.0, f_max=20.0, kappa1=3.0, kappa_v=3.0, kR=6.0, kOmega=0.4
)
STT_PARAMS = stt.STTParams(
    eta=[0, 0, 2], s0=[0, 0, 0.1], tc=12.0,
    k1=0.05, k2=0.0, k3=0.0,
    rho_max=3.0, rho_min=0.2, u=8.0,
    rhoR0=2.0, rhoR_inf=0.1, kR=0.4
)
OBSTACLES = []          # no obstacles — pure attitude recovery
T_END = 11.8        # s
DT = 0.005       # s
FLIP_ANGLE = np.pi - 0.001   # 179.94-deg (avoids exactly π in arccos)
E1 = np.array([1.0, 0.0, 0.0])
PKL_PATH = os.path.join(tempfile.gettempdir(), "scenario2_results.pkl")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..",
                        "out", "scenario2_flip.png")

N_SWEEP = 30          # number of angles in the sweep panel
ANGLES = np.linspace(0.05, np.pi - 0.001, N_SWEEP)


def _run_attitude_only(ctrl_fn, R_init, **ckwargs):
    """
    Integrate only the attitude (R, Ω_b) while p tracks σ(t) exactly.
    Returns arrays: theta_e [N+1], dcm_ok [N+1].
    """
    R = R_init.copy()
    Omega_b = np.zeros(3)
    sigma = STT_PARAMS.s0.copy()
    J = DRONE.J_diag
    n_steps = int(T_END / DT) + 1

    theta_e_hist = []
    dcm_ok_hist = []

    for i in range(n_steps):
        t = i * DT
        ud = dict(p=sigma.copy(),
                  v=stt.sigma_value(sigma, t, OBSTACLES, STT_PARAMS).copy(),
                  R=R, Omega_b=Omega_b, sigma=sigma.copy())

        f, tau, diag = ctrl_fn(t, ud, OBSTACLES, STT_PARAMS, DRONE, **ckwargs)
        theta_e_hist.append(diag.get("theta_e", np.nan))
        dcm_ok_hist.append(diag.get("dcm_ok", True))

        if not np.isfinite(f):
            break

        sigma_dot = stt.sigma_value(sigma, t, OBSTACLES, STT_PARAMS)
        sigma = sigma + sigma_dot * DT
        Omega_dot = (tau - np.cross(Omega_b, J * Omega_b)) / J
        Rdot = -0.5 * ga.geometric_product(R, ga.bivector(Omega_b))
        R = ga.normalize_rotor(R + Rdot * DT)
        Omega_b = Omega_b + Omega_dot * DT

    return np.array(theta_e_hist), np.array(dcm_ok_hist)


def _theta_at_1s(ctrl_fn, R_init, reset_fn=None, **ckwargs):
    """Single-number summary: θ_e at t=1s for a given initial rotor."""
    if reset_fn:
        reset_fn()
    R = R_init.copy()
    Omega_b = np.zeros(3)
    sigma = STT_PARAMS.s0.copy()
    J = DRONE.J_diag
    n_1s = int(1.0 / DT) + 1
    dcm_failures = 0

    for i in range(n_1s):
        t = i * DT
        ud = dict(p=sigma.copy(),
                  v=stt.sigma_value(sigma, t, OBSTACLES, STT_PARAMS).copy(),
                  R=R, Omega_b=Omega_b, sigma=sigma.copy())
        f, tau, diag = ctrl_fn(t, ud, OBSTACLES, STT_PARAMS, DRONE, **ckwargs)
        if not diag.get("dcm_ok", True):
            dcm_failures += 1
        if not np.isfinite(f):
            return np.nan, dcm_failures

        sigma_dot = stt.sigma_value(sigma, t, OBSTACLES, STT_PARAMS)
        sigma = sigma + sigma_dot * DT
        Omega_dot = (tau - np.cross(Omega_b, J * Omega_b)) / J
        Rdot = -0.5 * ga.geometric_product(R, ga.bivector(Omega_b))
        R = ga.normalize_rotor(R + Rdot * DT)
        Omega_b = Omega_b + Omega_dot * DT

    _, n_hat = ga.rotor_log_angle_axis(ga.normalize_rotor(
        ga.geometric_product(ga.reverse(ga.rotor_identity()), R)))
    # θ_e from the identity target (Rd = identity):
    _, n_hat2 = ga.rotor_log_angle_axis(ga.normalize_rotor(R))
    theta_e_final = ga.rotor_log_angle_axis(ga.normalize_rotor(R))[0]
    return theta_e_final, dcm_failures


# ── Panel A: 180-deg flip time history ────────────────────────────────────
print("=== Scenario 2: Flip Recovery ===")
R_flip = ga.exp_bivector(E1, FLIP_ANGLE)

print(f"Panel A: time history from {np.degrees(FLIP_ANGLE):.1f}° flip...")
t0 = time.time()
theta_ga, dcm_ga = _run_attitude_only(
    ctrl.compute_control, R_flip, Rp_value=ga.rotor_identity())
print(f"  GA-STT : {time.time()-t0:.1f}s  final θ_e={theta_ga[-1]:.5f} rad")

bctrl.reset_baseline_state()
t0 = time.time()
theta_bl, dcm_bl = _run_attitude_only(
    bctrl.compute_control_baseline, R_flip)
dcm_fail_A = int(np.sum(~dcm_bl))
print(f"  Baseline: {time.time()-t0:.1f}s  final θ_e={theta_bl[-1]:.5f} rad  "
      f"DCM SVD failures={dcm_fail_A}")

t_arr = np.arange(len(theta_ga)) * DT

# ── Panel B: angle sweep ──────────────────────────────────────────────────
print(
    f"Panel B: sweeping {N_SWEEP} initial angles 5°→{np.degrees(ANGLES[-1]):.1f}°...")
ga_at_1s = []
bl_at_1s = []
bl_dcm_fails = []

for ang in ANGLES:
    R_init = ga.exp_bivector(E1, ang)

    th_ga, _ = _theta_at_1s(ctrl.compute_control, R_init,
                            Rp_value=ga.rotor_identity())
    ga_at_1s.append(th_ga)

    bctrl.reset_baseline_state()
    th_bl, n_fail = _theta_at_1s(bctrl.compute_control_baseline, R_init)
    bl_at_1s.append(th_bl)
    bl_dcm_fails.append(n_fail)

ga_at_1s = np.array(ga_at_1s)
bl_at_1s = np.array(bl_at_1s)
print(f"  GA-STT  max θ_e@1s: {np.nanmax(ga_at_1s):.4f} rad")
print(f"  Baseline max θ_e@1s: {np.nanmax(bl_at_1s):.4f} rad")

# ── Save pkl ──────────────────────────────────────────────────────────────
with open(PKL_PATH, "wb") as fh:
    pickle.dump(dict(
        t=t_arr, theta_ga=theta_ga, theta_bl=theta_bl,
        dcm_fail_A=dcm_fail_A, flip_angle_deg=np.degrees(FLIP_ANGLE),
        angles_deg=np.degrees(ANGLES),
        ga_at_1s=ga_at_1s, bl_at_1s=bl_at_1s,
        bl_dcm_fails=bl_dcm_fails,
    ), fh)
print(f"Saved {PKL_PATH}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Scenario 2: Flip Recovery — GA-STT vs Baseline (p locked to σ(t))",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(t_arr, theta_ga, "tab:blue",   lw=2, label="GA-STT")
ax.plot(t_arr, theta_bl, "tab:orange", lw=2, label="Baseline (Lee + FD Ω_d)")
ax.axhline(np.pi, color="gray", ls=":", lw=1,
           label=f"initial θ_e ≈ {np.degrees(FLIP_ANGLE):.0f}°")
ax.set_xlabel("t [s]")
ax.set_ylabel(r"$\theta_e$ [rad]")
ax.set_title(f"Time history from {np.degrees(FLIP_ANGLE):.0f}° flip\n"
             f"(Baseline: {dcm_fail_A} DCM SVD failures)")
ax.legend(fontsize=9)
ax.set_xlim([0, T_END])

ax = axes[1]
ax.plot(np.degrees(ANGLES), ga_at_1s, "o-", color="tab:blue",
        ms=5, label="GA-STT  θ_e at t=1s")
ax.plot(np.degrees(ANGLES), bl_at_1s, "s-", color="tab:orange",
        ms=5, label="Baseline θ_e at t=1s")
ax.axhline(np.pi, color="gray", ls=":", lw=1, label="π (no recovery)")
ax.set_xlabel("Initial flip angle [deg]")
ax.set_ylabel(r"$\theta_e$ at $t=1$ s [rad]")
ax.set_title("Recovery speed vs initial angle\n(lower = faster)")
ax.legend(fontsize=9)
ax.set_xlim([0, 180])

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
fig.savefig(OUT_PATH, dpi=130)
# plt.show()             # uncomment for interactive plot
print(f"Saved {OUT_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────
t1s_idx = int(1.0 / DT)
print("\n=== Summary ===")
print(f"{'Controller':<12}  {'θ_e initial':>11}  {'θ_e at t=1s':>11}  {'θ_e final':>9}  {'DCM fails':>9}")
print(f"{'GA-STT':<12}  {theta_ga[0]:>11.4f}  {theta_ga[min(t1s_idx, len(theta_ga)-1)]:>11.4f}"
      f"  {theta_ga[-1]:>9.4f}  {'N/A':>9}")
print(f"{'Baseline':<12}  {theta_bl[0]:>11.4f}  {theta_bl[min(t1s_idx, len(theta_bl)-1)]:>11.4f}"
      f"  {theta_bl[-1]:>9.4f}  {dcm_fail_A:>9}")
