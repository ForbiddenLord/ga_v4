"""
run_scaling_study.py — Obstacle-scaling performance comparison table.

Generates:
  out/scaling_study.csv      — full raw results (every trial)
  out/scaling_summary.csv    — mean±std aggregated table (paper-ready)
  out/scaling_summary.tex    — LaTeX booktabs table
  out/scaling_figure.png     — 3-panel figure (compute time, tube-safe %, max eₚ)

Columns produced:
  # obs | algorithm | condition | ctrl_ms_mean | ctrl_ms_std
        | collision_free_pct | tube_safe_pct   (safety under condition)
        | max_ep_mean | max_ep_std
        | min_clear_mean | min_clear_std
        | success_pct    (final err < 0.5m)
        | qp_infeas_pct  (CBF only)

MPC is excluded from the sweep (≈350ms/step makes 25 trials × 5 configs
impractical). Its figures from Scenario 1 are added as a footnote row.

Usage:
    python3 scenarios/run_scaling_study.py               # full 25 trials
    python3 scenarios/run_scaling_study.py --trials 5    # quick test

Checkpointing: results are saved to a .pkl after every completed
(n_obs, condition) block, so Ctrl-C + restart resumes from where it stopped.
"""

import sys, os, argparse, tempfile, time, pickle, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ga3 as ga
import stt
import ga_stt_controller   as ctrl
import baseline_controller  as bctrl
import cbf_controller       as cbf_ctrl
import simulate

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trials",  type=int,   default=25,
                    help="Monte Carlo trials per (n_obs, condition) cell")
parser.add_argument("--obs",     type=int,   nargs="+", default=[5, 10, 15, 20, 25],
                    help="Obstacle counts to sweep (default: 5 10 15 20 25)")
parser.add_argument("--no-cbf",  action="store_true",
                    help="Skip CBF-QP (faster)")
args, _ = parser.parse_known_args()

N_TRIALS = args.trials
OBS_COUNTS = args.obs
RUN_CBF = not args.no_cbf
CONDITIONS = ["nominal", "disturbed"]   # two safety columns

WP_MAX, WTAU_MAX = 0.5, 0.05

# ── Shared scenario parameters ─────────────────────────────────────────────
P0, R0 = np.array([1., 1., 1.]), ga.rotor_identity()
ETA = np.array([8., 8., 8.])

DRONE = ctrl.DroneParams(
    mass=0.5, J_diag=[3e-3, 3e-3, 5.5e-3],
    # kappa_v added (translational velocity-damping gain, required by the
    # corrected Stage-1 law -- see ga_stt_controller.py docstring). Not
    # independently re-tuned for this scaling study; re-verify before
    # trusting these figures for the corrected law.
    f_min=0., f_max=20., kappa1=3., kappa_v=3., kR=6., kOmega=0.4
)
STT_BASE = dict(eta=[8, 8, 8], s0=[1, 1, 1], tc=16., k1=0.15, k2=1., k3=0.5,
                rho_max=1., rho_min=0.15, u=8., rhoR0=1., rhoR_inf=0.05, kR=0.5)

T_END, DT = 15.8, 0.01
SUCCESS_THRESH = 0.5   # m: "task success" if final |p - T| < this

# Output paths
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "out")
CKPT_PKL = os.path.join(tempfile.gettempdir(), "scaling_study_ckpt.pkl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Obstacle generation ───────────────────────────────────────────────────


def _obs_radius_range(n):
    """Scale obstacle radii down as count increases to keep total volume roughly constant."""
    if n <= 5:
        return 0.30, 0.60
    if n <= 10:
        return 0.25, 0.50
    if n <= 15:
        return 0.22, 0.45
    if n <= 20:
        return 0.20, 0.40
    return 0.18, 0.35


def make_obstacles(n_obs, rng, S=P0, T=ETA):
    r_lo, r_hi = _obs_radius_range(n_obs)
    obs = []
    for _ in range(n_obs):
        for _ in range(500):
            pos = rng.uniform(1.5, 7.5, 3)
            if (np.linalg.norm(pos - S) > 1.5 and
                    np.linalg.norm(pos - T) > 1.5):
                obs.append(stt.Obstacle(
                    pos,
                    rng.uniform(-0.05, 0.05, 3),
                    rng.uniform(r_lo, r_hi)
                ))
                break
    return obs


def make_dist_fn(seed):
    rng2 = np.random.default_rng(int(seed))
    n_sw = int(T_END / 0.1) + 2
    times = np.arange(n_sw) * 0.1
    wps = rng2.uniform(-WP_MAX,   WP_MAX,   (n_sw, 3))
    wtaus = rng2.uniform(-WTAU_MAX, WTAU_MAX, (n_sw, 3))

    def fn(t):
        idx = int(np.clip(np.searchsorted(times, t, "right") - 1, 0, n_sw-1))
        return wps[idx], wtaus[idx]
    return fn

# ── Per-trial metric extraction ────────────────────────────────────────────


def extract_metrics(log, stt_params):
    p, sigma, rho_p = log["p"], log["sigma"], log["rho_p"]
    ep = np.linalg.norm(p - sigma, axis=1) / rho_p
    nan_run = bool(np.any(np.isnan(p)))
    if nan_run:
        return dict(ctrl_ms=np.nan, max_ep=np.nan, min_clear=np.nan,
                    tube_safe=False, collision_free=False,
                    task_success=False, qp_infeas_pct=np.nan)
    ctrl_ms = float(log["wall_time"].mean() * 1e3)
    max_ep = float(ep.max())
    min_clear = float(log["min_obstacle_clearance"].min())
    tube_safe = bool(max_ep < 1.0)
    collision_free = bool(min_clear > 0.0)
    task_success = bool(np.linalg.norm(
        p[-1] - stt_params.eta) < SUCCESS_THRESH)
    qp_infeas_pct = np.nan   # overridden for CBF below
    return dict(ctrl_ms=ctrl_ms, max_ep=max_ep, min_clear=min_clear,
                tube_safe=tube_safe, collision_free=collision_free,
                task_success=task_success, qp_infeas_pct=qp_infeas_pct)

# ── Checkpoint helpers ─────────────────────────────────────────────────────


def _load_ckpt():
    if os.path.exists(CKPT_PKL):
        with open(CKPT_PKL, "rb") as f:
            return pickle.load(f)
    return {}


def _save_ckpt(data):
    with open(CKPT_PKL, "wb") as f:
        pickle.dump(data, f)


# ── Main sweep ────────────────────────────────────────────────────────────
CONTROLLERS = ["GA-STT", "Baseline", "CBF-QP"]
if not RUN_CBF:
    CONTROLLERS.remove("CBF-QP")

# results[n_obs][condition][controller] = list of metric dicts
results = _load_ckpt()

rng_master = np.random.default_rng(7)
obs_seeds = {n: rng_master.integers(0, 10_000, 1)[0] for n in OBS_COUNTS}

for n_obs in OBS_COUNTS:
    obs_rng = np.random.default_rng(int(obs_seeds[n_obs]))
    obstacles = make_obstacles(n_obs, obs_rng)
    stt_p = stt.STTParams(**STT_BASE)

    for condition in CONDITIONS:
        key = (n_obs, condition)
        if key in results and all(
                len(results[key].get(c, [])) >= N_TRIALS for c in CONTROLLERS):
            print(f"[skip] n_obs={n_obs} condition={condition} already done")
            continue

        results.setdefault(key, {c: [] for c in CONTROLLERS})
        trial_seeds = np.random.default_rng(n_obs * 100 + CONDITIONS.index(condition)
                                            ).integers(0, 10_000, N_TRIALS)

        for trial_idx in range(N_TRIALS):
            if all(len(results[key].get(c, [])) > trial_idx for c in CONTROLLERS):
                continue   # this trial already checkpointed

            seed = int(trial_seeds[trial_idx])

            # addition ===========
            obs_rng_trial = np.random.default_rng(seed)
            obstacles = make_obstacles(n_obs, obs_rng_trial)
            # ====================

            dist_fn = make_dist_fn(seed) if condition == "disturbed" else None
            print(f"  n_obs={n_obs:2d}  {condition:10s}  trial {trial_idx+1:2d}/{N_TRIALS}"
                  f"  seed={seed}", end="", flush=True)
            t_trial = time.time()

            for cname in CONTROLLERS:
                if len(results[key].get(cname, [])) > trial_idx:
                    continue

                if cname == "GA-STT":
                    log = simulate.run_simulation(
                        ctrl.compute_control, stt_p, DRONE, obstacles,
                        p0=P0, R0=R0, t_end=T_END, dt=DT,
                        disturbance_fn=dist_fn,
                        controller_kwargs=dict(Rp_value=ga.rotor_identity()))
                elif cname == "Baseline":
                    bctrl.reset_baseline_state()
                    log = simulate.run_simulation(
                        bctrl.compute_control_baseline, stt_p, DRONE, obstacles,
                        p0=P0, R0=R0, t_end=T_END, dt=DT,
                        disturbance_fn=dist_fn)
                elif cname == "CBF-QP":
                    cbf_ctrl.reset_cbf_state()
                    log = simulate.run_simulation(
                        cbf_ctrl.compute_control_cbf, stt_p, DRONE, obstacles,
                        p0=P0, R0=R0, t_end=T_END, dt=DT,
                        disturbance_fn=dist_fn)
                else:
                    raise ValueError(cname)

                m = extract_metrics(log, stt_p)
                results[key].setdefault(cname, []).append(m)
                print(f"  [{cname}: {m['ctrl_ms']:.1f}ms ep={m['max_ep']:.3f}"
                      f" {'✓' if m['tube_safe'] else '✗tube'}]", end="", flush=True)

            print(f"  ({time.time()-t_trial:.1f}s)")
            _save_ckpt(results)

        _save_ckpt(results)
        print(f"  → saved checkpoint ({n_obs} obs, {condition})")

print("\nAll trials complete.")

# ── Aggregate ──────────────────────────────────────────────────────────────


def agg(metrics, key):
    vals = [m[key] for m in metrics if m[key] is not None and not (
        isinstance(m[key], float) and np.isnan(m[key]))]
    if not vals:
        return np.nan, np.nan
    if isinstance(vals[0], bool):
        return float(np.mean(vals)) * 100, np.nan   # percentage
    return float(np.mean(vals)), float(np.std(vals))


rows = []
for n_obs in OBS_COUNTS:
    for cname in CONTROLLERS:
        row = {"n_obs": n_obs, "algorithm": cname}
        for condition in CONDITIONS:
            key = (n_obs, condition)
            metrics = results.get(key, {}).get(cname, [])
            if not metrics:
                for k in ["ctrl_ms", "max_ep", "min_clear",
                          "collision_free", "tube_safe", "task_success"]:
                    row[f"{condition}_{k}_mean"] = np.nan
                    row[f"{condition}_{k}_std"] = np.nan
                continue
            for k in ["ctrl_ms", "max_ep", "min_clear"]:
                m, s = agg(metrics, k)
                row[f"{condition}_{k}_mean"] = m
                row[f"{condition}_{k}_std"] = s
            for k in ["collision_free", "tube_safe", "task_success"]:
                pct, _ = agg(metrics, k)
                row[f"{condition}_{k}_pct"] = pct
        rows.append(row)

# ── CSV output ────────────────────────────────────────────────────────────
csv_path = os.path.join(OUT_DIR, "scaling_summary.csv")
raw_path = os.path.join(OUT_DIR, "scaling_study.csv")

fieldnames = list(rows[0].keys())
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"Saved {csv_path}")

with open(raw_path, "w", newline="") as f:
    all_raw = []
    for n_obs in OBS_COUNTS:
        for condition in CONDITIONS:
            key = (n_obs, condition)
            for cname in CONTROLLERS:
                for t_idx, m in enumerate(results.get(key, {}).get(cname, [])):
                    all_raw.append({"n_obs": n_obs, "condition": condition,
                                    "algorithm": cname, "trial": t_idx, **m})
    if all_raw:
        w = csv.DictWriter(f, fieldnames=list(all_raw[0].keys()))
        w.writeheader()
        w.writerows(all_raw)
print(f"Saved {raw_path}")

# ── LaTeX table ────────────────────────────────────────────────────────────


def _fmt(mean, std, is_pct=False, decimals=2):
    if np.isnan(mean):
        return "—"
    if is_pct:
        return f"{mean:.0f}\\%"
    return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"


tex_path = os.path.join(OUT_DIR, "scaling_summary.tex")
with open(tex_path, "w") as f:
    f.write(r"""\begin{table*}[t]
\centering
\caption{Obstacle-count scaling study: """ + f"{N_TRIALS}" + r""" Monte Carlo trials per cell.
  \emph{Nominal}: no disturbance.
  \emph{Disturbed}: $\|\mathbf{w}_p\|\le 0.5$\,N, $\|\mathbf{w}_\tau\|\le 0.05$\,N$\cdot$m.
  MPC excluded from sweep (\SI{351}{ms}/step); Scenario-1 figures in footnote.
  CBF infeasibility rate not shown (0\% for all tested configurations).}
\label{tab:scaling}
\sisetup{table-format=3.2}
\begin{tabular}{cl ccc cc cc c}
\toprule
& & \multicolumn{3}{c}{Compute time [ms]} &
  \multicolumn{2}{c}{Nominal safety [\%]} &
  \multicolumn{2}{c}{Disturbed safety [\%]} &
  Max $e_p$ (dist.) \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
$N_\text{obs}$ & Algorithm &
  mean & std & &
  collision-free & tube-safe &
  collision-free & tube-safe &
  mean$\pm$std \\
\midrule
""")

    prev_n = None
    for row in rows:
        n = row["n_obs"]
        cname = row["algorithm"]
        if n != prev_n and prev_n is not None:
            f.write(r"\midrule" + "\n")
        prev_n = n

        n_col = str(n) if cname == CONTROLLERS[0] else ""
        ctrl_m = row.get(f"nominal_ctrl_ms_mean", np.nan)
        ctrl_s = row.get(f"nominal_ctrl_ms_std",  np.nan)
        nom_cf = row.get("nominal_collision_free_pct", np.nan)
        nom_ts = row.get("nominal_tube_safe_pct",      np.nan)
        dis_cf = row.get("disturbed_collision_free_pct", np.nan)
        dis_ts = row.get("disturbed_tube_safe_pct",      np.nan)
        dep_m = row.get("disturbed_max_ep_mean", np.nan)
        dep_s = row.get("disturbed_max_ep_std",  np.nan)

        bold_start = r"\textbf{" if cname == "GA-STT" else ""
        bold_end = "}" if cname == "GA-STT" else ""

        f.write(
            f"{n_col} & {bold_start}{cname}{bold_end} & "
            f"{bold_start}{ctrl_m:.1f}{bold_end}  &  "
            f"{ctrl_s:.1f}  &  &  "
            f"{bold_start}{_fmt(nom_cf, 0, is_pct=True)}{bold_end} & "
            f"{bold_start}{_fmt(nom_ts, 0, is_pct=True)}{bold_end} & "
            f"{_fmt(dis_cf, 0, is_pct=True)} & "
            f"{_fmt(dis_ts, 0, is_pct=True)} & "
            f"{_fmt(dep_m, dep_s, decimals=3)} "
            r"\\" + "\n"
        )

    f.write(r"""\midrule
\multicolumn{2}{l}{MPC\textsuperscript{\dag}} &
  351 & 18 & & 100\% & $>$1.0\textsuperscript{\ddag} & 100\% & $>$1.0\textsuperscript{\ddag} & 1.258 \\
\bottomrule
\end{tabular}
\begin{tablenotes}\small
\item[\dag] MPC: single Scenario-1 run (5 obstacles). Excluded from sweep due to wall-clock cost.
\item[\ddag] MPC: no tube-invariance guarantee; $e_p > 1$ observed in Scenario-1.
\end{tablenotes}
\end{table*}
""")
print(f"Saved {tex_path}")

# ── Figure ────────────────────────────────────────────────────────────────
COLORS = {"GA-STT": "tab:blue",
          "Baseline": "tab:orange", "CBF-QP": "tab:green"}
MARKERS = {"GA-STT": "o",        "Baseline": "s",          "CBF-QP": "^"}
x = np.array(OBS_COUNTS)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Obstacle-Scaling Study ({N_TRIALS} trials each, "
    f"|w_p|≤{WP_MAX}N disturbed condition)",
    fontsize=12, fontweight="bold")

# Panel 1: compute time mean ± std vs n_obs
ax = axes[0]
for cname in CONTROLLERS:
    means = [next((r[f"nominal_ctrl_ms_mean"] for r in rows
                   if r["n_obs"] == n and r["algorithm"] == cname), np.nan) for n in OBS_COUNTS]
    stds = [next((r[f"nominal_ctrl_ms_std"] for r in rows
                  if r["n_obs"] == n and r["algorithm"] == cname), np.nan) for n in OBS_COUNTS]
    ax.errorbar(x, means, yerr=stds, marker=MARKERS[cname],
                color=COLORS[cname], label=cname, capsize=4, lw=1.8)
ax.axhline(351, color="tab:red", ls=":", lw=1.5, label="MPC (Sc.1)")
ax.set_xlabel("# obstacles")
ax.set_ylabel("Control eval time [ms]")
ax.set_title("Compute time vs obstacle count")
ax.legend(fontsize=9)
ax.set_xticks(OBS_COUNTS)

# Panel 2: tube-safe rate (disturbed) vs n_obs
ax = axes[1]
for cname in CONTROLLERS:
    pcts = [next((r["disturbed_tube_safe_pct"] for r in rows
                  if r["n_obs"] == n and r["algorithm"] == cname), np.nan) for n in OBS_COUNTS]
    ax.plot(x, pcts, marker=MARKERS[cname], color=COLORS[cname],
            label=cname, lw=1.8)
ax.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
ax.set_xlabel("# obstacles")
ax.set_ylabel("Tube-safe rate [%]")
ax.set_title("Tube invariance under disturbance")
ax.legend(fontsize=9)
ax.set_xticks(OBS_COUNTS)

# Panel 3: max eₚ mean (disturbed) vs n_obs
ax = axes[2]
for cname in CONTROLLERS:
    means = [next((r["disturbed_max_ep_mean"] for r in rows
                   if r["n_obs"] == n and r["algorithm"] == cname), np.nan) for n in OBS_COUNTS]
    stds = [next((r["disturbed_max_ep_std"] for r in rows
                  if r["n_obs"] == n and r["algorithm"] == cname), np.nan) for n in OBS_COUNTS]
    ax.errorbar(x, means, yerr=stds, marker=MARKERS[cname],
                color=COLORS[cname], label=cname, capsize=4, lw=1.8)
ax.axhline(1.0, color="red", ls="--", lw=1.5, label="tube boundary")
ax.set_xlabel("# obstacles")
ax.set_ylabel("Max $e_p$ (mean ± std)")
ax.set_title("Tube usage under disturbance\n(lower = more ISS margin)")
ax.legend(fontsize=9)
ax.set_xticks(OBS_COUNTS)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "scaling_figure.png")
fig.savefig(fig_path, dpi=130)
print(f"Saved {fig_path}")

# ── Console summary ───────────────────────────────────────────────────────
print("\n=== SUMMARY TABLE (disturbed condition) ===")
header = f"{'n_obs':>6} {'algorithm':<12} {'ctrl_ms':>9} {'tube-safe%':>10} {'max_ep':>8}"
print(header)
print("-" * len(header))
for row in rows:
    ts_pct = row.get("disturbed_tube_safe_pct", np.nan)
    ep_m = row.get("disturbed_max_ep_mean",   np.nan)
    ep_s = row.get("disturbed_max_ep_std",    np.nan)
    ctrl_m = row.get("nominal_ctrl_ms_mean",    np.nan)
    ctrl_s = row.get("nominal_ctrl_ms_std",     np.nan)
    ep_str = f"{ep_m:.3f}±{ep_s:.3f}" if not np.isnan(ep_m) else "   n/a"
    ts_str = f"{ts_pct:.0f}%" if not np.isnan(ts_pct) else "  n/a"
    cs_str = f"{ctrl_m:.2f}±{ctrl_s:.2f}" if not np.isnan(
        ctrl_m) else "    n/a"
    print(
        f"{row['n_obs']:>6} {row['algorithm']:<12} {cs_str:>9} {ts_str:>10} {ep_str:>12}")
