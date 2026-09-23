"""
tune_gains.py -- gain search for ga_stt_controller_final.py, for a vehicle
and scenario you specify on the command line. This is the actual
tuning-code deliverable (an earlier "tune_quick.py" in a prior version of
this delivery was a bare, hardcoded debug script with no CLI and no
documentation -- this replaces it).

WHAT IT DOES
------------
Runs a small closed-loop simulation (fast custom Euler integrator, not
MuJoCo -- for speed; always confirm the winner against the real MuJoCo run
in run_cf2_validation.py afterward) for every (kappa1, kappa_v, kR,
kOmega) combination in a grid, and reports max tube-margin usage (max e_p;
lower is better, must stay <1.0 to be "safe") for each. It also prints an
initial *guess* grid center from the closed-form linearization described
below, so you don't have to pick starting ranges blind.

METHODOLOGY (why these ranges, if you don't override them)
------------------------------------------------------------
Near the tube center (e1 small), the position loop's closed-loop dynamics
linearize to a damped spring: p_ddot_tilde + kappa_v*p_dot_tilde +
(4*kappa1/rho^2)*p_tilde ~= 0 (see GA_STT_Corrected.pdf Section 4 for the
exact nonlinear version this approximates). That gives:
    omega_n = sqrt(4*kappa1) / rho          (natural frequency, rad/s)
    zeta    = kappa_v / (2*omega_n)         (damping ratio)
Pick a target omega_n (how fast you want the vehicle to converge to the
tube, in rad/s -- e.g. 3-5x the inverse of how many seconds you're willing
to wait) and a target zeta (0.7-1.0 is a reasonable start: slightly
underdamped to critically damped), then:
    kappa1 = (omega_n * rho)^2 / 4
    kappa_v = 2 * zeta * omega_n
There is no comparably simple formula for kR/kOmega (the Stage-R law's
Jacobian is 1/sin(theta_e/2), not a constant), so this script just grid-
searches around a guess for those instead.

Pass --mu-max (an estimate of your scenario's worst-case tube shrink rate,
max(-rho_dot/rho)) to also apply the rigorous, independently-verified
kappa_v>mu_max and t_c-settling-time bounds from
STAGE1_GAIN_TUNING_ANALYSIS.md, which the plain omega_n/zeta heuristic
above does not know about (it can pick a kappa_v/kappa1 pair that looks
fine near a STATIC tube center but is provably unstable once the tube
starts shrinking, e.g. near an obstacle) -- see that document for the
derivation and the numerical verification against the exact nonlinear
Stage-1 dynamics.

USAGE
-----
    python3 tune_gains.py \\
        --mass 0.027 --inertia 2.3951e-5 2.3951e-5 3.2347e-5 \\
        --f-max 0.60 --tau-max 0.012 \\
        --target-omega-n 4.0 --target-zeta 0.9 \\
        --obstacle 0.6 0.6 0.5 0.15 \\
        --start 0.0 0.0 0.4 --target 1.2 1.2 1.0 --tc 8.0 --rho-max 0.35

Run with --help for the full option list and defaults (which reproduce
the search that picked the gains used in run_cf2_validation.py and
monte_carlo_cf2.py for the real Crazyflie 2).

Once you have a winner, copy the printed `DroneParams(...)` line directly
into run_cf2_validation.py / monte_carlo_cf2.py (or your own script).
"""
import sys, os, argparse, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import numpy as np
import stt
import ga_stt_controller_final as ctrl
import ga3 as ga


def simulate(drone, stt_params, obstacles, theta0, T, dt):
    """Fast closed-loop check: semi-implicit Euler rigid-body integration
    driven by ctrl.compute_control. Returns max tube-margin usage (max
    e_p over the run), or None if the controller diverges (NaN) or
    clearly blows up (e_p > 3, an early-exit heuristic to keep bad
    combinations cheap to screen)."""
    p = stt_params.s0.copy()
    v = np.zeros(3)
    axis = np.array([0.3, 0.5, 0.1]); axis = axis / np.linalg.norm(axis)
    R = ga.exp_bivector(axis, theta0)
    Omega_b = np.zeros(3)
    sigma = stt_params.s0.copy()
    max_ep = 0.0
    n = int(T / dt)
    g = 9.81
    for i in range(n):
        t = i * dt
        state = dict(p=p, v=v, R=R, Omega_b=Omega_b, sigma=sigma)
        f_cmd, tau, diag = ctrl.compute_control(t, state, obstacles, stt_params, drone, ga.rotor_identity())
        if not (np.isfinite(f_cmd) and np.all(np.isfinite(tau))):
            return None
        rho_p = stt.rho_p_value(sigma, t, obstacles, stt_params)
        ep = np.linalg.norm(p - sigma) / rho_p
        max_ep = max(max_ep, ep)
        if max_ep > 3.0:
            return None
        a = (f_cmd / drone.m) * ga.sandwich(R, np.array([0, 0, 1.0])) - g * np.array([0, 0, 1.0])
        v = v + a * dt
        p = p + v * dt
        Rdot = -0.5 * ga.geometric_product(R, ga.bivector(Omega_b))
        R = ga.normalize_rotor(R + Rdot * dt)
        Omega_dot = np.linalg.solve(drone.J, tau - np.cross(Omega_b, drone.J @ Omega_b))
        Omega_b = Omega_b + Omega_dot * dt
        sigma_dot = stt.sigma_value(sigma, t, obstacles, stt_params)
        sigma = sigma + sigma_dot * dt
    return max_ep


def linearized_guess(target_omega_n, target_zeta, rho_max):
    """kappa1, kappa_v from the near-center linearization described in the
    module docstring. Used only to CENTER the grid search, not as a final
    answer -- the real (nonlinear, barrier-transformed) dynamics differ
    away from the tube center, which is exactly why this script then
    grid-searches around this guess rather than trusting it directly."""
    kappa1 = (target_omega_n * rho_max) ** 2 / 4.0
    kappa_v = 2.0 * target_zeta * target_omega_n
    return kappa1, kappa_v


def theory_informed_bounds(rho_min, tc, mu_max, zeta, N=5.0):
    """Rigorous bounds from STAGE1_GAIN_TUNING_ANALYSIS.md, verified there
    against the exact nonlinear Stage-1 ODE (not just this linearization):

      kappa_v > mu_max                                (trace/Routh-Hurwitz;
                                                         mu_max = max(-rho_dot/rho),
                                                         the tube's own worst-case
                                                         shrink rate -- pass 0 if
                                                         you haven't estimated it,
                                                         which just disables this
                                                         check)
      kappa1  > mu_max*kappa_v*rho_min**2/4            (determinant/Routh-Hurwitz)
      kappa1 >= (2*N*rho_min/(zeta*tc))**2             (t_c settling-time bound)

    Returns (kappa1_min_tc, kappa_v_min_trace) -- the caller combines these
    with the omega_n/zeta guess above (take the max of both lower bounds).
    """
    kappa1_min_tc = (2.0 * N * rho_min / (zeta * tc)) ** 2
    kappa_v_min_trace = mu_max  # strict inequality; caller should add margin
    return kappa1_min_tc, kappa_v_min_trace


def geometric_grid(center, factors):
    return sorted(set(round(center * f, 8) for f in factors))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mass", type=float, default=0.027, help="vehicle mass, kg")
    ap.add_argument("--inertia", type=float, nargs=3, default=[2.3951e-5, 2.3951e-5, 3.2347e-5],
                    help="diagonal inertia Jxx Jyy Jzz, kg*m^2")
    ap.add_argument("--f-max", type=float, default=0.60, help="max total thrust, N")
    ap.add_argument("--tau-max", type=float, default=0.012, help="torque saturation norm, N*m")
    ap.add_argument("--target-omega-n", type=float, default=4.0,
                    help="target position-loop natural frequency near the tube center, rad/s")
    ap.add_argument("--target-zeta", type=float, default=0.9,
                    help="target position-loop damping ratio near the tube center")
    ap.add_argument("--obstacle", type=float, nargs=4, action="append", default=None,
                    metavar=("X", "Y", "Z", "RADIUS"),
                    help="static obstacle (repeatable); default is a single obstacle at 0.6 0.6 0.5 r=0.15")
    ap.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.4])
    ap.add_argument("--target", type=float, nargs=3, default=[1.2, 1.2, 1.0])
    ap.add_argument("--tc", type=float, default=8.0)
    ap.add_argument("--rho-max", type=float, default=0.35)
    ap.add_argument("--rho-min", type=float, default=0.08)
    ap.add_argument("--theta0", type=float, default=0.25, help="initial attitude error to test against, rad")
    ap.add_argument("--sim-duration", type=float, default=4.0, help="simulated seconds per trial")
    ap.add_argument("--kR-guess", type=float, default=20.0)
    ap.add_argument("--kOmega-guess", type=float, default=1e-3)
    ap.add_argument("--mu-max", type=float, default=0.0,
                    help="max tube shrink rate max(-rho_dot/rho) expected in your scenario "
                         "(1/s). 0 disables the theory-informed kappa1/kappa_v bounds from "
                         "STAGE1_GAIN_TUNING_ANALYSIS.md; if nonzero, the guess is raised to "
                         "satisfy kappa_v>mu_max and the tc-settling-time bound, not just the "
                         "omega_n/zeta heuristic")
    ap.add_argument("--tc-margin-N", type=float, default=5.0,
                    help="settling-time margin factor for the t_c-based kappa1 lower bound "
                         "(want to settle within tc/N; see STAGE1_GAIN_TUNING_ANALYSIS.md Section 3)")
    ap.add_argument("--grid-factors", type=float, nargs="+", default=[0.5, 1.5],
                    help="multipliers applied to each guess to build the search grid "
                         "(default: 2 points per gain, 16 combos total; pass e.g. "
                         "0.5 1.0 2.0 for a finer 3-point/81-combo search)")
    args = ap.parse_args()

    obstacles = ([stt.Obstacle(o[:3], [0, 0, 0], o[3]) for o in args.obstacle]
                if args.obstacle else
                [stt.Obstacle([0.6, 0.6, 0.5], [0.0, 0.0, 0.0], 0.15)])
    stt_params = stt.STTParams(eta=args.target, s0=args.start, tc=args.tc,
                               k1=0.2, k2=1.0, k3=0.5, rho_max=args.rho_max, rho_min=args.rho_min, u=8.0,
                               rhoR0=1.0, rhoR_inf=0.05, kR=1.0)

    kappa1_guess, kappa_v_guess = linearized_guess(args.target_omega_n, args.target_zeta, args.rho_max)
    print(f"omega_n/zeta guess: kappa1~={kappa1_guess:.4f}, kappa_v~={kappa_v_guess:.4f}")

    if args.mu_max > 0:
        kappa1_min_tc, kappa_v_min_trace = theory_informed_bounds(
            args.rho_min, args.tc, args.mu_max, args.target_zeta, N=args.tc_margin_N)
        print(f"theory-informed bounds (STAGE1_GAIN_TUNING_ANALYSIS.md): "
             f"kappa1 >= {kappa1_min_tc:.4f} (t_c settling), kappa_v > {kappa_v_min_trace:.4f} (trace/mu_max)")
        if kappa1_guess < kappa1_min_tc:
            print(f"  -> raising kappa1 guess from {kappa1_guess:.4f} to {kappa1_min_tc:.4f} "
                 f"(omega_n/zeta guess was too slow for t_c={args.tc})")
            kappa1_guess = kappa1_min_tc
        if kappa_v_guess <= kappa_v_min_trace:
            kappa_v_guess_new = 1.5 * kappa_v_min_trace  # 1.5x margin over the strict trace bound
            print(f"  -> raising kappa_v guess from {kappa_v_guess:.4f} to {kappa_v_guess_new:.4f} "
                 f"(omega_n/zeta guess did not clear the kappa_v>mu_max trace-stability condition)")
            kappa_v_guess = kappa_v_guess_new
        # Re-check the determinant condition at the (possibly-raised) guess; raise kappa1 once
        # more if still short, using rho_min (most conservative, smallest tube reached).
        det_min = args.mu_max * kappa_v_guess * args.rho_min ** 2 / 4.0
        if kappa1_guess < det_min:
            print(f"  -> raising kappa1 guess from {kappa1_guess:.4f} to {det_min:.4f} "
                 f"(determinant/Routh-Hurwitz condition at kappa_v={kappa_v_guess:.4f})")
            kappa1_guess = det_min
        print()

    print(f"Grid search will center on: kappa1~={kappa1_guess:.4f}, kappa_v~={kappa_v_guess:.4f}")

    grid_kappa1 = geometric_grid(kappa1_guess, args.grid_factors)
    grid_kappa_v = geometric_grid(kappa_v_guess, args.grid_factors)
    grid_kR = geometric_grid(args.kR_guess, args.grid_factors)
    grid_kOmega = geometric_grid(args.kOmega_guess, args.grid_factors)

    combos = list(itertools.product(grid_kappa1, grid_kappa_v, grid_kR, grid_kOmega))
    print(f"Testing {len(combos)} combinations "
         f"(kappa1 in {grid_kappa1}, kappa_v in {grid_kappa_v}, kR in {grid_kR}, kOmega in {grid_kOmega})\n")

    best = None
    for kappa1, kappa_v, kR, kOmega in combos:
        drone = ctrl.DroneParams(args.mass, args.inertia, 0.0, args.f_max,
                                 kappa1, kappa_v, kR, kOmega, tau_max=args.tau_max)
        res = simulate(drone, stt_params, obstacles, args.theta0, args.sim_duration, dt=0.002)
        tag = "DIVERGED" if res is None else f"max_ep={res:.4f}"
        print(f"kappa1={kappa1:<8.3f} kappa_v={kappa_v:<8.3f} kR={kR:<8.2f} kOmega={kOmega:<10.2e}  {tag}")
        if res is not None and res < 1.0 and (best is None or res < best[1]):
            best = ((kappa1, kappa_v, kR, kOmega), res)

    print()
    if best is None:
        print("No stable combination found in this grid. Try a lower --target-omega-n "
             "(slower, gentler response) or widen the kR/kOmega guesses.")
        sys.exit(1)

    (k1, kv, kr, ko), max_ep = best
    print(f"BEST: kappa1={k1}, kappa_v={kv}, kR={kr}, kOmega={ko}  (max_ep={max_ep:.4f})")
    print("\nCopy this into your script:")
    print(f"    drone = ctrl.DroneParams(mass={args.mass}, J_diag={list(args.inertia)}, "
         f"f_min=0.0, f_max={args.f_max},")
    print(f"                             kappa1={k1}, kappa_v={kv}, kR={kr}, kOmega={ko}, "
         f"tau_max={args.tau_max})")
    print("\nAlways confirm this against the real MuJoCo run "
         "(run_cf2_validation.py or your own equivalent) before trusting it.")
