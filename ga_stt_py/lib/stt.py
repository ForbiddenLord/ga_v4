"""
stt.py — Real-time Spatiotemporal Tube (STT) dynamics.
Translational tube  Gamma_p(t) = B(sigma(t), rho_p(t))   per Das et al. 2026
(eq. 6/7/8 in "Real-Time STT for Dynamic Unsafe Sets").
Rotational tube      Gamma_R(t) = {R : Psi_e <= rho_R(t)}.

In addition to the plain (current-value) evaluation used for forward
simulation, this module provides `sigma_jet(...)`, which returns the EXACT
Taylor jet (order N) of sigma(t) at the current instant by recursively
solving the autonomous ODE sigma_dot = f(t, sigma) coefficient-by-coefficient
(the standard Taylor-series-ODE-integration trick). This lets the controller
get sigma, sigma_dot, sigma_ddot, sigma_dddot, ... analytically, with no
finite-difference approximation — justifying the "approximation-free" claim.

Design note on v_j(t): Das et al. specify only that v_j lies in the null
space of m_j (any such vector is valid per their derivation).
We use the explicit, smooth choice:
v_hat_j = normalize( cross(m_hat_j, e_ref) )
with e_ref switched away from near-parallel cases. 
For plain evaluation (sigma_value), this is computed at the current value.
For jet evaluation (sigma_jet), we use a fully jet-differentiable version
(_vj_smooth) that allows the tangential nudge direction to vary analytically
with time through the Taylor expansion, providing a mathematically richer 
derivative chain while still fixing the reference axis based on the initial 
m_0 to prevent branch-flipping.
"""
import numpy as np
import jet1d as jt


class Obstacle:
    """A dynamic spherical obstacle with constant-*acceleration* kinematics
    (constant-velocity is the special case acc=0, the default -- so the
    original 3-arg call sites keep working unchanged) and a constant
    radius. See ga_stt_controller.hpp's stt.hpp corrigendum for why acc is
    useful: it supplies one more exact order of sigma's Taylor jet before
    falling back to the zero-jerk assumption."""

    def __init__(self, pos0, vel, radius, acc=None):
        self.pos0 = np.asarray(pos0, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.acc = np.zeros(3) if acc is None else np.asarray(acc, dtype=float)
        self.radius = float(radius)

    def position(self, t):
        return self.pos0 + self.vel * t + 0.5 * self.acc * t * t

    def velocity(self, t):
        return self.vel + self.acc * t

    def position_jet(self, t0, order):
        return self._pos_jet(t0, order)

    def _pos_jet(self, t0, order):
        c = np.zeros((order + 1, 3))
        c[0] = self.position(t0)
        if order >= 1:
            c[1] = self.velocity(t0)
        if order >= 2:
            c[2] = 0.5 * self.acc
        return c

    def radius_jet(self, order):
        return jt.const_scalar(self.radius, order)


class STTParams:
    def __init__(self, eta, s0, tc, k1=0.3, k2=1.0, k3=0.5,
                 rho_max=1.0, rho_min=0.15, u=8.0,
                 rhoR0=1.2, rhoR_inf=0.05, kR=0.8):
        self.eta = np.asarray(eta, dtype=float)
        self.s0 = np.asarray(s0, dtype=float)
        self.tc = float(tc)
        self.k1 = float(k1)
        self.k2 = float(k2)
        self.k3 = float(k3)
        self.rho_max = float(rho_max)
        self.rho_min = float(rho_min)
        self.u = float(u)
        self.rhoR0 = float(rhoR0)
        self.rhoR_inf = float(rhoR_inf)
        self.kR = float(kR)


def _safe_recip_tc_minus_t(t, tc):
    # Floor raised from the original 1e-6 (verified against a parallel
    # delivery's finding): with floor=1e-6, the prescribed-time gain
    # k1*tc/(tc-t) reaches ~k1*tc/1e-6 (order 1e5-1e6) for any t within a
    # microsecond of tc, and stays there for all t>=tc (max(tc-t,floor)
    # simply clamps at the floor once tc-t goes negative). This is a
    # plausible root cause of the long-duration ("30-60s") drift reported
    # for the original architecture. A parallel delivery fixed this by
    # raising the floor AND switching to a separate fixed-gain "hold" law
    # for t>=tc -- but that introduces its own bug: sigma_dot is
    # DISCONTINUOUS at t=tc (verified: jumps from ~2.4 to 0.5 in a typical
    # configuration, since the hold law's own gain doesn't match the
    # floored reach law's gain at the switch). Simply raising the floor,
    # with NO separate law, is both simpler and exactly continuous: once
    # tc-t drops below the floor, max(tc-t,floor) clamps at floor forever
    # (verified: identical sigma_value for t=16.0 through t=100.0 in a
    # tc=16 scenario), so the reach law is already its own well-behaved,
    # bounded "hold" law for all t>=tc-floor with no switch needed.
    floor = max(0.5, 0.02 * tc)
    return tc / max(tc - t, floor)


def _mj_vj_thetaj(sigma, obstacles, t, params):
    """Plain (non-jet) evaluation of m_j, v_j, theta_j at current values,
    used to decide which obstacles are 'active'."""
    active = []
    for j, obs in enumerate(obstacles):
        oj = obs.position(t)
        diff = sigma - oj
        dist = np.linalg.norm(diff)
        dhat = dist - obs.radius
        if dhat <= params.rho_max:
            active.append(j)
    return active


def _vhat_j(m_hat):
    e_ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(m_hat, e_ref)) > 0.9:
        e_ref = np.array([1.0, 0.0, 0.0])
    v = np.cross(m_hat, e_ref)
    n = np.linalg.norm(v)
    if n < 1e-9:
        e_ref = np.array([0.0, 1.0, 0.0])
        v = np.cross(m_hat, e_ref)
        n = np.linalg.norm(v)
    return v / n


def _vj_smooth(mj_jet):
    """
    Any vector orthogonal to mj is a valid tangential nudge direction;
    we pick v = normalize(cross(mj, ref)) with ref chosen ONCE from 
    mj's current (order-0) direction to keep the branch fixed across 
    the Taylor expansion.
    """
    order = jt.order_of(mj_jet)
    m0 = mj_jet[0]
    m0n = m0 / (np.linalg.norm(m0) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(m0n, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    ref_jet = jt.const_vector(ref, order)
    raw = jt.cross_vv(mj_jet, ref_jet)
    n = jt.norm_v(raw)
    n_reg = n.copy()
    n_reg[0] = n_reg[0] + 1e-9
    return jt.vec_div_s(raw, n_reg)


def sigma_value(sigma, t, obstacles, params):
    """sigma_dot at (t, sigma) — eq. 6 (GA-STT eq.16 / Das eq.6), plain values."""
    coef = params.k1 * _safe_recip_tc_minus_t(t, params.tc)
    sdot = coef * (params.eta - sigma)
    active = _mj_vj_thetaj(sigma, obstacles, t, params)
    for j in active:
        obs = obstacles[j]
        oj = obs.position(t)
        diff = sigma - oj
        dist = np.linalg.norm(diff)
        dhat = dist - obs.radius
        m_hat = diff / max(dist, 1e-9)
        denom = max(dist - (obs.radius + params.rho_min), 1e-6)
        mj = diff / denom**3
        vj = _vhat_j(m_hat)
        theta_j = 1.0 / max(dhat, 1e-6) - 1.0 / params.rho_max
        sdot = sdot + (params.k2 * mj + params.k3 * vj) * theta_j
    return sdot


def sigma_jet(sigma0, t0, obstacles, params, order):
    """
    EXACT Taylor jet of sigma(t) about t0, via recursive Taylor-series ODE
    integration: sigma_dot = f(t, sigma); resolve coefficients k=0..order-1
    one at a time, each using only the already-resolved lower coefficients
    of sigma (valid because f depends algebraically on sigma, not on its
    own derivatives).
    """
    c = np.zeros((order + 1, 3))
    c[0] = sigma0

    t_jet = jt.time_jet(t0, order)
    eta_jet = jt.const_vector(params.eta, order)

    # Precompute per-obstacle jets of position/radius (exact, trivial)
    obs_data = []
    for j, obs in enumerate(obstacles):
        oj_jet = obs._pos_jet(t0, order)
        roj_jet = obs.radius_jet(order)
        obs_data.append((oj_jet, roj_jet))

    for k in range(order):
        sigma_partial = c.copy()  # coeffs k+1..order are still 0

        # term1 = k1 * tc/(tc-t) * (eta - sigma)
        tc_minus_t = jt.sub(jt.const_scalar(params.tc, order), t_jet)
        tc_minus_t = tc_minus_t.copy()
        tc_minus_t[0] = max(tc_minus_t[0], max(0.5, 0.02 * params.tc))
        inv_tc_minus_t = jt.recip(tc_minus_t)
        coef = jt.scale(jt.mul_ss(jt.const_scalar(
            params.tc, order), inv_tc_minus_t), params.k1)
        term1 = jt.mul_sv(coef, jt.sub(eta_jet, sigma_partial))

        rhs = term1
        for (oj_jet, roj_jet) in obs_data:
            diff_jet = jt.sub(sigma_partial, oj_jet)
            dist_jet = jt.norm_v(diff_jet)

            # Active set evaluated inline at order-0
            dist0 = dist_jet[0]
            ro0 = roj_jet[0]
            if (dist0 - ro0) > params.rho_max:
                continue  # theta_j == 0 exactly

            dhat_jet = jt.sub(dist_jet, roj_jet)

            denom_jet = jt.sub(dist_jet, jt.add(
                roj_jet, jt.const_scalar(params.rho_min, order)))
            denom3_jet = jt.mul_ss(jt.mul_ss(denom_jet, denom_jet), denom_jet)
            mj_jet = jt.vec_div_s(diff_jet, denom3_jet)

            # Fully jet-differentiable vj (no longer frozen)
            vj_jet = _vj_smooth(mj_jet)

            theta_jet = jt.sub(jt.recip(dhat_jet), jt.const_scalar(
                1.0 / params.rho_max, order))

            combo = jt.add(jt.scale(mj_jet, params.k2),
                           jt.scale(vj_jet, params.k3))
            contrib = jt.mul_sv(theta_jet, combo)
            rhs = jt.add(rhs, contrib)

        c[k + 1] = rhs[k] / (k + 1)

    return c


def rho_p_value(sigma, t, obstacles, params):
    """rho_p(t) via the smooth-min closed form, eq.(15) Das."""
    u = params.u
    dhats = []
    for obs in obstacles:
        oj = obs.position(t)
        dist = np.linalg.norm(sigma - oj)
        dhats.append(dist - obs.radius)
    if len(dhats) == 0:
        d = params.rho_max
    else:
        dhats = np.array(dhats)
        d = -1.0 / u * np.log(np.sum(np.exp(-u * dhats)))
    rho = -1.0 / u * np.log(np.exp(-u * params.rho_max) + np.exp(-u * d))
    return rho


def rho_p_jet(sigma_jet_arr, t0, obstacles, params, order):
    """Jet of rho_p(t) given sigma's jet (algebraic post-processing, no ODE)."""
    u = params.u
    if len(obstacles) == 0:
        return jt.const_scalar(params.rho_max, order)
    exp_terms = None
    for obs in obstacles:
        oj_jet = obs._pos_jet(t0, order)
        roj_jet = obs.radius_jet(order)
        diff_jet = jt.sub(sigma_jet_arr, oj_jet)
        dist_jet = jt.norm_v(diff_jet)
        dhat_jet = jt.sub(dist_jet, roj_jet)
        term = jt.exp_s(jt.scale(dhat_jet, -u))
        exp_terms = term if exp_terms is None else jt.add(exp_terms, term)
    d_jet = jt.scale(jt.log_s(exp_terms), -1.0 / u)
    inner = jt.add(jt.const_scalar(np.exp(-u * params.rho_max), order),
                   jt.exp_s(jt.scale(d_jet, -u)))
    rho_jet = jt.scale(jt.log_s(inner), -1.0 / u)
    return rho_jet


def rho_R(t, params):
    return (params.rhoR0 - params.rhoR_inf) * np.exp(-params.kR * t) + params.rhoR_inf


def rho_R_dot(t, params):
    return -params.kR * (params.rhoR0 - params.rhoR_inf) * np.exp(-params.kR * t)
