"""
dynamics.py — GA equations of motion, ENU
convention, RK4 integration with rotor renormalization.

    p_dot = v
    m v_dot = -m g e3 + f R e3 Rrev + w_p
    R_dot = -1/2 R Omega_b
    J(Omega_b_dot) + [Omega_b, J(Omega_b)] = tau + w_tau
"""

import numpy as np
import ga3 as ga

G = 9.81
E3 = np.array([0.0, 0.0, 1.0])


class UAVState:
    __slots__ = ("p", "v", "R", "Omega_b")

    def __init__(self, p, v, R, Omega_b):
        self.p = np.asarray(p, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Omega_b = np.asarray(Omega_b, dtype=float)

    def copy(self):
        return UAVState(self.p.copy(), self.v.copy(), self.R.copy(), self.Omega_b.copy())

    def pack(self):
        return np.concatenate([self.p, self.v, self.R, self.Omega_b])

    @staticmethod
    def unpack(x):
        return UAVState(x[0:3], x[3:6], x[6:14], x[14:17])


def state_derivative(state, m, J_diag, f, tau, wp, wtau):
    p_dot = state.v
    v_dot = -G * E3 + (f / m) * ga.sandwich(state.R, E3) + wp / m
    R_dot = -0.5 * ga.geometric_product(state.R, ga.bivector(state.Omega_b))
    J = J_diag
    Omega_dot = (tau + wtau - np.cross(state.Omega_b, J * state.Omega_b)) / J
    return UAVState(p_dot, v_dot, R_dot, Omega_dot)


def rk4_step(state, dt, m, J_diag, control_fn, disturbance_fn, t):
    """control_fn(t, state) -> (f, tau); disturbance_fn(t) -> (wp, wtau)."""
    def deriv(s, t_local):
        f, tau = control_fn(t_local, s)
        wp, wtau = disturbance_fn(t_local)
        return state_derivative(s, m, J_diag, f, tau, wp, wtau), f, tau

    k1, f0, tau0 = deriv(state, t)
    s2 = UAVState(state.p + 0.5*dt*k1.p, state.v + 0.5*dt*k1.v,
                   state.R + 0.5*dt*k1.R, state.Omega_b + 0.5*dt*k1.Omega_b)
    s2.R = ga.normalize_rotor(s2.R)
    k2, _, _ = deriv(s2, t + 0.5*dt)
    s3 = UAVState(state.p + 0.5*dt*k2.p, state.v + 0.5*dt*k2.v,
                   state.R + 0.5*dt*k2.R, state.Omega_b + 0.5*dt*k2.Omega_b)
    s3.R = ga.normalize_rotor(s3.R)
    k3, _, _ = deriv(s3, t + 0.5*dt)
    s4 = UAVState(state.p + dt*k3.p, state.v + dt*k3.v,
                   state.R + dt*k3.R, state.Omega_b + dt*k3.Omega_b)
    s4.R = ga.normalize_rotor(s4.R)
    k4, _, _ = deriv(s4, t + dt)

    new_p = state.p + (dt/6.0)*(k1.p + 2*k2.p + 2*k3.p + k4.p)
    new_v = state.v + (dt/6.0)*(k1.v + 2*k2.v + 2*k3.v + k4.v)
    new_R = state.R + (dt/6.0)*(k1.R + 2*k2.R + 2*k3.R + k4.R)
    new_R = ga.normalize_rotor(new_R)
    new_Omega = state.Omega_b + (dt/6.0)*(k1.Omega_b + 2*k2.Omega_b + 2*k3.Omega_b + k4.Omega_b)

    new_state = UAVState(new_p, new_v, new_R, new_Omega)
    return new_state, f0, tau0
