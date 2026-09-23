"""
hardware_realism.py -- motor/ESC dynamics, sensor noise, and communication
delay for the MuJoCo validation harnesses in this folder. NONE of this
existed anywhere in this delivery before now (it was explicitly listed as
a known limitation in the top-level DELIVERY_STATUS.md); this module adds
it, reusable across the Crazyflie 2 and Holybro X500 validation scripts.

What's here and what it models:

1. QuadMixer -- converts a commanded (total thrust f, body torque tau)
   into 4 individual motor thrusts (and back), for a generic X-configuration
   quadrotor given rotor positions and spin directions. This is needed
   BEFORE motor lag can be modeled realistically: lagging the aggregate
   (f, tau) directly is not the same as lagging each motor independently
   and re-deriving the net wrench from the (possibly momentarily
   inconsistent) actual per-motor thrusts, which is what a real ESC/motor
   stack actually does.

2. MotorModel -- a first-order lag per motor,
       T_actual_dot = (T_commanded - T_actual) / tau_motor
   clamped to [0, T_max_per_motor], integrated with the same dt as the
   physics step. tau_motor is a literature-typical brushless motor+ESC
   electrical+mechanical time constant (NOT hardware-measured for either
   vehicle modeled here -- see each validation script's own docstring for
   the specific value used and why).

3. SensorNoise -- additive Gaussian noise (with an optional slowly-varying
   gyro bias, a random walk) applied to the STATE FED INTO THE CONTROLLER
   only -- physics always evolves on the true, noise-free state. Models
   gyro noise (rad/s), a slow gyro bias drift, and position/velocity
   noise (representing state-estimator / GPS / optical-flow error, not a
   specific sensor's raw output).

4. CommDelay -- a fixed-length ring buffer delaying an array by N
   simulation steps before it's used. Applied to the CONTROLLER'S OUTPUT
   (f, tau) here, modeling the combination of computation latency and any
   link between wherever (f, tau) is computed and the ESCs -- this is a
   deliberately generic choice (see each validation script for how it's
   used) since this controller is written as a full geometric controller
   producing (f, tau) directly (analogous to running ON a flight
   controller, like the actual controller_ga_stt.cpp firmware module in
   this delivery), not as a high-level setpoint generator talking to a
   separate low-level attitude controller over a slower link.

None of the numeric parameters below (tau_motor, noise standard
deviations, delay length) are hardware-measured for either vehicle. They
are literature-typical order-of-magnitude choices, explicitly flagged as
such everywhere they're set. If you have real system-ID numbers for your
specific vehicle, use those instead.
"""
import numpy as np


class QuadMixer:
    """Generic X-configuration quadrotor mixer.

    rotor_xy: list of (x, y) rotor positions relative to the body origin, m.
    spin: list of 'ccw' or 'cw' per rotor (reaction-torque direction).
    k_m: torque-to-thrust ratio (N*m per N of thrust) -- a property of the
         propeller, NOT hardware-measured here; see the validation script
         that constructs this mixer for the value used and its source.

    Sign convention (internally consistent; not tied to any particular
    firmware's convention since this mixer is only used for physics
    realism here, not for driving real ESCs):
        f    = sum(T_i)
        tau_x =  sum(T_i * y_i)
        tau_y = -sum(T_i * x_i)
        tau_z =  sum(c_i * T_i),   c_i = +k_m if ccw else -k_m
    """
    def __init__(self, rotor_xy, spin, k_m):
        assert len(rotor_xy) == 4 and len(spin) == 4
        self.rotor_xy = np.asarray(rotor_xy, dtype=float)
        self.k_m = float(k_m)
        c = np.array([k_m if s == 'ccw' else -k_m for s in spin])
        x = self.rotor_xy[:, 0]
        y = self.rotor_xy[:, 1]
        # Forward matrix: motor thrusts (4,) -> [f, taux, tauy, tauz]
        self.A = np.array([
            np.ones(4),
            y,
            -x,
            c,
        ])
        self.A_inv = np.linalg.inv(self.A)

    def wrench_to_motors(self, f, tau):
        """Commanded (f, tau=[taux,tauy,tauz]) -> 4 motor thrusts.
        Can return negative or out-of-range values if the commanded
        wrench isn't achievable -- callers should clamp (MotorModel does
        this automatically when driven through it)."""
        b = np.array([f, tau[0], tau[1], tau[2]])
        return self.A_inv @ b

    def motors_to_wrench(self, T):
        """4 actual motor thrusts -> the (f, tau) they actually produce."""
        b = self.A @ np.asarray(T)
        return b[0], b[1:4]


class MotorModel:
    """First-order lag on each of 4 motor thrusts, with saturation.

    tau_motor: time constant, seconds (literature-typical ~0.02-0.06s for
               small-to-mid brushless motor+ESC stacks; NOT measured for
               either vehicle here).
    t_min, t_max: per-motor thrust clamp, N.
    """
    def __init__(self, tau_motor, t_min, t_max, n_motors=4):
        self.tau = float(tau_motor)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.T = np.full(n_motors, max(t_min, 0.0))

    def reset(self, T0=None):
        self.T = np.array(T0) if T0 is not None else np.zeros_like(self.T)

    def step(self, T_commanded, dt):
        T_commanded = np.clip(T_commanded, self.t_min, self.t_max)
        self.T = self.T + (T_commanded - self.T) * (dt / self.tau)
        self.T = np.clip(self.T, self.t_min, self.t_max)
        return self.T.copy()


class SensorNoise:
    """Additive Gaussian noise applied to the state fed into the
    controller. Physics must always be stepped on the TRUE state; only the
    copy handed to compute_control() should go through this.

    gyro_noise_std: rad/s, white noise added to Omega_b each control call.
    gyro_bias_walk_std: rad/s per sqrt(s), a slow random-walk bias added to
                        the gyro reading (models slow bias drift, not just
                        white noise -- real gyros have both).
    pos_noise_std: m, white noise on position (state-estimator/GPS proxy).
    vel_noise_std: m/s, white noise on velocity.
    """
    def __init__(self, gyro_noise_std=0.0, gyro_bias_walk_std=0.0,
                pos_noise_std=0.0, vel_noise_std=0.0, seed=0):
        self.gyro_noise_std = gyro_noise_std
        self.gyro_bias_walk_std = gyro_bias_walk_std
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = np.zeros(3)

    def apply(self, p, v, Omega_b, dt):
        self.gyro_bias = self.gyro_bias + self.rng.normal(0, self.gyro_bias_walk_std * np.sqrt(dt), 3)
        Omega_noisy = Omega_b + self.gyro_bias + self.rng.normal(0, self.gyro_noise_std, 3)
        p_noisy = p + self.rng.normal(0, self.pos_noise_std, 3)
        v_noisy = v + self.rng.normal(0, self.vel_noise_std, 3)
        return p_noisy, v_noisy, Omega_noisy


class CommDelay:
    """Fixed-length delay line for an array-valued signal (e.g. the
    controller's (f, tau) output). n_steps=0 disables it (pass-through).

    The buffer is pre-filled with init_value (NOT zeros): a zero-filled
    buffer would command zero thrust/torque for the first n_steps calls
    while it flushes, an artificial "actuators switched off" transient
    that has nothing to do with communication delay and can by itself
    cause an otherwise-fine controller to diverge purely from the initial
    transient (found by testing: a zero-filled buffer combined with a
    large initial attitude error caused exactly this). Pass the vehicle's
    hover wrench (or your own best first guess) as init_value.
    """
    def __init__(self, n_steps, shape, init_value=None):
        self.n_steps = int(n_steps)
        fill = np.zeros(shape) if init_value is None else np.asarray(init_value, dtype=float)
        self.buf = [fill.copy() for _ in range(max(self.n_steps, 1))]

    def push_and_get(self, x):
        if self.n_steps <= 0:
            return np.asarray(x)
        self.buf.append(np.asarray(x))
        return self.buf.pop(0)
