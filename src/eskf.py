"""
15-State Error-State Kalman Filter -- INS/GNSS/Baro/Mag loose coupling
======================================================================

Built from first principles, term by term. Frame: local NED, flat-earth
position (metres), WITH Earth-rate/transport-rate/Coriolis corrections.

State vector
------------
  Nominal (what we actually estimate, propagated by the IMU):
      p  (3)  position   NED, metres
      v  (3)  velocity   NED, m/s
      q  (4)  attitude   quaternion [w,x,y,z], body -> nav
      ba (3)  accel bias body frame, m/s^2
      bg (3)  gyro  bias body frame, rad/s

  Error state (what the Kalman filter estimates, 15x1):
      dp     (3)  position error
      dv     (3)  velocity error
      dtheta (3)  attitude error -- SMALL ROTATION VECTOR, not a quaternion.
                  3 params, not 4, because rotation has only 3 DOF; a
                  4-element error would make P singular.
      dba    (3)  accel bias error
      dbg    (3)  gyro  bias error

Role of each matrix
-------------------
  Q -- process noise. Driven by the IMU (accel/gyro noise + bias random
       walk). The IMU is the PROCESS MODEL here, not a measurement, so it
       has no R and no H.
  R -- measurement noise. One per aiding sensor: GPS, baro, mag.
  P -- state covariance, 15x15. Grows in predict(), shrinks in update().
  F -- error-state dynamics (continuous), discretised to
       Phi = I + F*dt + (F*dt)^2/2 (second-order truncation of exp(F*dt)).
  H -- maps error state to expected measurement residual. One per sensor.

Sensors
-------
  IMU   -- drives predict(), ~100 Hz
  GPS   -- position + velocity, 6-row update, ~1 Hz
  Baro  -- altitude only, 1-row update, ~10 Hz
  Mag   -- yaw only, 1-row update, ~5 Hz
"""

import numpy as np
from dataclasses import dataclass, field

# ============================================================
#  CONSTANTS
# ============================================================
G_MAG    = 9.80665       # standard gravity, m/s^2
OMEGA_IE = 7.292115e-5   # Earth rotation rate, rad/s
R_0      = 6378137.0     # WGS84 equatorial radius, m
E2       = 0.00669437999014   # WGS84 eccentricity squared

# ── dt bounds ────────────────────────────────────────────────────────
# predict() clamps its timestep into this range. Rationale:
#
#   Phi = I + F*dt + (F*dt)^2/2 is a second-order truncation of
#   exp(F*dt), and the nominal update assumes constant acceleration
#   across the step. Both are fine at 10 ms and poor at 800 ms -- the
#   second-order term widens the usable range but does not remove the
#   need for a bound, since the constant-acceleration assumption in the
#   nominal update degrades independently of how Phi is truncated.
#
#   Worse, the error is asymmetric between the state and P. Position
#   error from a wrong dt grows linearly and quadratically, but Q's
#   position term goes as dt^3 -- so an 80x dt inflates position process
#   noise by ~500,000x. The filter's stated uncertainty explodes far
#   faster than its actual error, P goes enormous, and the next GPS
#   update arrives with gain ~1 and snaps the state to the measurement,
#   discarding minutes of converged bias estimates.
#
#   A dt that is too SMALL fails the other way: Q under-injected, P too
#   small, filter overconfident, gains shrink, steady-state error
#   plateaus above the sensor noise floor.
#
#   dt <= 0 is the case that actually breaks things. Zero gives Phi = I
#   and Q = 0 (a wasted step, survivable). NEGATIVE propagates the filter
#   backwards and puts negative entries on Q's diagonal -- P is no longer
#   positive-definite, and the Kalman gain produces either NaNs or
#   plausible-looking garbage. On a Pi this is not hypothetical:
#   time.time() is wall-clock and NTP can step it backwards mid-run.
#   Prefer time.monotonic() for the caller's clock, which removes that
#   case but not scheduling delays -- so the clamp still earns its place.
DT_MIN = 1e-4    # 10 kHz -- below this the clock is misbehaving
DT_MAX = 0.1     # 10 Hz  -- above this, samples have been lost

# ── Non-finite recovery escalation ───────────────────────────────────
# If the state or P goes NaN/inf, the operation is rolled back to the
# last good values and the run continues -- a field run should not die
# on one bad sample. But a silent rollback that keeps happening is worse
# than a crash: the filter freezes while the vehicle keeps moving, and
# nothing in the state LOOKS wrong. So repeated failures escalate.
NONFINITE_WARN   = 5    # consecutive rollbacks -> print why, keep going
NONFINITE_RESET  = 10   # -> reset velocity to zero, re-inflate P
NONFINITE_FREEZE = 15   # -> stop updating, flag the filter dead

# Covariance inflation applied when the gate escape hatch fires. See
# _passes_gate: forcing one measurement through without widening P leaves
# the filter fusing ~1 measurement in 6.
GATE_RECOVERY_INFLATE = 4.0

# ── Magnetometer disturbance rejection ───────────────────────────────
# The magnetometer measures the TOTAL field: Earth's, plus whatever else
# is nearby. Motor current, a steel table leg, a battery lead carrying
# 20 A -- each adds its own contribution. The heading computed from a
# disturbed sample is wrong, but smoothly and plausibly wrong: the
# innovation lands a few degrees out, well inside the chi2 gate, and gets
# fused. So the gate does not catch this and a separate check is needed.
#
# That check: Earth's field magnitude is roughly constant at a given
# location (~49 uT in the UK). If |B| is well off the expected value,
# something is adding to it and the heading from that sample is suspect.
#
# The expected value is LEARNED from the first readings rather than taken
# from a geomagnetic model, because a hard/soft-iron calibration with a
# scale error makes every reading differ from the model by that factor --
# a learned baseline absorbs it. The risk is learning a disturbed value
# if the vehicle starts next to something ferrous, which is what the
# spread check and the sanity band below guard against.
MAG_LEARN_N     = 20      # samples used to learn the baseline
MAG_LEARN_SPREAD = 0.10   # reject the baseline if they disagree by more
MAG_FIELD_TOL   = 0.25    # deviation from baseline that counts as disturbed
MAG_FIELD_MIN   = 20.0    # uT -- sanity band for the learned baseline
MAG_FIELD_MAX   = 80.0    # (Earth's field is ~25-65 uT worldwide)

# ── Zero-velocity update (ZUPT) ──────────────────────────────────────
# When the vehicle is genuinely stationary, velocity is known to be
# exactly zero. That is free information, and it is the single most
# effective way to pin down accelerometer bias -- which is otherwise only
# weakly observable, since at rest a tilt error and an accel bias produce
# nearly the same signature.
#
# The update itself is trivial. The hard part is knowing you are actually
# stopped: injecting "velocity is zero" while moving is far worse than
# having no ZUPT at all, because it is a confident measurement that is
# flatly wrong, and the filter has no way to tell.
#
# Detection uses the VARIANCE of the IMU over a window, not the mean.
# The mean is useless here: a stationary accelerometer reads 9.81 (not
# zero), and one in steady level flight reads the same. What separates
# them is that real motion is never perfectly steady -- vibration,
# turbulence, control inputs all show up as variance. Gyro magnitude is
# checked too, since a vehicle rotating in place has near-zero accel
# variance but is not stationary in any useful sense.
ZUPT_WINDOW      = 50       # samples (0.5 s at 100 Hz)
# Sized from a measured stationary run rather than guessed. Rolling
# 50-sample statistics over 20260827_160750, comparing a quiet window
# (t=200-400 s) against a window where the unit was physically handled
# (t=424-462 s, IMU peaked at 3.93 rad/s and 22 m/s^2):
#
#                 quiet max    handled max    separation
#   acc_var         0.01309         41.07         3100x
#   gyr_var         0.00018          1.733        9600x
#   gyr_mag         0.00412          1.311         320x
#
# The separation is enormous, so these can sit close above the quiet
# figures without risking false negatives.
#
# NOTE gyr_var was LOOSENED, not tightened. The old 1e-4 was BELOW the
# quiet maximum of 1.8e-4, so genuinely stationary samples failed it --
# only 93% of a known-static window passed the detector. A threshold
# tighter than the sensor's own noise floor rejects the very condition it
# is meant to detect. At 5e-4 the same window passes 100%.
ZUPT_ACC_VAR     = 0.02     # (m/s^2)^2 -- above this, treat as moving
ZUPT_GYR_VAR     = 5e-4     # (rad/s)^2
ZUPT_GYR_MAG     = 0.01     # rad/s (~0.6 deg/s) -- rules out rotating in place

# Consecutive samples the detector must agree on before a ZUPT is allowed.
#
# The variance window is ZUPT_WINDOW long, so at a motion boundary it
# straddles both regimes: half quiet, half moving. Those mixed windows can
# average out under the thresholds and let a ZUPT through exactly when the
# vehicle is starting or stopping. A dwell requirement makes the detector
# wait until the window has fully cleared the transition. Measured on the
# handling window above, dwell=100 cut boundary firings from 375 to 91 with
# no loss of coverage on the quiet window.
ZUPT_DWELL       = 100
                            # NOTE: applied to BIAS-CORRECTED gyro. Using the
                            # raw reading is a trap: a MEMS unit with a 6 deg/s
                            # bias reads 0.11 rad/s at rest, twice this
                            # threshold, so is_static would be False forever --
                            # disabling ZUPT precisely because of the bias that
                            # ZUPT exists to observe.
ZUPT_VEL_STD     = 0.02     # m/s -- how firmly to assert zero velocity

# ── Accel-bias prior ─────────────────────────────────────────────────
# How firmly to assert that the accel bias is still what the six-position
# calibration measured. See update_accel_bias_prior for the degeneracy
# this breaks and the run that exposed it.
#
# 0.05 m/s^2 is ~60x looser than the 0.0008 m/s^2 residual the calibration
# actually left, so real thermal drift over a long run is unimpeded, but
# it is tight enough that g*sin(1 deg) = 0.17 m/s^2 of tilt cannot hide in
# the bias state.
ACCEL_BIAS_PRIOR_STD = 0.05


# ============================================================
#  QUATERNION / ROTATION UTILITIES
#  [w, x, y, z], Hamilton convention, body -> nav
# ============================================================

def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product p (x) q."""
    pw, px, py, pz = p
    qw, qx, qy, qz = q
    return np.array([
        pw*qw - px*qx - py*qy - pz*qz,
        pw*qx + px*qw + py*qz - pz*qy,
        pw*qy - px*qz + py*qw + pz*qx,
        pw*qz + px*qy - py*qx + pz*qw,
    ])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Body-to-nav rotation matrix R (= C_bn)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) -> quaternion. Used for gyro integration in
    predict() and for injecting dtheta in the correction step."""
    angle = np.linalg.norm(rv)
    if angle < 1e-8:
        return quat_normalize(np.array([1.0, *(0.5 * rv)]))
    axis = rv / angle
    return np.array([np.cos(angle/2.0), *(np.sin(angle/2.0) * axis)])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """[roll, pitch, yaw] in radians. Output/diagnostics only -- the
    filter never uses Euler angles internally."""
    w, x, y, z = q
    return np.array([
        np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
        np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)),
        np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)),
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """Cross-product matrix: skew(v) @ x == v x x. The [.]_x operator."""
    return np.array([
        [  0.0, -v[2],  v[1]],
        [ v[2],   0.0, -v[0]],
        [-v[1],  v[0],   0.0],
    ])


def radii_of_curvature(lat: float):
    """Meridian (R_N) and transverse (R_E) radii of curvature."""
    s2 = np.sin(lat)**2
    denom = 1.0 - E2 * s2
    R_N = R_0 * (1.0 - E2) / denom**1.5
    R_E = R_0 / np.sqrt(denom)
    return R_N, R_E


# ============================================================
#  DATA CONTAINERS
# ============================================================

@dataclass
class NomState:
    p:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    v:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    q:  np.ndarray = field(default_factory=quat_identity)
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class IMUMeas:
    dt:    float
    accel: np.ndarray   # raw specific force, body frame, m/s^2
    gyro:  np.ndarray   # raw angular rate, body frame, rad/s


@dataclass
class GPSMeas:
    position: np.ndarray        # NED, m
    velocity: np.ndarray        # NED, m/s
    pos_std_h: float = 3.0      # horizontal 1-sigma, m
    pos_std_v: float = 6.0      # vertical 1-sigma, m (worse than horizontal:
                                # every satellite is above the horizon, so the
                                # geometry constraining height is one-sided)
    vel_std:   float = 0.15     # velocity 1-sigma, m/s
    vel_valid: tuple = (True, True, True)
    # Which velocity components the receiver ACTUALLY measured. NMEA RMC
    # gives speed-over-ground and course but no vertical rate, so callers
    # routinely pass vD = 0.0 -- and fusing that row asserts "vertical
    # velocity is exactly zero, sigma 0.15 m/s" once per fix, forever. That
    # is a fabricated measurement a real climb has to fight against, and it
    # removes roughly half the climb rate per update. Setting the flag False
    # drops the row instead: "no information" rather than "confidently zero".


@dataclass
class BaroMeas:
    altitude: float             # metres above the NED reference
    std:      float = 2.0


@dataclass
class MagMeas:
    yaw: float                  # heading, rad, 0 = north
    std: float = np.deg2rad(5.0)
    field_norm: float = None    # measured |B| in uT. None disables the
                                # disturbance check -- pass it if you have
                                # the raw 3-axis reading available.


# ============================================================
#  CHI-SQUARE GATE THRESHOLDS
# ============================================================
# NIS is chi-square distributed with dof = number of measurement rows,
# so the threshold must depend on how many rows are being fused. Values
# are quantiles of the chi2 distribution -- i.e. a chosen false-alarm
# rate converted into a number, not an arbitrary constant.
#
# 3-sigma (0.9973 quantile): tight. Falsely rejects ~1 in 370 good
# measurements. Fine for high-rate sensors where losing one costs little.
_CHI2_3SIG = {1: 9.00, 2: 11.83, 3: 14.16, 4: 16.25, 6: 20.06}

# Looser (0.99989 quantile): falsely rejects ~1 in 9000. Used for GPS
# because INS drift accumulates between 1 Hz fixes, so innovations are
# legitimately larger than the sensor noise model alone would predict --
# a tight gate would reject good corrections.
_CHI2_LOOSE = {1: 14.96, 2: 18.23, 3: 20.91, 4: 23.31, 5: 25.53, 6: 27.64}


def chi2_gate(dof: int, loose: bool = False) -> float:
    table = _CHI2_LOOSE if loose else _CHI2_3SIG
    return table.get(dof, float(3.5 * dof))


# ============================================================
#  THE FILTER
# ============================================================

class ESKF:
    N = 15

    # Consecutive rejections tolerated before a sensor's next update is
    # forced through regardless of the gate.
    #
    # The failure mode this guards against: if the filter diverges, P grows,
    # every innovation fails the gate, nothing is ever fused, and it can
    # never recover -- the gate locks the filter out permanently. Five
    # genuine outliers back-to-back is far less likely than "the filter is
    # broken and needs pulling back", so after N rejections we let one
    # through to give it a chance to recover.
    #
    # Per-sensor, because a single count is really a TIME threshold wearing
    # a sample-count disguise: five rejections is five seconds of GPS but
    # half a second of 10 Hz baro or ZUPT. A shared value of 5 makes the
    # fast sensors trip the recovery path an order of magnitude sooner than
    # intended, inflating P on half a second of evidence. These are set to
    # roughly five seconds of each sensor's own cadence.
    MAX_CONSEC_GATE = {'gps': 5, 'gps_vel': 5, 'baro': 50,
                       'mag': 25, 'zupt': 50}
    _DEFAULT_CONSEC_GATE = 5

    # error-state index layout
    IP  = slice(0, 3)
    IV  = slice(3, 6)
    ITH = slice(6, 9)
    IBA = slice(9, 12)
    IBG = slice(12, 15)

    def __init__(self, nom: NomState, P0: np.ndarray, ref_lat: float,
                 sig_accel: float, sig_gyro: float,
                 sig_ba_rw: float, sig_bg_rw: float,
                 use_earth_rate: bool = True):
        """
        sig_accel : accel white-noise root-PSD, m/s^2/sqrt(Hz)
        sig_gyro  : gyro  white-noise root-PSD, rad/s/sqrt(Hz)
        sig_ba_rw : accel bias random-walk root-PSD, m/s^3/sqrt(Hz)
        sig_bg_rw : gyro  bias random-walk root-PSD, rad/s^2/sqrt(Hz)
            -> all four come from Allan-variance analysis of a stationary
               IMU log (datasheet values are only a starting point).
        ref_lat   : latitude, rad (for Earth-rate / transport-rate)
        use_earth_rate : False drops Coriolis/Earth-rate/transport-rate,
            giving the flat-frame (Sola-style) filter. At small-drone
            scale the difference is below the MEMS noise floor, so this
            is a legitimate simplification -- set False to compare.
        """
        assert P0.shape == (self.N, self.N)
        self.nom = nom
        self.P   = P0.copy()
        self.ref_lat = ref_lat
        self.sig_accel = sig_accel
        self.sig_gyro  = sig_gyro
        self.sig_ba_rw = sig_ba_rw
        self.sig_bg_rw = sig_bg_rw
        self.use_earth_rate = use_earth_rate

        # diagnostics: NIS history per sensor, for consistency checking
        self.nis_gps     = []   # position rows
        self.nis_gps_vel = []   # velocity rows -- gated separately,
                                # see update_gps for why
        self.nis_baro = []
        self.nis_mag  = []
        self.nis_zupt = []

        # Gating: per-sensor consecutive-rejection counters. Each sensor
        # tracks its own streak -- baro being gated repeatedly should not
        # force a GPS update through, or vice versa.
        self._consec_gated = {'gps': 0, 'gps_vel': 0, 'baro': 0,
                              'mag': 0, 'zupt': 0, 'ba_prior': 0}
        self.gated_count   = 0      # total rejections, all sensors
        self.gate_recoveries = 0    # times the escape hatch inflated P

        # dt clamping diagnostics. A climbing dt_clamped means the caller's
        # loop is not keeping up -- that is a finding about the system, not
        # something to absorb silently, so it is counted and the worst case
        # is kept.
        self.dt_clamped  = 0
        self.dt_worst    = 0.0      # largest out-of-range dt seen

        # Non-finite recovery. nonfinite_events records (step, operation,
        # action) for every rollback so a run can be diagnosed afterwards
        # from the log rather than from a stack trace at the moment of
        # failure. step_count makes each event locatable against the
        # caller's own timestamped logs.
        self.step_count       = 0
        self.nonfinite_events = []
        self._consec_nonfinite = 0
        self.frozen           = False   # True once NONFINITE_FREEZE is hit

        # Yaw initialisation. False until the first heading source arrives.
        # See set_yaw() for why the first heading must bypass the filter.
        self.yaw_initialized = False

        # Timing. self.t is the filter's own clock, advanced by predict(),
        # so it tracks how far the state has actually been propagated.
        # t_last_meas remembers when each sensor last contributed, which is
        # what makes out-of-order detection possible.
        #
        # Why this matters: a serial read returns a buffer that can hold
        # several NMEA sentences. If parse order and timestamp order ever
        # disagree, an older fix gets fused after a newer one -- the filter
        # is corrected backwards in time and the output stays plausible
        # while being wrong.
        #
        # NOTE this catches out-of-ORDER, not merely OLD. A GPS fix that is
        # genuinely 200 ms stale from receiver processing delay arrives in
        # the right order and passes this check. It describes where you
        # WERE, and correcting that properly is latency compensation, which
        # is a separate and larger piece of work.
        self.t = 0.0
        self.t_last_meas  = {'gps': -np.inf, 'baro': -np.inf,
                             'mag': -np.inf, 'zupt': -np.inf,
                             'ba_prior': -np.inf}
        self.stale_rejects = {'gps': 0, 'baro': 0, 'mag': 0, 'zupt': 0,
                              'ba_prior': 0}

        # Magnetometer field-magnitude baseline, learned at startup.
        self.mag_field_expected = None    # None until learned
        self._mag_learn_buf     = []
        self.mag_disturbed      = 0       # readings rejected as disturbed
        self.mag_learn_failed   = False   # startup samples disagreed

        # ZUPT: rolling IMU buffer for stationarity detection.
        self._zupt_acc  = []
        self._zupt_gyr  = []
        self.is_static  = False    # current detector verdict
        self._static_run = 0       # consecutive samples meeting the
                                   # thresholds, for the ZUPT_DWELL check
        self.zupt_count = 0        # zero-velocity updates applied

    # --------------------------------------------------------
    #  Earth-rate / transport-rate
    # --------------------------------------------------------
    def _earth_rate_ned(self) -> np.ndarray:
        if not self.use_earth_rate:
            return np.zeros(3)
        L = self.ref_lat
        return np.array([OMEGA_IE*np.cos(L), 0.0, -OMEGA_IE*np.sin(L)])

    def _transport_rate_ned(self, alt: float) -> np.ndarray:
        if not self.use_earth_rate:
            return np.zeros(3)
        L = self.ref_lat
        R_N, R_E = radii_of_curvature(L)
        vN, vE = self.nom.v[0], self.nom.v[1]
        return np.array([
             vE / (R_E + alt),
            -vN / (R_N + alt),
            -vE * np.tan(L) / (R_E + alt),
        ])

    # --------------------------------------------------------
    #  PREDICT -- IMU-driven propagation
    # --------------------------------------------------------
    def predict(self, imu: IMUMeas, alt: float = 0.0) -> None:
        if self.frozen:
            return
        self.step_count += 1
        snap = self._snapshot()

        # Clamp rather than skip: skipping leaves a gap in the propagation
        # entirely, which is worse than propagating a slightly wrong
        # interval. See DT_MIN/DT_MAX for why an unclamped dt is dangerous.
        dt = float(imu.dt)
        if not (DT_MIN <= dt <= DT_MAX):
            self.dt_clamped += 1
            if abs(dt) > abs(self.dt_worst):
                self.dt_worst = dt
            dt = float(np.clip(dt, DT_MIN, DT_MAX))
        self.t += dt              # the filter's own clock
        nom = self.nom

        R   = quat_to_dcm(nom.q)        # body -> nav
        a_c = imu.accel - nom.ba        # bias-corrected specific force
        w_c = imu.gyro  - nom.bg        # bias-corrected angular rate

        w_ie_n = self._earth_rate_ned()
        rho_n  = self._transport_rate_ned(alt)
        w_in_n = w_ie_n + rho_n

        # ---- nominal state ---------------------------------------------
        # v_dot = R a_c + g - (2 w_ie + rho) x v
        g_nav    = np.array([0.0, 0.0, +G_MAG])   # NED: down is +z
        coriolis = np.cross(2.0*w_ie_n + rho_n, nom.v)
        a_nav    = R @ a_c + g_nav - coriolis

        nom.p = nom.p + nom.v*dt + 0.5*a_nav*dt**2
        nom.v = nom.v + a_nav*dt

        w_in_b = R.T @ w_in_n            # nav-frame rate expressed in body
        w_nb_b = w_c - w_in_b            # body rate relative to NED
        nom.q  = quat_normalize(quat_mul(nom.q, quat_from_rotvec(w_nb_b*dt)))
        # ba, bg unchanged in the nominal state -- pure random walk, Q only

        # ---- error-state dynamics F (continuous) -----------------------
        F = np.zeros((self.N, self.N))
        F[self.IP,  self.IV ] = np.eye(3)                      # dp_dot = dv
        F[self.IV,  self.IV ] = -skew(2.0*w_ie_n + rho_n)      # Coriolis damping
        F[self.IV,  self.ITH] = -R @ skew(a_c)                 # attitude error tilts
                                                               # the specific-force vector
        F[self.IV,  self.IBA] = -R                             # accel bias -> velocity
        F[self.ITH, self.ITH] = -skew(w_in_n)                  # nav-frame rotation
        F[self.ITH, self.IBG] = -np.eye(3)                     # gyro bias -> attitude
        # bias rows stay zero: random walk, no deterministic coupling

        # Second-order discretisation: Phi ~ I + F dt + (F dt)^2 / 2.
        # First-order is adequate at a steady 10 ms, but dt is clamped up to
        # DT_MAX = 100 ms and real loops overrun -- the second-order term is
        # cheap and keeps propagation honest when they do.
        Fdt = F * dt
        Phi = np.eye(self.N) + Fdt + 0.5 * (Fdt @ Fdt)

        # ---- process noise Q -------------------------------------------
        # Dimensionally: variance of a PSD S over dt is S*dt, NOT
        # (sqrt(S)*dt)^2. Getting this wrong under-weights Q by 1/dt
        # (100x at 100 Hz), which makes the filter overconfident and
        # produces a steady-state error plateau well above the sensor
        # noise floor.
        S_a = self.sig_accel**2
        S_g = self.sig_gyro**2
        Q = np.zeros((self.N, self.N))
        # position/velocity are coupled: both are integrals of the SAME
        # accel-noise sample over this step, so the cross-terms are real.
        Q[self.IP, self.IP] = np.eye(3) * S_a * dt**3 / 3.0
        Q[self.IP, self.IV] = np.eye(3) * S_a * dt**2 / 2.0
        Q[self.IV, self.IP] = np.eye(3) * S_a * dt**2 / 2.0
        Q[self.IV, self.IV] = np.eye(3) * S_a * dt
        Q[self.ITH, self.ITH] = np.eye(3) * S_g * dt
        Q[self.IBA, self.IBA] = np.eye(3) * self.sig_ba_rw**2 * dt
        Q[self.IBG, self.IBG] = np.eye(3) * self.sig_bg_rw**2 * dt

        self.P = Phi @ self.P @ Phi.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        if not self._is_finite():
            self._handle_nonfinite(snap, 'predict')
        else:
            self._consec_nonfinite = 0
            # Bias-corrected, not raw -- see ZUPT_GYR_MAG. a_c/w_c are the
            # same corrected values the mechanisation just used.
            self._zupt_track(a_c, w_c)

    # --------------------------------------------------------
    #  SHARED CORRECTION -- identical for every sensor
    # --------------------------------------------------------
    def _correct(self, H: np.ndarray, R_meas: np.ndarray,
                 innov: np.ndarray, operation: str = 'update') -> bool:
        if self.frozen:
            return False
        snap = self._snapshot()

        S = H @ self.P @ H.T + R_meas
        try:
            K = np.linalg.solve(S.T, H @ self.P).T   # = P H' inv(S), stably
        except np.linalg.LinAlgError:
            # Singular S. Rare, but it happens if P has degenerated -- and
            # it is exactly the case where blindly continuing produces
            # garbage, so treat it as a non-finite event and escalate.
            self._handle_nonfinite(snap, operation)
            return False

        dx = K @ innov                            # the error state --
                                                  # nonzero only right here

        # inject into the nominal state
        nom = self.nom
        nom.p  = nom.p  + dx[self.IP]
        nom.v  = nom.v  + dx[self.IV]
        nom.q  = quat_normalize(quat_mul(nom.q, quat_from_rotvec(dx[self.ITH])))
        nom.ba = nom.ba + dx[self.IBA]
        nom.bg = nom.bg + dx[self.IBG]
        # error state resets to zero implicitly: dx is local, and the next
        # innovation is measured against the freshly-corrected nominal state

        # Joseph form -- stays positive-definite under numerical error
        IKH = np.eye(self.N) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_meas @ K.T

        # ESKF covariance reset. Injecting dtheta rotates the frame the
        # attitude error is expressed in, so the error covariance must be
        # rotated to match: P <- G P G', with G = I - [dtheta/2]_x on the
        # attitude block. Skipping this leaves P describing the error in the
        # OLD frame -- a small inconsistency per update that accumulates.
        G = np.eye(self.N)
        G[self.ITH, self.ITH] = np.eye(3) - skew(0.5 * dx[self.ITH])
        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

        if not self._is_finite():
            self._handle_nonfinite(snap, operation)
            return False
        self._consec_nonfinite = 0
        return True

    def _snapshot(self):
        """Cheap copy of everything that can go non-finite."""
        return (self.nom.p.copy(), self.nom.v.copy(), self.nom.q.copy(),
                self.nom.ba.copy(), self.nom.bg.copy(), self.P.copy())

    def _restore(self, snap) -> None:
        (self.nom.p, self.nom.v, self.nom.q,
         self.nom.ba, self.nom.bg, self.P) = snap

    def _is_finite(self) -> bool:
        return (np.all(np.isfinite(self.nom.p)) and
                np.all(np.isfinite(self.nom.v)) and
                np.all(np.isfinite(self.nom.q)) and
                np.all(np.isfinite(self.nom.ba)) and
                np.all(np.isfinite(self.nom.bg)) and
                np.all(np.isfinite(self.P)))

    def _handle_nonfinite(self, snap, operation: str) -> None:
        """Roll back a failed operation and escalate if it keeps happening.

        Stage 1 (NONFINITE_WARN):   print the reason, keep going.
        Stage 2 (NONFINITE_RESET):  velocity to zero, P re-inflated. The
            state is clearly not recoverable by rollback alone at this
            point, so we discard velocity (the state that runs away
            fastest) and widen P so incoming measurements are trusted
            enough to pull the filter back.
        Stage 3 (NONFINITE_FREEZE): stop updating entirely and set
            self.frozen. Continuing past this would emit numbers that look
            like estimates but are not -- worse than emitting nothing.
        """
        self._restore(snap)
        self._consec_nonfinite += 1
        n = self._consec_nonfinite
        action = 'rollback'

        if n >= NONFINITE_FREEZE and not self.frozen:
            self.frozen = True
            action = 'freeze'
            print(f"\n*** ESKF FROZEN at step {self.step_count} ***\n"
                  f"  {n} consecutive non-finite results, most recently in "
                  f"'{operation}'.\n"
                  f"  Reset at stage 2 did not recover it. The filter has "
                  f"stopped updating;\n"
                  f"  every value it reports from here is stale. Check the "
                  f"IMU feed and the\n"
                  f"  nonfinite_events log for where this started.")
        elif n == NONFINITE_RESET:
            action = 'reset'
            self.nom.v = np.zeros(3)
            self.P = make_P0()
            print(f"\nESKF: {n} consecutive non-finite results in "
                  f"'{operation}' at step {self.step_count}.\n"
                  f"  Rollback alone is not recovering it -- resetting "
                  f"velocity to zero and\n"
                  f"  re-inflating P so measurements can pull the filter "
                  f"back. Position,\n"
                  f"  attitude and bias estimates are kept (no better guess "
                  f"available).")
        elif n == NONFINITE_WARN:
            action = 'warn'
            print(f"\nESKF: {n} consecutive non-finite results in "
                  f"'{operation}' at step {self.step_count}.\n"
                  f"  Each was rolled back, so the state has not advanced "
                  f"for {n} steps --\n"
                  f"  if the vehicle is moving, the estimate is now falling "
                  f"behind reality.\n"
                  f"  Likely causes: bad IMU samples (NaN from a dropped "
                  f"I2C read), a\n"
                  f"  singular innovation covariance, or P having lost "
                  f"positive-definiteness.")

        self.nonfinite_events.append((self.step_count, operation, action))

    def _check_timing(self, t_meas: float, sensor: str) -> bool:
        """Reject a measurement older than one already fused from this sensor.

        Per-sensor rather than global, because sensors legitimately arrive
        interleaved: a 10 Hz baro sample timestamped before the 1 Hz GPS fix
        that was just processed is perfectly normal, not an error.
        """
        if not np.isfinite(t_meas):
            self.stale_rejects[sensor] += 1
            return False
        if t_meas <= self.t_last_meas[sensor]:
            self.stale_rejects[sensor] += 1
            return False
        self.t_last_meas[sensor] = t_meas
        return True

    def _nis(self, H, R_meas, innov) -> float:
        """Normalised innovation squared -- consistency diagnostic.
        Should average to the number of measurement rows. Much lower means
        R is too conservative; much higher means the filter is overconfident."""
        S = H @ self.P @ H.T + R_meas
        return float(innov @ np.linalg.solve(S, innov))

    def _inflate_observed(self, H: np.ndarray) -> None:
        """Inflate P only in the subspace the failing sensor observes.

        Multiplying the whole matrix (P *= f) is the obvious implementation
        and the wrong one. Measured: a persistently-failing BAROMETER -- a
        one-row, vertical-only sensor -- inflated the GYRO BIAS 1-sigma from
        0.05 to 0.40 deg/s, 8x, across three recoveries. Nothing about a
        struggling baro is evidence about gyro bias, and on a unit whose
        gyro bias converges slowly and whose divergence mode IS gyro bias,
        discarding that convergence is an expensive way to fix an unrelated
        sensor.

        Targeting the observed subspace is also the mathematically direct
        fix: the gate tests nis = innov' S^-1 innov with S = H P H' + R, so
        the only part of P that can widen the gate is the part H sees.
        Inflating anything else is pure collateral damage.

        Implemented as P <- D P D with D = diag(sqrt(f) on observed states,
        1 elsewhere). That scales observed variances by f, leaves unobserved
        variances untouched, and scales the cross-covariances by sqrt(f) --
        which keeps correlation coefficients unchanged rather than silently
        rescaling the relationships between states.
        """
        if H is None:
            # No H supplied -- fall back to the old whole-matrix behaviour
            # rather than silently doing nothing.
            self.P *= GATE_RECOVERY_INFLATE
        else:
            observed = np.any(np.abs(H) > 0.0, axis=0)
            d = np.where(observed, np.sqrt(GATE_RECOVERY_INFLATE), 1.0)
            self.P = self.P * np.outer(d, d)
        self.P = 0.5 * (self.P + self.P.T)

    def _passes_gate(self, nis: float, threshold: float, sensor: str,
                     H: np.ndarray = None) -> bool:
        """Innovation gate. Returns True if the measurement should be fused.

        NIS is measured in sigmas, not metres, so the gate is automatically
        scale-aware: a 20 m innovation is implausible when P is small and
        well-converged, but entirely reasonable after a long GPS outage when
        the INS has drifted. The threshold stays fixed in NIS space so it can
        adapt in physical units.

        H is used only if the recovery path fires -- see _inflate_observed.
        """
        if nis <= threshold:
            self._consec_gated[sensor] = 0
            return True

        limit = self.MAX_CONSEC_GATE.get(sensor, self._DEFAULT_CONSEC_GATE)
        if self._consec_gated[sensor] >= limit:
            # Escape hatch -- see MAX_CONSEC_GATE. Persistent rejection is
            # more likely to mean the filter has diverged than that we are
            # seeing genuine outliers back to back, so force this one through.
            #
            # Forcing alone is not enough. Resetting the counter and doing
            # nothing else gives a 1-in-6 duty cycle: five rejections, one
            # forced, repeat -- so a diverged filter fuses 17% of its
            # measurements and crawls back, if it gets back at all. The
            # reason the gate keeps failing is that P is too tight for the
            # error the state actually has, so P is what has to change.
            # Inflating it makes the NEXT innovations pass on their own
            # merits and restores a full-rate correction stream.
            self._inflate_observed(H)
            self.gate_recoveries += 1
            self._consec_gated[sensor] = 0
            return True

        self._consec_gated[sensor] += 1
        self.gated_count += 1
        return False

    # --------------------------------------------------------
    #  HARD YAW SET -- bypasses the filter, for initialisation
    # --------------------------------------------------------
    def set_yaw(self, yaw: float, yaw_std: float = np.deg2rad(10.0)) -> None:
        """Set heading directly, without going through a measurement update.

        Why this exists: at startup yaw is unknown, and the nominal
        quaternion is typically built with yaw = 0 because nothing has
        measured it yet. If the vehicle is actually pointing east, the
        first heading measurement carries a 90 degree innovation.

        The error-state update cannot handle that. Every H and F term we
        use is a LINEARISATION valid for small dtheta -- in particular
        F[dv, dtheta] = -R [a_c]_x, which approximates how a small
        rotation error tilts the specific-force vector. At 90 or 180
        degrees that approximation does not describe reality, so feeding
        such a measurement through update_mag() asks the filter to
        converge through a regime where its own model is wrong. The
        observed failure is attitude diverging to ~178 degrees and never
        recovering, with every sensor's innovations then failing the chi2
        gate against that wrong attitude.

        So: the FIRST heading is set here, hard. Everything after it is a
        genuinely small correction and goes through update_mag() normally.
        Same pattern as position, where the first GPS fix re-anchors the
        frame rather than being filtered in against a dead-reckoned prior.

        Applied as a rotation about the NAV-frame down axis, pre-multiplied
        onto the current attitude (q_new = dq_nav (x) q_old, because the
        rotation is expressed in the nav frame, not the body frame). Roll
        and pitch are preserved structurally rather than by decomposing to
        Euler angles and rebuilding -- that round-trip is both wasteful and
        ill-conditioned near vertical, where roll and yaw stop being
        separately well defined.
        """
        current_yaw = quat_to_euler(self.nom.q)[2]
        dyaw = (yaw - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
        dq = quat_from_rotvec(np.array([0.0, 0.0, dyaw]))
        self.nom.q = quat_normalize(quat_mul(dq, self.nom.q))

        # Only the yaw entry of the attitude covariance is reset. Roll and
        # pitch were already well observed via gravity and are untouched.
        self.P[8, 8] = yaw_std ** 2
        self.yaw_initialized = True

    def check_P_health(self) -> dict:
        """Numerical health of the covariance matrix.

        Two failure modes worth watching, neither of which shows up in the
        state estimate until well after the damage is done:

        Condition number -- the ratio of largest to smallest eigenvalue.
        P mixes states with wildly different natural scales (position in
        metres, gyro bias in rad/s, ~1e-5), so a large condition number is
        NORMAL here and not itself alarming. What matters is the trend: a
        condition number climbing steadily means P is collapsing toward
        singular in some direction, and the Kalman gain computed from it
        is losing precision.

        Negative eigenvalues -- P is no longer positive-definite, which is
        physically meaningless (a negative variance) and mathematically
        fatal. Joseph form plus symmetrisation makes this rare, but it can
        still happen through accumulated float error over a long run.
        """
        try:
            eig = np.linalg.eigvalsh(self.P)
        except np.linalg.LinAlgError:
            return dict(ok=False, reason='eigendecomposition failed',
                        cond=float('inf'), min_eig=float('nan'))

        min_eig = float(eig.min())
        max_eig = float(eig.max())
        cond = max_eig / min_eig if min_eig > 0 else float('inf')
        ok = min_eig > 0 and np.all(np.isfinite(eig))
        return dict(ok=ok, cond=cond, min_eig=min_eig, max_eig=max_eig,
                    reason='' if ok else 'P is not positive-definite')

    def repair_P(self, floor: float = 1e-12) -> bool:
        """Force P back to positive-definite by clipping its eigenvalues.

        A last resort, not routine maintenance. If this is needed, the
        interesting question is what drove P singular in the first place --
        usually a measurement fused with an R that was effectively zero, or
        a state that has become completely unobservable. Returns True if a
        repair was actually applied.
        """
        eig, vec = np.linalg.eigh(self.P)
        if np.all(eig > 0):
            return False
        self.P = vec @ np.diag(np.clip(eig, floor, None)) @ vec.T
        self.P = 0.5 * (self.P + self.P.T)
        return True

    # --------------------------------------------------------
    #  ZUPT -- zero-velocity update
    # --------------------------------------------------------
    def _zupt_track(self, accel: np.ndarray, gyro: np.ndarray) -> None:
        """Maintain the rolling IMU window and the stationarity verdict.

        Called from predict() with BIAS-CORRECTED accel and gyro. Passing
        raw values here silently disables ZUPT on any unit whose gyro bias
        exceeds ZUPT_GYR_MAG -- which is most consumer MEMS parts, and is
        exactly the case where ZUPT would help most.

        Called on every good sample, so the detector always reflects the
        last ZUPT_WINDOW samples regardless of when the caller chooses to
        apply an update.
        """
        if not (np.all(np.isfinite(accel)) and np.all(np.isfinite(gyro))):
            # Do not leave a stale verdict standing. If samples stop
            # arriving -- a NaN burst, a rolled-back predict -- the window
            # no longer describes the present, and is_static would keep
            # reporting True from before the gap, letting ZUPT fire on
            # evidence that is now seconds old.
            # The dwell counter must reset too, not just the verdict. Left
            # standing it would carry its pre-gap value into the next good
            # sample, so is_static could go True again immediately and the
            # dwell requirement would be silently bypassed across exactly
            # the discontinuity it exists to cover.
            self.is_static = False
            self._static_run = 0
            return
        self._zupt_acc.append(np.asarray(accel, dtype=float))
        self._zupt_gyr.append(np.asarray(gyro, dtype=float))
        if len(self._zupt_acc) > ZUPT_WINDOW:
            self._zupt_acc.pop(0)
            self._zupt_gyr.pop(0)

        if len(self._zupt_acc) < ZUPT_WINDOW:
            self.is_static = False       # not enough history to claim it
            self._static_run = 0
            return

        acc = np.array(self._zupt_acc)
        gyr = np.array(self._zupt_gyr)
        # Variance, summed over axes. Deliberately NOT the mean: a
        # stationary accelerometer reads ~9.81, and so does one in steady
        # level flight. Variance is what separates them, because real
        # motion is never perfectly steady.
        acc_var = float(np.sum(np.var(acc, axis=0)))
        gyr_var = float(np.sum(np.var(gyr, axis=0)))
        # Mean gyro magnitude rules out rotating in place, which has low
        # accel variance but is not stationary in any useful sense.
        gyr_mag = float(np.linalg.norm(np.mean(gyr, axis=0)))

        instant = (acc_var < ZUPT_ACC_VAR and
                   gyr_var < ZUPT_GYR_VAR and
                   gyr_mag < ZUPT_GYR_MAG)
        # Dwell: the thresholds alone are satisfied by a window that
        # straddles a motion boundary, since half of it is still quiet.
        # Requiring agreement across ZUPT_DWELL consecutive samples makes
        # the detector wait for the window to clear the transition before
        # claiming the vehicle has stopped. See ZUPT_DWELL.
        self._static_run = self._static_run + 1 if instant else 0
        self.is_static = self._static_run >= ZUPT_DWELL

    def update_accel_bias_prior(self, t: float,
                                ba_std: float = ACCEL_BIAS_PRIOR_STD) -> bool:
        """Pull the accel-bias estimate back toward the calibrated value.

        Exists to break a degeneracy that ZUPT cannot. At rest, a tilt error
        and a horizontal accel bias produce the SAME signature: gravity leaks
        into the horizontal axes as g*sin(tilt), and a bias adds a constant
        offset. Any split satisfying

            g*sin(attitude error) + accel bias = g*sin(true tilt)

        fits the measurements equally well, so the filter is free to wander
        along that null space -- and does. ZUPT makes it worse rather than
        better: it asserts v = 0 firmly, which constrains the SUM while
        saying nothing about the split, and it keeps velocity at ~0.01 m/s
        the whole time so nothing looks wrong.

        Measured on 20260901_163533. Over 8 stationary minutes:

            attitude drifted 3.49 deg from the seed
            |ba| grew to      0.613 m/s^2
            g*sin(3.49 deg) = 0.597 m/s^2      <- matches to 3%

        i.e. the "bias" was entirely absorbed tilt. Process noise cannot
        explain it: sig_ba_rw*sqrt(480 s) allows 0.009 m/s^2, seventy times
        less. It was actively estimated, from measurements that could not
        distinguish it from attitude.

        Invisible until the vehicle moves. Then ZUPT correctly stops firing,
        0.613 m/s^2 integrates into velocity at 0.61 m/s per second, the GPS
        velocity gate is exceeded within a second, and the filter locks
        itself out. On that walk it reached 127 m/s and 1.4 km.

        The fix is to supply the information the geometry lacks. A six-
        position calibration measures the residual bias directly -- on this
        unit it left 0.0008 m/s^2 -- so "ba is near zero" is not an
        assumption, it is a measurement, and asserting it weakly costs
        nothing while removing the null space.

        Deliberately weak (ACCEL_BIAS_PRIOR_STD, ~0.05 m/s^2): far looser
        than the calibration residual, so genuine thermal drift over a long
        run is still free to be estimated, but tight enough that a whole
        degree of tilt cannot hide here. Apply only while stationary -- in
        motion the bias becomes genuinely observable and the prior should
        not fight real evidence.
        """
        if not self._check_timing(t, 'ba_prior'):
            return False
        H = np.zeros((3, self.N))
        H[0:3, self.IBA] = np.eye(3)
        R_meas = np.eye(3) * ba_std ** 2
        innov = -self.nom.ba          # the "measurement" is ba = 0
        # No gate. A gate here would reject exactly the runaway this exists
        # to catch: the further ba has wandered, the larger the innovation,
        # and a gate would call that an outlier and refuse to correct it.
        return self._correct(H, R_meas, innov, 'ba_prior')

    def update_zupt(self, t: float, vel_std: float = ZUPT_VEL_STD,
                    force: bool = False) -> bool:
        """Apply a zero-velocity update if the detector says we are stopped.

        Returns False without doing anything when moving. Pass force=True
        to override the detector -- only when the caller genuinely knows
        better (motors off, clamped to a bench), since a wrongly forced
        ZUPT is a confident measurement that is flatly wrong.
        """
        if not (force or self.is_static):
            return False
        if not self._check_timing(t, 'zupt'):
            return False

        H = np.zeros((3, self.N))
        H[0:3, self.IV] = np.eye(3)
        R_meas = np.eye(3) * vel_std ** 2
        # The "measurement" is that velocity is zero, so the innovation --
        # measured minus predicted -- is simply -v.
        innov = -self.nom.v

        nis = self._nis(H, R_meas, innov)
        self.nis_zupt.append(nis)
        if not self._passes_gate(nis, chi2_gate(3), 'zupt', H):
            return False
        ok = self._correct(H, R_meas, innov, 'zupt')
        self.zupt_count += int(ok)
        return ok

    # --------------------------------------------------------
    #  GPS UPDATE -- 6 rows (position + velocity)
    # --------------------------------------------------------
    def update_gps(self, gps: GPSMeas, t: float) -> bool:
        """t is REQUIRED, not optional. An optional timestamp that a caller
        forgets to pass makes this check a silent no-op: stale_rejects reads
        zero, which is indistinguishable from 'no stale measurements
        occurred'. Requiring it turns forgetting into an immediate TypeError
        instead of an invisible gap in the diagnostics."""
        if not self._check_timing(t, 'gps'):
            return False

        # Position rows always; velocity rows only where the receiver
        # actually measured something (see GPSMeas.vel_valid).
        # Position and velocity are fused as SEPARATE updates with their own
        # gates, not as one six-row measurement.
        #
        # Measured on 20260901_163533 (8 min stationary, then a 22 min walk).
        # Standing still, the accel bias estimate wandered to 0.613 m/s^2 --
        # 766x the 0.0008 m/s^2 the six-position calibration had just left,
        # so it is not bias at all but a TILT error being absorbed. At rest
        # the two are indistinguishable and ZUPT hides the whole thing by
        # pinning velocity at 0.01 m/s. Nothing looks wrong.
        #
        # Then the walk starts, ZUPT correctly stops firing, and 0.613 m/s^2
        # integrates straight into velocity at 0.61 m/s per second. Against
        # vel_std = 0.5 the velocity rows blow the gate in under a second --
        # and with one joint gate that rejected the POSITION rows too, which
        # were perfectly good (raw GPS median speed 0.95 m/s, HDOP 1.03).
        # Nothing was left to correct the state, so it ran to 1.4 km with
        # 237 gate recoveries and three quarters of the fixes discarded.
        #
        # Splitting them costs nothing mathematically: R is already diagonal,
        # and sequential processing of uncorrelated measurements is exactly
        # equivalent to one joint update. What it buys is that a diverged
        # velocity can no longer throw away good position.
        ok_any = False

        # ---- position rows ------------------------------------------
        H_p = np.zeros((3, self.N))
        H_p[0:3, self.IP] = np.eye(3)    # observes dp directly; zeros
        # elsewhere because GPS does NOT directly observe attitude or bias --
        # those are corrected indirectly, through correlations F built in P.
        R_p = np.diag([gps.pos_std_h**2, gps.pos_std_h**2, gps.pos_std_v**2])
        innov_p = gps.position - self.nom.p

        nis_p = self._nis(H_p, R_p, innov_p)
        self.nis_gps.append(nis_p)
        # Loose gate: INS drift between 1 Hz fixes makes innovations
        # legitimately large, so a tight threshold would reject good data.
        if self._passes_gate(nis_p, chi2_gate(3, loose=True), 'gps', H_p):
            ok_any |= self._correct(H_p, R_p, innov_p, 'gps')

        # ---- velocity rows ------------------------------------------
        # Only the components the receiver actually measured. NMEA RMC has no
        # vertical rate, so callers pass vD = 0.0 -- see GPSMeas.vel_valid for
        # why fusing that row is a fabricated measurement.
        keep = np.asarray(gps.vel_valid, dtype=bool)
        n_v = int(keep.sum())
        if n_v:
            H_v = np.zeros((3, self.N))
            H_v[0:3, self.IV] = np.eye(3)
            H_v = H_v[keep]
            R_v = (np.eye(3) * gps.vel_std**2)[np.ix_(keep, keep)]
            innov_v = (gps.velocity - self.nom.v)[keep]

            nis_v = self._nis(H_v, R_v, innov_v)
            self.nis_gps_vel.append(nis_v)
            if self._passes_gate(nis_v, chi2_gate(n_v, loose=True),
                                 'gps_vel', H_v):
                ok_any |= self._correct(H_v, R_v, innov_v, 'gps_vel')

        return ok_any

    # --------------------------------------------------------
    #  BARO UPDATE -- 1 row (altitude)
    # --------------------------------------------------------
    def update_baro(self, baro: BaroMeas, t: float) -> bool:
        if not self._check_timing(t, 'baro'):
            return False

        H = np.zeros((1, self.N))
        H[0, 2] = -1.0                   # altitude = -p_D (down is +z)

        R_meas = np.array([[baro.std**2]])
        innov  = np.array([baro.altitude - (-self.nom.p[2])])

        nis = self._nis(H, R_meas, innov)
        self.nis_baro.append(nis)

        # Tight gate: at 10 Hz, little INS drift accumulates between
        # updates, so genuine innovations stay small.
        if not self._passes_gate(nis, chi2_gate(1), 'baro', H):
            return False
        return self._correct(H, R_meas, innov, 'baro')

    # --------------------------------------------------------
    #  MAG UPDATE -- 1 row (yaw)
    # --------------------------------------------------------
    def update_mag(self, mag: MagMeas, t: float) -> bool:
        if not self._check_timing(t, 'mag'):
            return False

        # Field-magnitude disturbance check, before anything else. See the
        # MAG_* constants for why the chi2 gate cannot catch this case.
        if mag.field_norm is not None and np.isfinite(mag.field_norm):
            if self.mag_field_expected is None and not self.mag_learn_failed:
                self._mag_learn_buf.append(float(mag.field_norm))
                if len(self._mag_learn_buf) >= MAG_LEARN_N:
                    buf = np.array(self._mag_learn_buf)
                    med = float(np.median(buf))
                    spread = float(np.max(np.abs(buf - med)) / max(med, 1e-9))
                    if (spread <= MAG_LEARN_SPREAD and
                            MAG_FIELD_MIN <= med <= MAG_FIELD_MAX):
                        self.mag_field_expected = med
                    else:
                        # Either the startup samples disagreed with each
                        # other, or the magnitude is outside any plausible
                        # Earth field. Both suggest the baseline would be
                        # learned from disturbed data, so the check is
                        # disabled rather than anchored to a bad value --
                        # a wrong baseline rejects GOOD readings forever,
                        # which is worse than not checking at all.
                        self.mag_learn_failed = True
                        print(f"\nESKF: magnetometer baseline not learned "
                              f"(median {med:.1f} uT, spread {spread*100:.1f}%)."
                              f"\n  Disturbance checking is off for this run. "
                              f"Likely something ferrous\n  near the vehicle at "
                              f"startup, or a calibration problem.")
            elif self.mag_field_expected is not None:
                dev = abs(mag.field_norm - self.mag_field_expected) \
                      / self.mag_field_expected
                if dev > MAG_FIELD_TOL:
                    self.mag_disturbed += 1
                    return False

        # First heading of the run bypasses the filter entirely -- see
        # set_yaw() for why a large initial yaw error breaks the
        # linearisation. Doing it here rather than leaving it to the
        # caller means it cannot be forgotten.
        if not self.yaw_initialized:
            self.set_yaw(mag.yaw, yaw_std=max(mag.std, np.deg2rad(10.0)))
            return True

        H = np.zeros((1, self.N))
        H[0, 8] = 1.0                    # index 8 = dtheta_D = heading error

        R_meas = np.array([[mag.std**2]])

        yaw_nom = quat_to_euler(self.nom.q)[2]
        dy = mag.yaw - yaw_nom
        dy = (dy + np.pi) % (2.0*np.pi) - np.pi    # wrap to [-pi, pi]:
        # without this, 179 deg vs -179 deg reads as a -358 deg error
        # instead of +2 deg, and the filter yanks the heading the wrong way
        innov = np.array([dy])

        nis = self._nis(H, R_meas, innov)
        self.nis_mag.append(nis)

        if not self._passes_gate(nis, chi2_gate(1), 'mag', H):
            return False
        return self._correct(H, R_meas, innov, 'mag')

    # --------------------------------------------------------
    #  ACCESSORS
    # --------------------------------------------------------
    @property
    def position(self):  return self.nom.p.copy()
    @property
    def velocity(self):  return self.nom.v.copy()
    @property
    def euler_deg(self): return np.rad2deg(quat_to_euler(self.nom.q))
    @property
    def pos_1sigma(self): return np.sqrt(np.diag(self.P)[0:3])
    @property
    def vel_1sigma(self): return np.sqrt(np.diag(self.P)[3:6])
    @property
    def accel_bias(self): return self.nom.ba.copy()
    @property
    def gyro_bias(self):  return self.nom.bg.copy()


# ============================================================
#  INITIALISATION HELPER
# ============================================================

def seed_static_alignment(accel_samples, gyro_samples,
                          gyro_still_thresh=0.02):
    """Initial gyro bias, roll and pitch from a stationary IMU window.

    Returns (bg, roll, pitch, ok). ok is False if the samples do not look
    stationary, in which case the caller should NOT use the seed -- a bias
    "estimate" taken while moving is worse than no estimate.

    Why this is not optional if you want ZUPT:

    ZUPT's stationarity detector tests |w_c| = |gyro - bg| against
    ZUPT_GYR_MAG. With bg = 0 (the NomState default) and a consumer MEMS
    part sitting at 6 deg/s of bias, |w_c| = 0.11 rad/s against a 0.05
    threshold -- never static, so ZUPT never fires, so bias is never
    observed, so |w_c| never drops. A closed loop with no way in.

    Bias-correcting the gyro inside the detector is necessary but does not
    break this on its own; the bias has to be non-zero to begin with.
    Another aiding source (GPS) can also break it, by converging bias the
    slow way -- which is why a test with GPS running will show ZUPT
    working and hide the deadlock entirely.

    The gyro mean IS the bias, because a stationary gyro should read only
    Earth rate (~15 deg/hr = 7.3e-5 rad/s), which is negligible against
    consumer MEMS bias. Roll and pitch come from the gravity direction.
    Yaw does not -- gravity says nothing about heading; that is set_yaw's
    job.
    """
    acc = np.asarray(accel_samples, dtype=float)
    gyr = np.asarray(gyro_samples, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3 or len(acc) < 10:
        return np.zeros(3), 0.0, 0.0, False

    bg = gyr.mean(axis=0)
    a_mean = acc.mean(axis=0)

    # Stationarity checks, both necessary. Gyro SPREAD catches motion
    # during the window; |accel| catches a window that was not at rest at
    # all (or wrong units), since a stationary accelerometer must read g.
    gyr_spread = float(np.max(np.std(gyr, axis=0)))
    a_norm = float(np.linalg.norm(a_mean))
    ok = gyr_spread < gyro_still_thresh and abs(a_norm - G_MAG) < 0.5

    # Level attitude from the gravity vector. Sign convention follows the
    # NED/FRD setup used throughout: at rest the accelerometer reads the
    # reaction to gravity, i.e. -g on body z when level.
    roll  = float(np.arctan2(-a_mean[1], -a_mean[2]))
    pitch = float(np.arctan2(a_mean[0],
                             np.hypot(a_mean[1], a_mean[2])))
    return bg, roll, pitch, ok


def make_P0(pos_std=5.0, vel_std=1.0, att_std_deg=2.0,
            ba_std=0.5, bg_std=np.deg2rad(10.0)) -> np.ndarray:
    """Initial covariance. Each block gets its own physically-reasoned
    1-sigma in its own units, then squared to variance.

    The bias defaults are sized for CONSUMER MEMS, not tactical grade.
    An MPU6050-class part can sit at 6 deg/s (~23,000 deg/hr) of gyro bias
    straight out of the box. A 10 deg/HR default -- as this function used
    to ship -- is then 2000x overconfident, and combined with NomState()
    defaulting bg to zeros and no static seeding, the filter starts certain
    of a bias value that is wrong by three orders of magnitude. It diverges
    and the covariance never opens up enough to recover.

    So: 10 deg/SECOND for gyro, 0.5 m/s^2 for accel. Deliberately loose.
    Seeding bg from a static average and tightening these to match is
    strictly better, but a loose prior recovers while a tight wrong one
    does not.

    P0 matters more than it looks: an attitude sigma that is too large
    tells the filter gravity might be leaking into horizontal accel by
    g*sin(sigma), which inflates velocity and position uncertainty fast
    during any GPS gap.
    """
    return np.diag(
        [pos_std**2]*3 +
        [vel_std**2]*3 +
        [np.deg2rad(att_std_deg)**2]*3 +
        [ba_std**2]*3 +
        [bg_std**2]*3
    )
