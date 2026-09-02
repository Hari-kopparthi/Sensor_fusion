"""
Live INS/GNSS/Baro fusion on Raspberry Pi: MPU6050 (I2C) + BMP180 (I2C) +
u-blox NEO-6M (UART) driving the 15-state LC-ESKF from navigation.py.

Extends pi_live_nav.py (IMU+GPS only) with a third sensor: the BMP180
barometer corrects vertical drift between GPS fixes at a much higher rate
(~8 Hz) than GPS (~1 Hz) can. Time synchronization between all three
sensors is nearest-neighbour, not fixed-schedule: the IMU loop is the
master clock (runs every iteration, whatever dt actually elapsed since
last iteration), and GPS / baro each get folded in on whichever loop
iteration they happen to be ready on. See PROJECT_STATUS.txt / the
navigation.py module docstring for why this is accurate enough here --
BMP180 noise (~0.25 m) and GPS noise (~2.5 m) both dwarf the sub-10 ms
timing slop this induces.

QNH (barometric reference) is not known in advance, and the real
atmosphere does not track the ISA standard atmosphere exactly -- so the
ISA-derived altitude is continuously re-anchored to GPS altitude at every
accepted GPS fix (see recalibrate_qnh() below). Between GPS fixes the QNH
offset is held constant, which is accurate to <1 m over a 1 Hz GPS
interval since pressure/temperature drift negligibly in that time.

Axis convention: same FRD body frame as pi_live_nav.py -- see that file's
docstring for the MPU6050 axis_map() rationale.

Run on the Pi:
    cd ~/dronepi-project && source venv/bin/activate
    python3 pi_live_nav_baro.py
Needs: numpy, scipy, mpu6050-raspberrypi, pyserial, pynmea2, smbus2,
plus navigation.py / gnss_module.py / imu_module.py / tc_module.py /
adaptive_noise.py / raim.py copied into the same directory.

If smbus2 install is blocked by PEP 668 ("externally-managed-environment"):
    pip install smbus2 --break-system-packages
or:
    sudo apt install python3-smbus
"""
import csv
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pynmea2
import serial
import smbus2
from mpu6050 import mpu6050

from imu_module import mems_imu_errors
from navigation import (
    ESKF, BaroMeas, GPSMeas, IMUMeas, MagMeas, NomState,
    euler_to_quat, lla_to_ned, ned_to_lla, quat_to_dcm, quat_to_euler,
)

# Magnetometer support is optional: this build now uses the LIS2MDL
# (swapped in after the LSM303D's Z-axis proved to be sitting in a strong
# static field, see lis2mdl_find_placement.py), but the fusion loop must
# still run with or without it rather than failing to start.
try:
    from lis2mdl_find_placement import LIS2MDL, UT_PER_LSB as MAG_UT_PER_LSB
    _MAG_AVAILABLE = True
except Exception:
    _MAG_AVAILABLE = False

DEG2RAD = np.pi / 180.0
KNOTS_TO_MS = 0.514444

GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600
I2C_BUS  = 1
BMP180_ADDR = 0x77
BMP180_OSS  = 3              # ultra-high-res: 25.5 ms/read, ~0.03 hPa noise

PRINT_PERIOD    = 0.5        # seconds between status prints
EST_LOG_PERIOD  = 0.1        # seconds between fused-estimate log rows (10 Hz)
BARO_PERIOD     = 0.125      # seconds between baro reads (~8 Hz, < 1/25.5ms budget)
BARO_STD        = 5.0        # m, 1-sigma -- conservative; ISA-vs-real-atmosphere
                              # error between GPS recalibrations, not sensor noise

# ── QNH re-anchoring rate ────────────────────────────────────────────────
# Fraction of the GPS-vs-ISA altitude discrepancy folded into qnh_offset at
# each accepted fix. NOT a hard replacement, which is what this used to do.
#
# Hard replacement assumed GPS altitude is accurate. Measured on this
# hardware it is not: a stationary 764 s run showed 70.6 m of raw GPS
# altitude spread (HDOP was fine at 1.37 mean -- this is multipath, not bad
# geometry). Re-anchoring to that every epoch injected GPS's vertical noise
# straight into the barometer's reference, so the baro-derived altitude
# stepped by tens of metres once per second while the filter's own estimate
# stayed smooth. Every baro measurement then looked like a gross outlier and
# the chi-squared gate rejected essentially all of them -- destroying the
# vertical channel the barometer exists to provide.
#
# Smoothing instead lets QNH track genuine pressure/weather drift (slow,
# minutes-to-hours) while averaging out GPS altitude noise (fast, per-epoch).
# At 1 Hz fixes this is roughly a 30 s time constant.
QNH_SMOOTH_ALPHA = 0.03
# m, 1-sigma pos noise at HDOP=1; scaled by HDOP below.
#
# Was 2.5 (the NEO-6M datasheet CEP, which assumes open sky and no
# multipath). Measured instead from a 690 s stationary run at this site:
# per-axis sample std was 7.32 m North / 4.96 m East against a mean HDOP of
# 1.37, i.e. an implied base of ~5.3 m. The datasheet figure was roughly
# 2x optimistic here, which told the chi-squared gate to expect tighter
# agreement than the receiver can actually deliver in this environment.
#
# Note this is a SITE-SPECIFIC measurement, not a universal constant --
# the dominant error is multipath, so it should be re-measured somewhere
# with a genuinely open sky view before being treated as final.
GPS_POS_STD_BASE = 5.3
# Vertical is measured at 1.95x the horizontal sigma on this hardware
# (14.30 m Up vs 7.32 m North over the same 690 s static log) -- consistent
# with the usual 2-3x GNSS rule of thumb. Applied as a ratio rather than an
# absolute so it still scales with HDOP alongside the horizontal term.
GPS_POS_STD_V_RATIO = 1.95
GPS_POS_STD_MIN  = 2.0       # m
GPS_POS_STD_MAX  = 50.0      # m, cap so a garbage HDOP doesn't zero out the gain
LOG_DIR = os.path.expanduser("~/dronepi-project/logs")

# ── GPS course-over-ground yaw aiding ────────────────────────────────────
# Without a magnetometer, yaw is unobservable: nothing in the accelerometer,
# gyro, GPS-position or baro measurements constrains rotation about the
# gravity vector. On this hardware it was measured drifting ~0.9 deg/s,
# reaching -171 deg over 190 s -- harmless while stationary, but it means a
# real acceleration gets projected into the wrong compass direction, which
# is the mechanism behind every divergence in the project's field logs.
#
# GPS course-over-ground closes that gap whenever the vehicle is actually
# moving. Two limitations are inherent, not implementation shortcuts:
#
#   1. It is meaningless at low speed. Course is derived from the velocity
#      vector; near standstill that vector is noise, and feeding it in
#      would inject random headings. Hence the speed gate below.
#
#   2. It measures direction of TRAVEL, not direction of POINTING. For a
#      car, or a person walking, these coincide. For an aircraft in
#      crosswind or a multirotor translating sideways they differ by the
#      crab/sideslip angle, and this update would then be biased by that
#      angle. A magnetometer remains the correct fix; this is aiding for
#      moving tests, not a replacement.
GPS_YAW_MIN_SPEED   = 1.0            # m/s -- below this, course is noise
GPS_VEL_ACCURACY    = 0.1            # m/s, NEO-6M datasheet velocity accuracy
GPS_YAW_STD_MIN     = 3.0 * DEG2RAD  # floor on reported heading uncertainty

# ── Magnetometer yaw aiding (LIS2MDL) ────────────────────────────────────
# Unlike GPS course-over-ground, this works at any speed including
# stationary, and measures where the vehicle POINTS rather than where it
# travels. It is the proper fix for the yaw-observability gap; the GPS
# course update above stays enabled as a cross-check and as a fallback for
# whenever the magnetometer is unavailable.
USE_MAG_YAW    = True
MAG_PERIOD     = 0.1                 # s -- 10 Hz, well under the 100 Hz ODR
MAG_YAW_STD    = 5.0 * DEG2RAD       # 1-sigma heading uncertainty
MAG_CAL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "lis2mdl_calibration.json")

# Magnetic declination: angle from magnetic north to TRUE north, positive
# east. The filter's yaw is referenced to true north, and GPS course is
# also true-referenced, so the magnetometer heading must be corrected or
# the two aiding sources would disagree by this angle. Near-zero in the UK
# at present; look up the current value for your location if elsewhere.
MAG_DECLINATION = 0.0 * DEG2RAD

# Consecutive failed magnetometer reads before giving up on it for the rest
# of the run. A few retries ride through a transient I2C glitch, but if the
# sensor is gone for good, continuing to retry every cycle would just stall
# the fusion loop with I2C timeouts.
MAG_MAX_CONSEC_FAIL = 50

# ── No-GPS mode (--no-gps) ───────────────────────────────────────────────
# Runs IMU + baro + magnetometer with no GNSS aiding at all. Intended for
# bench-testing the magnetometer yaw fix when the GPS is unavailable.
#
# What this mode CAN show: whether yaw now holds steady instead of drifting
# freely. That is the specific defect behind every divergence in this
# project's field logs, and the magnetometer is the fix for it.
#
# What it CANNOT show: anything about position or velocity accuracy. With
# no GNSS there is nothing bounding horizontal drift, so position will run
# away exactly as it does in pi_ins_live.py -- expected, not a fault.
#
# The reference position below only anchors the local NED frame and the
# gravity/Earth-rate model; it does not need to be exact. Default is taken
# from this project's own logged GPS fixes.
FALLBACK_REF_LAT = 52.048572
FALLBACK_REF_LON = -0.748767
FALLBACK_REF_ALT = 108.3


def load_mag_calibration(path: str = MAG_CAL_PATH):
    """Load hard/soft-iron calibration written by lis2mdl_calibrate.py.

    Returns (offset_lsb, scale) or None. Running uncalibrated is allowed
    but produces a heading with a systematic, heading-dependent error --
    see lis2mdl_calibrate.py's docstring for why that is worse than it
    sounds.
    """
    try:
        with open(path) as f:
            cal = json.load(f)
        return (np.array(cal["offset_lsb"], dtype=float),
                np.array(cal["scale"], dtype=float))
    except (OSError, KeyError, ValueError):
        return None


def mag_axis_map(v3):
    """LIS2MDL chip axes -> FRD body frame (x forward, y right, z down).

    MUST match however the magnetometer board is physically mounted, and
    must agree with the MPU-6050's axis_map() -- the two sensors have to
    describe the same body frame or tilt compensation mixes frames and the
    heading is wrong in a way that varies with attitude.

    The identity mapping below assumes the magnetometer is mounted in the
    same orientation as the IMU. Verify before trusting it: with the board
    level and pointing north, x should be strongly positive and y near
    zero; tilting nose-down should push z positive.
    """
    return np.array([v3[0], v3[1], v3[2]])


def tilt_compensated_heading(m_body, roll, pitch):
    """Heading (rad, 0 = north, positive east) from a body-frame magnetic
    vector, corrected for roll and pitch.

    Without this correction, tilting the vehicle rotates the measured field
    vector and the apparent heading swings with attitude -- unusable on
    anything that banks or pitches.

    Implemented by rotating the measurement into a LEVEL frame using the
    filter's own roll/pitch with yaw deliberately set to zero, so the
    residual horizontal angle is exactly the yaw being measured. Reuses
    navigation.py's tested quaternion/DCM code rather than open-coding the
    trigonometric form.
    """
    C_bn_level = quat_to_dcm(euler_to_quat(roll, pitch, 0.0))
    m_level = C_bn_level @ m_body
    return math.atan2(-m_level[1], m_level[0])

# ISA standard-atmosphere constants (troposphere, 0-11km)
_ISA_T0  = 288.15             # K, sea-level standard temperature
_ISA_L   = 0.0065             # K/m, temperature lapse rate
_ISA_P0  = 1013.25            # hPa, sea-level standard pressure
_ISA_EXP = 287.058 * _ISA_L / 9.80665   # ~0.190266


def isa_alt(p_hpa: float, t_k: float) -> float:
    """Pressure+temperature -> geometric altitude, ISA model with a
    temperature correction (real air is rarely exactly ISA-cold/warm).

    Raises ValueError outside 300-1100 hPa (covers everywhere from the
    highest inhabited ground to well below sea level). Guards against a
    corrupted I2C read: a negative or zero p_hpa raised to the
    non-integer _ISA_EXP silently returns a Python complex number
    instead of erroring, which then crashes the NEXT line's max()
    comparison rather than failing at the actual bad input.
    """
    if not (300.0 <= p_hpa <= 1100.0):
        raise ValueError(f"implausible pressure reading: {p_hpa} hPa")
    h_isa = (_ISA_T0 / _ISA_L) * (1.0 - (p_hpa / _ISA_P0) ** _ISA_EXP)
    t_isa = _ISA_T0 - _ISA_L * max(h_isa, 0.0)
    return h_isa * (t_k / t_isa)


# ═══════════════════════════════════════════════════════════════════════════
#  BMP180 DRIVER
# ═══════════════════════════════════════════════════════════════════════════

class BMP180:
    """Bosch BMP180 barometer over I2C. Reads factory calibration once at
    init, then applies the datasheet compensation formula on every read()."""

    _WAIT_MS = {0: 4.5e-3, 1: 7.5e-3, 2: 13.5e-3, 3: 25.5e-3}

    def __init__(self, bus: int = I2C_BUS, addr: int = BMP180_ADDR,
                 oss: int = BMP180_OSS):
        self._bus  = smbus2.SMBus(bus)
        self._addr = addr
        self._oss  = oss
        self._wait = self._WAIT_MS[oss]
        cal = self._bus.read_i2c_block_data(addr, 0xAA, 22)

        def s16(hi, lo):
            v = (hi << 8) | lo
            return v - 65536 if v > 32767 else v

        self.AC1 = s16(cal[0],  cal[1]);  self.AC2 = s16(cal[2],  cal[3])
        self.AC3 = s16(cal[4],  cal[5]);  self.AC4 = (cal[6]  << 8) | cal[7]
        self.AC5 = (cal[8]  << 8) | cal[9]
        self.AC6 = (cal[10] << 8) | cal[11]
        self.B1  = s16(cal[12], cal[13]); self.B2  = s16(cal[14], cal[15])
        self.MB  = s16(cal[16], cal[17]); self.MC  = s16(cal[18], cal[19])
        self.MD  = s16(cal[20], cal[21])

    def read(self):
        """Returns (pressure_hPa, temperature_C)."""
        self._bus.write_byte_data(self._addr, 0xF4, 0x2E)
        time.sleep(4.5e-3)
        d = self._bus.read_i2c_block_data(self._addr, 0xF6, 2)
        UT = (d[0] << 8) | d[1]

        self._bus.write_byte_data(self._addr, 0xF4, 0x34 | (self._oss << 6))
        time.sleep(self._wait)
        d = self._bus.read_i2c_block_data(self._addr, 0xF6, 3)
        UP = ((d[0] << 16) | (d[1] << 8) | d[2]) >> (8 - self._oss)

        X1 = ((UT - self.AC6) * self.AC5) >> 15
        X2 = (self.MC << 11) // (X1 + self.MD)
        B5 = X1 + X2
        T  = (B5 + 8) >> 4

        B6 = B5 - 4000
        X1 = (self.B2 * (B6 * B6 >> 12)) >> 11
        X2 = (self.AC2 * B6) >> 11
        X3 = X1 + X2
        B3 = (((self.AC1 * 4 + X3) << self._oss) + 2) >> 2
        X1 = (self.AC3 * B6) >> 13
        X2 = (self.B1 * (B6 * B6 >> 12)) >> 16
        X3 = ((X1 + X2) + 2) >> 2
        B4 = (self.AC4 * (X3 + 32768)) >> 15
        B7 = (UP - B3) * (50000 >> self._oss)
        p  = (B7 * 2) // B4 if B7 < 0x80000000 else (B7 // B4) * 2
        X1 = (p >> 8) ** 2
        X1 = (X1 * 3038) >> 16
        X2 = (-7357 * p) >> 16
        p  = p + ((X1 + X2 + 3791) >> 4)

        return p / 100.0, T / 10.0   # hPa, degC


# ═══════════════════════════════════════════════════════════════════════════
#  IMU HELPERS  (identical to pi_live_nav.py)
# ═══════════════════════════════════════════════════════════════════════════

def axis_map(v3):
    """MPU6050 flat/chip-up -> FRD body frame. Adjust if mounted differently."""
    return np.array([v3[0], -v3[1], -v3[2]])


# Accelerometer bias/scale correction, populated at startup from
# accel_calibration.json (written by accel_calibrate.py). Identity until
# loaded, so an uncalibrated run still works -- just with the raw error.
#
# This matters more than it sounds. Measured on this unit, the raw
# accelerometer reports gravity as 9.193 m/s^2 against a true 9.80665.
# The filter adds an exact 9.80665 gravity model, so the 0.614 m/s^2
# shortfall is indistinguishable from real downward acceleration and is
# integrated forever: it predicted a vertical velocity ramp of 0.31, 0.61,
# 0.92, 1.23 m/s at half-second intervals, and the live run produced 0.34,
# 0.63, 0.88, 1.18. Essentially all of the "dead-reckoning drift" seen
# while sitting still was this one uncalibrated scale factor.
ACCEL_BIAS  = np.zeros(3)
ACCEL_SCALE = np.ones(3)
ACCEL_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "accel_calibration.json")


def load_accel_calibration(path: str = ACCEL_CAL_PATH) -> bool:
    """Load bias/scale written by accel_calibrate.py. Returns True if found."""
    global ACCEL_BIAS, ACCEL_SCALE
    try:
        with open(path) as f:
            cal = json.load(f)
        ACCEL_BIAS  = np.array(cal["bias"],  dtype=float)
        ACCEL_SCALE = np.array(cal["scale"], dtype=float)
        return True
    except (OSError, KeyError, ValueError):
        return False


def read_imu(imu):
    a = imu.get_accel_data()          # m/s^2
    g = imu.get_gyro_data()           # deg/s
    accel = axis_map([a['x'], a['y'], a['z']])
    # Applied after axis_map so bias/scale are in the FRD body frame the
    # filter works in, matching how accel_calibrate.py measured them.
    accel = (accel - ACCEL_BIAS) / ACCEL_SCALE
    gyro = axis_map([g['x'], g['y'], g['z']]) * DEG2RAD
    return accel, gyro


def estimate_initial_attitude(imu, n_samples=50, sample_dt=0.02):
    """Coarse static leveling plus gyro-bias alignment.

    Returns (roll0, pitch0, gyro_bias). See pi_live_nav.py's docstring for
    why skipping the leveling (assuming roll=pitch=0) is a real hazard.

    The gyro bias comes free with the same samples: the vehicle is already
    being held still, and a stationary gyro's mean output IS its bias (Earth
    rate is 7.3e-5 rad/s, two orders of magnitude below what this sensor's
    bias turns out to be, so it is not worth subtracting here).

    Measured on this unit: gx sits at -0.1116 rad/s (-6.39 deg/s) at rest,
    against a P0 gyro-bias 1-sigma of 0.02 rad/s -- so the filter began 5.6x
    more confident than the truth warranted. With sig_bg_rw = 1e-8 the bias
    state could then move only ~2e-7 rad/s over an entire run, i.e. it was
    effectively frozen at zero and never converged.

    The visible consequence was in Q rather than in attitude: the
    manoeuvre-conditioned inflation term (1 + 50*||w_nb_b||) is designed to
    sit near 1.0 when straight and level, but with an uncorrected 0.11 rad/s
    bias it averaged 7.3 on a completely stationary bench run -- the filter
    was inflating its process noise sevenfold in response to a bias it
    should have estimated away.
    """
    accels = []
    gyros  = []
    for _ in range(n_samples):
        accel, gyro = read_imu(imu)
        accels.append(accel)
        gyros.append(gyro)
        time.sleep(sample_dt)
    f = np.mean(accels, axis=0)
    g_mag = np.linalg.norm(f)
    if abs(g_mag - 9.80665) > 1.0:
        print(f"WARNING: at-rest accel magnitude {g_mag:.2f} m/s^2 is far from "
              f"g=9.80665 -- keep the IMU still during leveling, retake if needed.")

    gyro_bias = np.mean(gyros, axis=0)
    # Spread across the averaging window. If the vehicle was NOT actually
    # still, this is motion rather than bias and seeding it would inject a
    # permanent error -- so it is checked rather than trusted blindly.
    gyro_spread = np.std(gyros, axis=0)
    if np.any(gyro_spread > 0.05):
        print(f"WARNING: gyro varied by {np.round(gyro_spread, 4)} rad/s during "
              f"alignment.\n  That looks like motion, not bias -- the vehicle "
              f"must be still. Bias NOT seeded.")
        gyro_bias = np.zeros(3)

    roll0  = math.atan2(-f[1], -f[2])
    pitch0 = math.atan2(f[0], math.hypot(f[1], f[2]))
    return roll0, pitch0, gyro_bias


def open_logs():
    """Five CSVs per run: raw IMU, raw GPS, raw baro, raw magnetometer, and
    the fused estimate (10 Hz)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    f_imu  = open(f"{LOG_DIR}/{stamp}_imu.csv",  "w", newline="")
    f_gps  = open(f"{LOG_DIR}/{stamp}_gps.csv",  "w", newline="")
    f_baro = open(f"{LOG_DIR}/{stamp}_baro.csv", "w", newline="")
    f_mag  = open(f"{LOG_DIR}/{stamp}_mag.csv",  "w", newline="")
    f_est  = open(f"{LOG_DIR}/{stamp}_est.csv",  "w", newline="")
    w_imu, w_gps, w_baro, w_mag, w_est = (csv.writer(f) for f in
                                           (f_imu, f_gps, f_baro, f_mag, f_est))
    w_imu.writerow(["t", "dt", "ax", "ay", "az", "gx", "gy", "gz"])  # FRD, m/s^2, rad/s
    w_gps.writerow(["t", "lat_deg", "lon_deg", "alt_m", "vN", "vE", "hdop",
                    "accepted", "course_deg", "yaw_std_deg", "yaw_accepted"])
    w_baro.writerow(["t", "p_hpa", "t_c", "isa_alt_m", "qnh_offset_m",
                     "alt_above_ref_m", "accepted"])
    # ba_*/bg_* are the filter's estimated IMU biases. Logged because
    # without them there is no way to check the single most basic question
    # about an ESKF -- whether its bias states actually converge. A run can
    # look healthy on position and attitude while the bias states sit frozen
    # at their initial values, which is exactly what was happening before
    # the gyro bias was seeded at alignment.
    w_est.writerow(["t", "pN", "pE", "pD", "vN", "vE", "vD",
                    "roll_deg", "pitch_deg", "yaw_deg",
                    "lat_deg", "lon_deg", "alt_m",
                    "gps_updates", "gps_attempts", "baro_updates",
                    "baro_attempts", "yaw_updates", "yaw_attempts",
                    "mag_updates", "mag_attempts",
                    "ba_x", "ba_y", "ba_z", "bg_x", "bg_y", "bg_z"])
    # mag_x/y/z are calibration-corrected LSB counts; yaw_mag_deg is the
    # tilt-compensated heading derived from them, and yaw_filter_deg the
    # filter's estimate at that instant -- logging both makes it possible to
    # see afterwards whether the magnetometer was actually pulling yaw into
    # agreement, or being rejected by the chi2 gate.
    w_mag.writerow(["t", "mag_x", "mag_y", "mag_z",
                    "yaw_mag_deg", "yaw_filter_deg", "accepted"])
    print(f"Logging to {LOG_DIR}/{stamp}_{{imu,gps,baro,mag,est}}.csv")
    return ((f_imu, f_gps, f_baro, f_mag, f_est),
            (w_imu, w_gps, w_baro, w_mag, w_est))


def _sigterm_handler(signum, frame):
    """Turn SIGTERM into the same KeyboardInterrupt the Ctrl+C path handles.

    Needed for unattended operation under systemd. 'systemctl stop' and a
    clean shutdown both send SIGTERM, whose default action kills the process
    outright -- the 'finally' block never runs, the CSV writers never flush,
    and the tail of every log is lost. Routing it through the existing
    interrupt path means an unattended run ends exactly like a Ctrl+C one.
    """
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)

    no_gps = "--no-gps" in sys.argv

    imu  = mpu6050(0x68)
    baro = BMP180()
    ser  = None if no_gps else serial.Serial(GPS_PORT, GPS_BAUD, timeout=0)
    files, (w_imu, w_gps, w_baro, w_mag, w_est) = open_logs()

    if load_accel_calibration():
        print(f"Accel calibration loaded: bias={np.round(ACCEL_BIAS, 4)} "
              f"scale={np.round(ACCEL_SCALE, 5)}")
    else:
        print("WARNING: no accel_calibration.json found. This unit's raw")
        print("  accelerometer reads gravity ~6% low, which the filter cannot")
        print("  distinguish from real downward acceleration -- velocity will")
        print("  ramp continuously even at rest. Run accel_calibrate.py.")

    accel, _ = read_imu(imu)
    g_mag = float(np.linalg.norm(accel))
    print(f"At-rest accel (FRD, expect ~[0, 0, -9.8]): {np.round(accel, 2)}"
          f"   |a| = {g_mag:.3f} m/s^2")
    if abs(g_mag - 9.80665) > 0.1:
        print(f"  ** |a| is off true gravity by {g_mag - 9.80665:+.3f} m/s^2."
              f" That error integrates\n     into a velocity ramp of the same"
              f" size per second.")
    p0, t0 = baro.read()
    print(f"Baro at rest: {p0:.2f} hPa, {t0:.1f} C")

    # ── Magnetometer (optional) ──────────────────────────────────────────
    mag = None
    mag_cal = None
    if USE_MAG_YAW and _MAG_AVAILABLE:
        try:
            mag = LIS2MDL()
            m0 = mag.read_raw()
            print(f"Magnetometer at rest (raw LSB): {m0}")
            mag_cal = load_mag_calibration()
            if mag_cal is None:
                print("  WARNING: no lis2mdl_calibration.json found. Heading")
                print("  will carry a systematic, heading-dependent error from")
                print("  hard-iron distortion. Run lis2mdl_calibrate.py first.")
            else:
                off, scl = mag_cal
                print(f"  Calibration loaded: offset={np.round(off, 1)} LSB, "
                      f"scale={np.round(scl, 3)}")
        except Exception as e:
            print(f"  Magnetometer unavailable ({e.__class__.__name__}): {e}")
            print("  Continuing without it -- GPS course-over-ground will be")
            print("  the only yaw aiding, so yaw stays unobservable at rest.")
            mag = None
    elif USE_MAG_YAW:
        print("Magnetometer driver not importable -- continuing without it.")

    print("Estimating initial attitude (keep the vehicle still)...")
    roll0, pitch0, gyro_bias0 = estimate_initial_attitude(imu)
    print(f"Initial gyro bias (FRD, rad/s): {np.round(gyro_bias0, 5)}"
          f"   = {np.round(gyro_bias0 / DEG2RAD, 3)} deg/s")
    print(f"Initial roll={math.degrees(roll0):6.1f} deg  pitch={math.degrees(pitch0):6.1f} deg "
          f"(yaw unknown, starts at 0 -- no compass)")

    # ── Establish a provisional NED origin ───────────────────────────────
    # Deliberately does NOT wait for a GPS fix. The IMU, barometer and
    # magnetometer are all usable immediately, and blocking on GNSS would
    # throw away everything they could measure during a cold start -- which
    # can be 30-60 s outdoors, and forever indoors.
    #
    # Instead the local frame starts anchored to a fallback reference, and
    # the first valid GPS fix RE-ANCHORS it (see the gps_origin_set branch
    # in the main loop). Until then the filter dead-reckons: attitude and
    # heading are meaningful, position is not.
    #
    # Re-anchoring rather than treating the first fix as an ordinary
    # measurement matters for two reasons. lla_to_ned() is a flat-Earth
    # approximation only valid near its reference, so a fallback hundreds
    # of kilometres from the truth would distort the conversion. And the
    # innovation against a dead-reckoned position would be enormous, so the
    # chi-squared gate would reject the very fix that was meant to fix it.
    ref_lat = FALLBACK_REF_LAT * DEG2RAD
    ref_lon = FALLBACK_REF_LON * DEG2RAD
    ref_alt = FALLBACK_REF_ALT
    gps_origin_set = False
    buf = ""

    p_now, t_now = baro.read()
    try:
        qnh_offset = ref_alt - isa_alt(p_now, t_now + 273.15)
    except ValueError as e:
        print(f"WARNING: bad initial baro reading ({e}), starting with "
              f"qnh_offset=0.0 -- it will self-correct at the first GPS fix.")
        qnh_offset = 0.0

    if no_gps:
        print("\n*** NO-GPS MODE *** (--no-gps)")
        print("  GNSS disabled entirely. Position and velocity are unbounded")
        print("  and will drift without limit -- expected, not a fault. What")
        print("  this run tests is whether the magnetometer holds YAW steady.")
    else:
        print("\nStarting immediately on IMU + Baro"
              + (" + Mag" if mag is not None else "") + ".")
        print("  GPS will be folded in automatically when it acquires a fix;")
        print("  the local frame is re-anchored to that first fix. Until then")
        print("  position is dead-reckoned and will drift.")
    print(f"  Provisional origin: lat={ref_lat/DEG2RAD:.6f} "
          f"lon={ref_lon/DEG2RAD:.6f} alt={ref_alt:.1f}m  "
          f"qnh_offset={qnh_offset:+.1f}m")

    # ── Initial state ────────────────────────────────────────────────────────
    nom0 = NomState(
        p=np.zeros(3),
        v=np.zeros(3),
        q=euler_to_quat(roll0, pitch0, 0.0),
        ba=np.zeros(3),
        # Seeded from the static alignment rather than assumed zero -- see
        # estimate_initial_attitude() for what an unseeded 6.4 deg/s bias
        # did to the manoeuvre-conditioned Q term.
        bg=gyro_bias0,
    )
    P0 = np.diag(
        [5.0**2] * 3 +
        [0.5**2] * 3 +
        [np.deg2rad(3.0)**2] * 2 + [np.deg2rad(60.0)**2] +
        [0.1**2] * 3 +
        [0.02**2] * 3
    )
    imu_errs = mems_imu_errors()
    noise_cfg = dict(
        sig_accel=float(3.0 * imu_errs['accel_noise_root_PSD']),
        sig_gyro=float(3.0 * imu_errs['gyro_noise_root_PSD']),
        sig_vel_rw=0.05,
        sig_pos_rw=0.05,
    )
    # adaptive_gps_r=False: the adapter anchors its "nominal" R to whichever
    # epoch happens to be the FIRST GPS update (see navigation.py's
    # AdaptiveR wiring) and never re-anchors it as HDOP changes afterward.
    # On real hardware HDOP genuinely swings (e.g. 0.8 in calm conditions
    # to 4-5+ during handling/multipath), so a tight-nominal anchor from an
    # early good fix silently overrides the correctly HDOP-scaled R computed
    # fresh every call -- making exactly the fixes taken during real motion
    # (when HDOP is worst and a correction is needed most) look like
    # statistical outliers and get rejected by the chi2 gate. The synthetic
    # simulation data this adapter was tuned against doesn't have this
    # failure mode (its GPS noise is far more homogeneous epoch to epoch),
    # so this override is hardware-specific, not a navigation.py change.
    ekf = ESKF(nom0, P0, ref_lat, ref_lon, ref_alt, noise_cfg=noise_cfg,
              adaptive_gps_r=False)

    last_vel_ned = np.zeros(3)
    have_vel = False
    n_gps = n_gps_attempts = 0
    n_baro = n_baro_attempts = 0
    n_yaw = n_yaw_attempts = 0
    n_mag = n_mag_attempts = 0
    mag_consec_fail = 0
    last_mag = 0.0
    gps_serial_fail_count = 0
    # Most recent VALID (p_hpa, t_c) from the 8 Hz baro loop. GPS-epoch QNH
    # recalibration reuses this instead of issuing its own baro.read(): each
    # read blocks ~30 ms in the BMP180's conversion sleeps, and doing that
    # inside the GPS branch stalled IMU propagation and added avoidable
    # traffic to an I2C bus already shared with the MPU6050 and LIS2MDL.
    # None until the first good read, in which case QNH simply is not
    # recalibrated that epoch.
    last_baro_sample = None
    # (course_rad, std_rad, accepted) of the most recent yaw update, logged
    # alongside the next GPS position fix so the two can be correlated.
    last_yaw_meas = None
    last_t = time.time()
    t_start = last_t          # run start, used to report time-to-first-fix
    last_baro = 0.0
    last_print = 0.0
    last_est_log = 0.0
    buf = ""

    sensors = "IMU + Baro" if no_gps else "IMU + GPS + Baro"
    if mag is not None:
        sensors += " + Mag"
    print(f"Fusion running ({sensors}). Ctrl+C to stop.")
    if no_gps and mag is not None:
        print("  Watch the yaw= field: it should HOLD STEADY rather than")
        print("  drifting. Turn the vehicle and it should follow, then settle.")
    try:
      while True:
        # ── IMU predict -- always runs, master clock ─────────────────────
        now = time.time()
        dt = now - last_t
        last_t = now
        accel, gyro = read_imu(imu)
        ekf.predict(IMUMeas(dt=dt, accel=accel, gyro=gyro))
        w_imu.writerow([f"{now:.4f}", f"{dt:.5f}",
                        *[f"{v:.5f}" for v in accel],
                        *[f"{v:.6f}" for v in gyro]])

        # ── Baro update -- nearest-neighbour, ~8 Hz ──────────────────────
        if now - last_baro >= BARO_PERIOD:
            last_baro = now
            p_hpa, t_c = baro.read()
            try:
                isa_val = isa_alt(p_hpa, t_c + 273.15)
            except ValueError as e:
                # A corrupted I2C read is skipped rather than crashing the
                # whole fusion loop over one bad sample.
                print(f"\nBad baro reading skipped: {e}")
                n_baro_attempts += 1
            else:
                last_baro_sample = (p_hpa, t_c)
                h_msl = isa_val + qnh_offset
                alt_above_ref = h_msl - ref_alt
                accepted = ekf.update_baro(BaroMeas(altitude=alt_above_ref, std=BARO_STD))
                n_baro_attempts += 1
                if accepted:
                    n_baro += 1
                w_baro.writerow([f"{now:.4f}", f"{p_hpa:.2f}", f"{t_c:.1f}",
                                 f"{isa_val:.2f}",
                                 f"{qnh_offset:+.2f}", f"{alt_above_ref:.2f}",
                                 int(accepted)])

        # ── Magnetometer yaw update -- nearest-neighbour, ~10 Hz ─────────
        # The LIS2MDL's 100 Hz ODR comfortably outpaces this 10 Hz poll, but
        # data_ready() is still checked so a read never returns a sample
        # already consumed by the previous poll.
        #
        # Two structural points, both learned from failures on this build:
        #
        #  * last_mag is stamped when the PERIOD elapses, not when a read
        #    succeeds. Gating the whole branch on data_ready() instead left
        #    last_mag unstamped whenever the sensor had no new sample, so it
        #    was re-polled on every loop iteration (~200 Hz) rather than at
        #    10 Hz -- flooding an I2C bus shared with the MPU6050 and the
        #    BMP180, whose multi-byte pressure reads are the most
        #    corruption-prone traffic on it.
        #  * data_ready() is itself an I2C transaction and can raise. It
        #    belongs inside the try, not in the loop condition, where an
        #    OSError would escape unhandled and kill the run -- the same
        #    failure mode already fixed for the baro and GPS paths.
        if mag is not None and now - last_mag >= MAG_PERIOD:
            last_mag = now
            try:
                if mag.data_ready():
                    m_raw = np.array(mag.read_raw(), dtype=float)
                    mag_consec_fail = 0

                    if mag_cal is not None:
                        off, scl = mag_cal
                        m_raw = (m_raw - off) * scl      # hard + soft iron

                    m_body = mag_axis_map(m_raw)

                    # Tilt-compensate using the filter's current roll/pitch.
                    # Those are well observed (gravity constrains them via
                    # the accelerometer), unlike yaw -- which is exactly the
                    # state this measurement is here to fix.
                    r_now, p_now_att, _ = quat_to_euler(ekf.nom.q)
                    yaw_mag = tilt_compensated_heading(m_body, r_now,
                                                       p_now_att)
                    yaw_true = yaw_mag + MAG_DECLINATION

                    mag_ok = ekf.update_mag(MagMeas(yaw=yaw_true,
                                                    std=MAG_YAW_STD))
                    n_mag_attempts += 1
                    if mag_ok:
                        n_mag += 1
                    w_mag.writerow([f"{now:.4f}",
                                    *[f"{v:.1f}" for v in m_raw],
                                    f"{yaw_true / DEG2RAD:.2f}",
                                    f"{quat_to_euler(ekf.nom.q)[2] / DEG2RAD:.2f}",
                                    int(mag_ok)])

            except OSError:
                # A dropped I2C read is skipped rather than retried inline,
                # so a failing magnetometer never stalls the IMU propagation.
                mag_consec_fail += 1
                n_mag_attempts += 1
                if mag_consec_fail >= MAG_MAX_CONSEC_FAIL:
                    print(f"\nMagnetometer failed {mag_consec_fail} reads in a "
                          f"row -- disabling it for the rest of this run.\n"
                          f"GPS course-over-ground remains as yaw aiding.")
                    mag = None

        # ── GPS update -- nearest-neighbour, non-blocking read, ~1 Hz ────
        try:
            data = ser.read(2048) if ser is not None else b""
        except serial.SerialException as e:
            # A transient USB/UART hiccup is skipped rather than crashing
            # the whole fusion loop -- same philosophy as the mag/baro
            # error handling above. Unlike the mag, GPS has no fallback
            # sensor, so it is never disabled outright, only retried.
            data = b""
            gps_serial_fail_count += 1
            if gps_serial_fail_count == 1 or gps_serial_fail_count % 50 == 0:
                print(f"\nGPS serial read failed ({gps_serial_fail_count} "
                      f"times so far): {e}")
            time.sleep(0.05)
        if data:
            buf += data.decode('ascii', errors='replace')
        while '\n' in buf:
            line, buf = buf.split('\n', 1)
            line = line.strip()
            if line.startswith(('$GPRMC', '$GNRMC')):
                try:
                    msg = pynmea2.parse(line)
                    if (msg.status == 'A' and msg.spd_over_grnd is not None
                            and msg.true_course is not None):
                        spd = float(msg.spd_over_grnd) * KNOTS_TO_MS
                        crs = float(msg.true_course) * DEG2RAD
                        # RMC carries speed-over-ground and course, so only
                        # the HORIZONTAL velocity is measured. The third
                        # component is a structural placeholder, not a
                        # reading -- it is flagged as such via vel_valid at
                        # the update call so the filter excludes it rather
                        # than treating "0.0" as a vertical measurement.
                        last_vel_ned = np.array([spd * math.cos(crs),
                                                 spd * math.sin(crs), 0.0])
                        have_vel = True

                        # ── Course-over-ground yaw update ────────────────
                        # See the GPS_YAW_* block at the top for why this
                        # exists and what its two limitations are.
                        if spd >= GPS_YAW_MIN_SPEED:
                            # Two error sources, and the larger one wins:
                            #
                            #  * At low speed, GPS velocity noise dominates:
                            #    a fixed ~0.1 m/s error subtends a bigger
                            #    heading angle the slower you go (5.7 deg at
                            #    1 m/s, 4.1 deg at walking pace).
                            #
                            #  * Above ~2 m/s that term falls below the
                            #    floor, and the floor takes over. The floor
                            #    is NOT GPS noise -- the receiver's heading
                            #    spec is 0.5 deg. It represents the
                            #    travel-vs-pointing model error: course over
                            #    ground only equals true heading when there
                            #    is no crab or sideslip. Claiming 0.5 deg
                            #    would tell the filter to trust this far
                            #    more than the underlying assumption
                            #    deserves.
                            yaw_std = max(math.atan2(GPS_VEL_ACCURACY, spd),
                                          GPS_YAW_STD_MIN)
                            yaw_ok = ekf.update_mag(MagMeas(yaw=crs,
                                                            std=yaw_std))
                            n_yaw_attempts += 1
                            if yaw_ok:
                                n_yaw += 1
                            last_yaw_meas = (crs, yaw_std, int(yaw_ok))
                except (pynmea2.ParseError, ValueError, TypeError):
                    pass
            elif line.startswith(('$GPGGA', '$GNGGA')):
                try:
                    msg = pynmea2.parse(line)
                    if msg.gps_qual and int(msg.gps_qual) > 0:
                        hdop = float(msg.horizontal_dil) if msg.horizontal_dil else 2.0
                        pos_std = float(np.clip(GPS_POS_STD_BASE * hdop,
                                                GPS_POS_STD_MIN, GPS_POS_STD_MAX))
                        vel = last_vel_ned if have_vel else np.zeros(3)

                        if not gps_origin_set:
                            # ── First fix: RE-ANCHOR the local frame ─────
                            # Everything dead-reckoned so far is discarded
                            # rather than reconciled: without GNSS there was
                            # nothing bounding horizontal drift, so that
                            # position carried no information worth keeping.
                            #
                            # Attitude and bias KEEP their current estimate
                            # (there is no better guess to replace them with),
                            # but their COVARIANCE is inflated back to the
                            # startup uncertainty too, not just position and
                            # velocity's. Measured on this hardware: a long
                            # time-to-first-fix let position run away to
                            # >7 km, and roll/pitch -- which should stay near
                            # 0 deg sitting still -- diverged right alongside
                            # it (up to 178 deg) and never recovered even
                            # after GPS pulled position back to normal. The
                            # filter kept trusting that wrong attitude at its
                            # original tight confidence, so every subsequent
                            # GPS/baro/mag correction looked like an outlier
                            # against it and got rejected by the chi-squared
                            # gate -- explaining why ALL THREE sensors' accept
                            # rates crashed together, not just the one that
                            # actually drifted. Re-inflating here costs
                            # nothing if attitude/bias were fine (later
                            # updates just re-tighten it quickly); it is the
                            # only way to recover if they were not.
                            ref_lat = msg.latitude * DEG2RAD
                            ref_lon = msg.longitude * DEG2RAD
                            ref_alt = float(msg.altitude)
                            ekf.ref_lat, ekf.ref_lon, ekf.ref_alt = (
                                ref_lat, ref_lon, ref_alt)
                            ekf.nom.p = np.zeros(3)
                            ekf.nom.v = vel.copy()
                            # Position/velocity are now known to GNSS
                            # accuracy, so replace the inflated dead-reckoned
                            # covariance instead of letting it stay huge.
                            ekf.P[0:3, 0:3] = np.eye(3) * pos_std ** 2
                            ekf.P[3:6, 3:6] = np.eye(3) * 0.5 ** 2
                            # Attitude (deg2rad(3) roll/pitch, deg2rad(60)
                            # yaw) and bias -- same startup values as P0 in
                            # main(), so post-reanchor confidence matches
                            # cold-start confidence rather than the filter's
                            # (possibly wrong) pre-fix certainty.
                            ekf.P[6:9, 6:9] = np.diag(
                                [np.deg2rad(3.0) ** 2] * 2
                                + [np.deg2rad(60.0) ** 2])
                            ekf.P[9:12, 9:12] = np.eye(3) * 0.1 ** 2
                            ekf.P[12:15, 12:15] = np.eye(3) * 0.02 ** 2
                            # Hard set (not smoothed) is correct HERE and only
                            # here: this establishes the initial reference
                            # rather than tracking drift in an existing one,
                            # so there is no prior estimate worth easing from.
                            # Uses the cached sample rather than a fresh
                            # blocking read -- see last_baro_sample.
                            if last_baro_sample is not None:
                                p_baro, t_baro = last_baro_sample
                                try:
                                    qnh_offset = ref_alt - isa_alt(
                                        p_baro, t_baro + 273.15)
                                except ValueError as e:
                                    print(f"\nWARNING: bad baro reading at GPS "
                                          f"acquisition ({e}), keeping prior "
                                          f"qnh_offset -- next accepted fix "
                                          f"will retry.")
                            gps_origin_set = True
                            print(f"\nGPS ACQUIRED after {now - t_start:.1f}s "
                                  f"-- origin re-anchored to "
                                  f"{msg.latitude:.6f}, {msg.longitude:.6f}, "
                                  f"{ref_alt:.1f}m (HDOP {hdop:.2f})")
                            accepted = True
                            n_gps_attempts += 1
                            n_gps += 1
                            w_gps.writerow([f"{now:.4f}",
                                            f"{msg.latitude:.7f}",
                                            f"{msg.longitude:.7f}",
                                            f"{ref_alt:.2f}",
                                            f"{vel[0]:.3f}", f"{vel[1]:.3f}",
                                            f"{hdop:.2f}", 1, "", "", ""])
                            continue

                        pos_ned = lla_to_ned(
                            msg.latitude * DEG2RAD, msg.longitude * DEG2RAD,
                            float(msg.altitude), ref_lat, ref_lon, ref_alt)
                        # Which velocity axes are real:
                        #   have_vel True  -> N/E measured by RMC, D is not
                        #   have_vel False -> nothing measured yet; `vel` is
                        #                     np.zeros(3), pure placeholder
                        #
                        # Before this was flagged, the zeros were fused as a
                        # measurement: a full three-axis zero-velocity update
                        # asserted at sigma=0.5 m/s before the first RMC, and
                        # a vD == 0 assertion at every epoch thereafter. The
                        # pre-RMC case overlapped precisely with the 114 s
                        # time-to-first-fix window in which this build's
                        # position ran away to >7 km and roll/pitch diverged
                        # to ~178 deg -- the filter was being told it was
                        # stationary while the IMU said otherwise, and that
                        # contradiction is resolved into attitude/bias error.
                        vel_valid = ((True, True, False) if have_vel
                                     else (False, False, False))
                        accepted = ekf.update_gps(GPSMeas(
                            position=pos_ned, velocity=vel,
                            pos_std=pos_std,
                            pos_std_v=pos_std * GPS_POS_STD_V_RATIO,
                            vel_std=0.5, vel_valid=vel_valid))
                        n_gps_attempts += 1
                        if accepted:
                            n_gps += 1
                        # ── GPS-aided QNH recalibration ──────────────────
                        # Two guards the original version lacked, both of
                        # which caused the barometer to be rejected wholesale
                        # (see QNH_SMOOTH_ALPHA at the top for the measured
                        # numbers):
                        #
                        #  1. Only ACCEPTED fixes are used. A fix the chi2
                        #     gate just rejected as an outlier is exactly the
                        #     one whose altitude should not be trusted to
                        #     re-anchor the vertical reference -- but the old
                        #     code recalibrated on every GGA regardless.
                        #  2. The offset is EASED toward the new estimate
                        #     rather than replaced outright, so per-epoch GPS
                        #     altitude noise averages out instead of stepping
                        #     the baro's reference by tens of metres a second.
                        if accepted and last_baro_sample is not None:
                            p_now, t_now = last_baro_sample
                            try:
                                qnh_target = float(msg.altitude) - isa_alt(
                                    p_now, t_now + 273.15)
                            except ValueError:
                                # Bad cached reading -- keep the prior offset
                                # and retry at the next accepted fix.
                                pass
                            else:
                                qnh_offset += QNH_SMOOTH_ALPHA * (
                                    qnh_target - qnh_offset)
                        if last_yaw_meas is None:
                            yaw_cols = ["", "", ""]
                        else:
                            c, s, ok = last_yaw_meas
                            yaw_cols = [f"{c / DEG2RAD:.1f}",
                                        f"{s / DEG2RAD:.2f}", ok]
                        w_gps.writerow([f"{now:.4f}",
                                        f"{msg.latitude:.7f}",
                                        f"{msg.longitude:.7f}",
                                        f"{float(msg.altitude):.2f}",
                                        f"{vel[0]:.3f}", f"{vel[1]:.3f}",
                                        f"{hdop:.2f}", int(accepted)]
                                       + yaw_cols)
                except (pynmea2.ParseError, ValueError, TypeError):
                    pass

        # ── Fused-estimate log (10 Hz) ───────────────────────────────────
        if now - last_est_log >= EST_LOG_PERIOD:
            last_est_log = now
            lat, lon, alt = ned_to_lla(ekf.nom.p, ref_lat, ref_lon, ref_alt)
            rpy = quat_to_euler(ekf.nom.q) / DEG2RAD
            w_est.writerow([f"{now:.4f}",
                            *[f"{v:.3f}" for v in ekf.nom.p],
                            *[f"{v:.3f}" for v in ekf.nom.v],
                            *[f"{v:.2f}" for v in rpy],
                            f"{lat/DEG2RAD:.7f}", f"{lon/DEG2RAD:.7f}",
                            f"{alt:.2f}", n_gps, n_gps_attempts,
                            n_baro, n_baro_attempts, n_yaw, n_yaw_attempts,
                            n_mag, n_mag_attempts,
                            *[f"{v:.6f}" for v in ekf.nom.ba],
                            *[f"{v:.7f}" for v in ekf.nom.bg]])

            if now - last_print >= PRINT_PERIOD:
                last_print = now
                # Flag dead-reckoning explicitly: without it, a plausible
                # looking position printed before GPS acquisition invites
                # the reader to trust a number that is pure drift.
                fix_tag = "" if gps_origin_set else "[NO GPS - DR] "
                print(f"{fix_tag}"
                      f"NED p=({ekf.nom.p[0]:7.2f},{ekf.nom.p[1]:7.2f},{ekf.nom.p[2]:6.2f})m "
                      f"v=({ekf.nom.v[0]:5.2f},{ekf.nom.v[1]:5.2f},{ekf.nom.v[2]:5.2f})m/s "
                      f"yaw={rpy[2]:6.1f}deg alt={alt:7.2f}m "
                      f"gps={n_gps}/{n_gps_attempts} baro={n_baro}/{n_baro_attempts} "
                      f"yawfix={n_yaw}/{n_yaw_attempts} "
                      f"mag={n_mag}/{n_mag_attempts} "
                      f"lat={lat/DEG2RAD:.6f} lon={lon/DEG2RAD:.6f}")

        time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nStopping, closing log files...")
    finally:
        for f in files:
            f.close()
        print("Logs saved.")


if __name__ == "__main__":
    main()
