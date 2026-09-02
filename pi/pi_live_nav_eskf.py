"""
Live INS/GNSS/Baro/Mag fusion on the Pi, driving eskf.py.

Bridge between the hardware/logging layer that already works
(pi_live_nav_baro.py: BMP180 and LIS2MDL drivers, NMEA parsing, CSV
logging, systemd handling) and the newer filter in eskf.py.

Written as a SEPARATE script rather than an edit to pi_live_nav_baro.py
on purpose: the Pi is the only test rig, and a bridge that fails should
leave a known-good script to fall back to.

What changes versus pi_live_nav_baro.py
---------------------------------------
  * every update takes a TIMESTAMP -- eskf.py rejects out-of-order
    measurements per sensor, which needs the measurement's own time
  * GPSMeas splits horizontal and vertical sigma (pos_std_h/pos_std_v)
  * MagMeas carries field_norm in MICROTESLA for the disturbance check.
    The logger writes calibration-corrected LSB counts, so it is scaled
    here -- passing raw counts puts the learned baseline outside
    eskf.py's 20-80 uT sanity band, and the check then disables itself
    silently for the whole run.
  * ZUPT runs at 10 Hz. The detector inside the filter decides whether it
    applies, so calling unconditionally is safe.
  * gyro bias, roll and pitch are seeded from a stationary window before
    the filter is constructed. Not optional: with bg = 0 and this
    hardware's ~6 deg/s bias, the ZUPT detector never sees a stationary
    sample and ZUPT never fires.

Run on the Pi:
    cd ~/dronepi-project && source venv/bin/activate
    python3 pi_live_nav_eskf.py [--no-gps]

Needs eskf.py copied alongside it.
"""
import math
import signal
import sys
import time

import numpy as np
import pynmea2
import serial

from mpu6050 import mpu6050

# Filter under test.
from eskf import (ESKF, NomState, IMUMeas, GPSMeas, BaroMeas, MagMeas,
                  make_P0, seed_static_alignment, quat_to_euler, G_MAG)

# Frame conversions. eskf.py is deliberately frame-agnostic and does not
# provide them; geodetic.py carries the three that are actually needed,
# extracted verbatim from the project's navigation package so this runs
# standalone.
from geodetic import lla_to_ned, ned_to_lla, euler_to_quat

# Hardware + logging, reused wholesale rather than reimplemented.
from pi_live_nav_baro import (
    BMP180, read_imu, mag_axis_map, tilt_compensated_heading, isa_alt,
    load_accel_calibration, load_mag_calibration, open_logs,
    ACCEL_BIAS, ACCEL_SCALE,
    GPS_PORT, GPS_BAUD, BARO_PERIOD, BARO_STD, MAG_PERIOD, MAG_YAW_STD,
    MAG_DECLINATION, MAG_MAX_CONSEC_FAIL, PRINT_PERIOD, EST_LOG_PERIOD,
    GPS_POS_STD_BASE, GPS_POS_STD_MIN, GPS_POS_STD_MAX,
    GPS_POS_STD_V_RATIO, GPS_YAW_MIN_SPEED, GPS_VEL_ACCURACY,
    GPS_YAW_STD_MIN, QNH_SMOOTH_ALPHA,
    FALLBACK_REF_LAT, FALLBACK_REF_LON, FALLBACK_REF_ALT,
    DEG2RAD, KNOTS_TO_MS, _sigterm_handler,
)
from lis2mdl_find_placement import LIS2MDL, UT_PER_LSB as MAG_UT_PER_LSB

# ── Noise densities ──────────────────────────────────────────────────
# MEASURED, not guessed. Allan variance over 20260901_132235: 2.70 h
# stationary indoors, first 30 min discarded as thermal warm-up, leaving
# 2.20 h at a measured 51.654 Hz (not the 100 Hz the loop nominally
# targets -- allan.py needs the real rate, and assuming the nominal one
# would have scaled every result).
#
# How wrong the previous guesses were:
#
#     parameter     guessed      measured    ratio
#     sig_accel   3.9227e-03   6.8272e-03    1.74x
#     sig_gyro    5.0000e-04   4.9586e-04    0.99x
#     sig_ba_rw   1.0000e-04   3.9666e-04    3.97x
#     sig_bg_rw   1.0000e-06   3.5977e-05   35.98x
#
# sig_gyro was right to within 1%. sig_bg_rw was 36x too SMALL, i.e. the
# filter was far too confident that gyro bias holds still -- and the same
# log shows bg_x drifting 0.21 deg/s over 2.7 h, exactly the motion a
# 36x-too-tight value forbids. Replayed on 20260901_004512 the measured
# set moves GPS NIS 4.76 -> 5.60 (expect ~6) and mag NIS 1.38 -> 0.93
# (expect ~1): both closer to consistent, which is the point of measuring.
SIG_ACCEL = 6.8272e-03         # m/s^2/sqrt(Hz)   (696 ug/sqrt(Hz))
SIG_GYRO  = 4.9586e-04         # rad/s/sqrt(Hz)   (1.70 deg/sqrt(hr))
SIG_BA_RW = 3.9666e-04         # m/s^3/sqrt(Hz)
# gyro_z's rate-random-walk fit is NOT trustworthy: log-log slope -0.02
# where +0.50 is wanted, and tau_min = 2537 s against a 7920 s log (ratio
# 3.1, below the factor-of-5 guideline). Averaging it in drags the result
# down to 3.5977e-05. This is the mean of gyro_x and gyro_y alone, whose
# slopes (+0.47, +0.42) are sound. Erring toward the larger value is also
# the safer direction -- an under-sized Q is what caused the problem.
SIG_BG_RW = 5.1150e-05         # rad/s^2/sqrt(Hz)

# Samples averaged for the startup static alignment.
SEED_SAMPLES = 200
SEED_DT      = 0.01
# Attempts allowed before falling back to level + zero bias. Retrying is
# cheap (2 s each) and the alternative is a whole run wasted -- see the
# fallback branch in main() for what a failed alignment actually costs.
SEED_RETRIES = 5

# ── Barometer rate limit ─────────────────────────────────────────────
# Largest pressure CHANGE accepted between consecutive samples.
#
# isa_alt()'s absolute band (300-1100 hPa) is far too wide to catch this
# hardware's corruption: readings of 873 and 1300 hPa both sit inside it,
# pass silently, and land in the altitude solution as errors of ~1000 m
# and ~2500 m respectively. That is what drove the observed +308 m /
# -154 m vertical swings -- not a filter fault, corrupted input that
# looked legitimate.
#
# A rate limit catches what an absolute band cannot, because physics
# bounds how fast pressure can actually move:
#   * weather   -- a few hPa per HOUR
#   * altitude  -- 1 hPa per ~8 m climbed
# At BARO_PERIOD = 0.125 s, 5 hPa between samples corresponds to a
# 40 m step, i.e. a 320 m/s climb rate. Nothing this vehicle does comes
# close, so anything larger is corruption, not signal.
BARO_MAX_DELTA_HPA = 5.0
# Consecutive rate rejections before the baseline is re-established.
# Without this, one corrupted sample accepted as the reference would
# reject every subsequent GOOD reading forever -- the filter would go
# permanently blind rather than lose one sample.
BARO_RESYNC_AFTER = 20

# ── GPS velocity sigma ───────────────────────────────────────────────
# MEASURED, not assumed. On 20260901_174748 the receiver reported
# 1.05-1.15 m/s of horizontal speed during stretches where the vehicle
# was demonstrably stationary -- truth was exactly 0.00, so that IS the
# error. The previous 0.5 claimed twice the accuracy the receiver has.
#
# Swept against that same log:
#
#     vel_std   posNIS   velNIS   statRMS   spikes>4m
#               (~3)     (~2)
#       0.50     2.86     7.79     2.53 m       18
#       1.10     3.11     3.19     2.12 m       97
#       2.00     4.78     2.16     2.15 m      201
#
# 1.10 minimises TRUE error (stationary rms, the only place truth is
# known) and puts position NIS at 3.11 against a target of 3. At 2.00 the
# velocity NIS finally hits its target but position NIS degrades to 4.78 --
# the filter starts under-trusting GPS overall, and true error stops
# improving. That is past the optimum.
#
# The cost is a rougher trajectory: loosening R accepts more of the noisy
# GPS velocity, and each accepted fix injects that noise into the state.
# Accepted deliberately, because stationary rms is measurable and the
# spike count is not scoreable without truth during motion.
#
# Worth revisiting with a speed-dependent value: 1.1 m/s is the error at
# REST, and Doppler-derived velocity is usually better once genuinely
# moving. One constant is a compromise between two regimes.
GPS_VEL_STD = 1.10             # m/s

# ZUPT is attempted every Nth IMU iteration. The detector still has the
# final say; this only bounds how often it is asked.
ZUPT_EVERY = 10


def collect_static_window(imu, n=SEED_SAMPLES, dt=SEED_DT):
    """Gather a stationary IMU window for seed_static_alignment()."""
    acc, gyr = [], []
    for _ in range(n):
        a, g = read_imu(imu)
        acc.append(a)
        gyr.append(g)
        time.sleep(dt)
    return np.array(acc), np.array(gyr)


def main():
    signal.signal(signal.SIGTERM, _sigterm_handler)
    no_gps = "--no-gps" in sys.argv

    imu  = mpu6050(0x68)
    baro = BMP180()
    ser  = None if no_gps else serial.Serial(GPS_PORT, GPS_BAUD, timeout=0)
    files, (w_imu, w_gps, w_baro, w_mag, w_est) = open_logs()

    if load_accel_calibration():
        print(f"Accel calibration loaded.")
    else:
        print("WARNING: no accel_calibration.json -- gravity will read low "
              "and that error integrates into velocity.")

    # ── Magnetometer ─────────────────────────────────────────────────
    mag = None
    mag_cal = None
    try:
        mag = LIS2MDL()
        mag_cal = load_mag_calibration()
        print(f"Magnetometer at rest (raw LSB): {mag.read_raw()}")
        if mag_cal is None:
            print("  WARNING: no lis2mdl_calibration.json -- heading will "
                  "carry a heading-dependent hard-iron error.")
    except Exception as e:
        print(f"  Magnetometer unavailable ({e.__class__.__name__}): {e}")
        mag = None

    # ── Static alignment ─────────────────────────────────────────────
    # Must happen BEFORE the filter is built: bg is seeded into NomState,
    # and eskf.py's ZUPT detector depends on it being non-zero.
    # Retry rather than giving up after one window. Under systemd there is
    # no console to time the start against, so the alignment lands whenever
    # the service happens to come up -- often while the unit is still being
    # set down. Run 20260901_173820 failed exactly that way and the whole
    # 4-minute run was worthless.
    bg0 = np.zeros(3)
    roll0 = pitch0 = 0.0
    seed_ok = False
    for attempt in range(1, SEED_RETRIES + 1):
        print(f"Static alignment {attempt}/{SEED_RETRIES} -- keep the vehicle "
              f"still ({SEED_SAMPLES * SEED_DT:.1f} s)...")
        acc_w, gyr_w = collect_static_window(imu)
        bg0, roll0, pitch0, seed_ok = seed_static_alignment(acc_w, gyr_w)
        if seed_ok:
            print(f"  seeded: bg = {np.round(np.rad2deg(bg0), 3)} deg/s, "
                  f"roll/pitch = {math.degrees(roll0):.2f}/"
                  f"{math.degrees(pitch0):.2f} deg")
            break
        print(f"  not stationary (gyro spread or |a| off) -- retrying")
        time.sleep(1.0)

    if not seed_ok:
        # Keep the measured roll/pitch, drop only the bias.
        #
        # Not obvious, and worth stating: a NOISY gravity average is still a
        # roughly correct attitude. On run 20260901_173820 the box was
        # standing on its end and the window gave pitch = 79.9 deg -- which
        # was RIGHT. Forcing "level" there would have asserted 0 deg against
        # a true 80 and been far worse than the noisy estimate.
        #
        # The gyro mean is different. It is the bias only if the unit was
        # still; measured during motion it is bias plus whatever rotation
        # occurred, and there is no way to separate them. So attitude is
        # kept and bias is discarded.
        #
        # bg = 0 is still bad -- 6.1 deg/s of uncorrected bias integrates
        # straight into attitude, ZUPT never fires, and on 173820 velocity
        # passed 1 m/s within 1.5 s and the run was worthless. Hence the
        # retries above and the noise here.
        bg0 = np.zeros(3)
        print(f"\n  *** STATIC ALIGNMENT FAILED after {SEED_RETRIES} "
              f"attempts ***")
        print(f"  Attitude kept from the last window (roll/pitch "
              f"{math.degrees(roll0):.1f}/{math.degrees(pitch0):.1f} deg) --")
        print("  a noisy gravity average is still roughly right. GYRO BIAS is")
        print("  ZERO, which is not: ZUPT cannot fire, and ~6 deg/s will")
        print("  integrate into attitude from the first sample. Stop, set it")
        print("  on a still surface, and restart before trusting this run.\n")

    p0, t0 = baro.read()
    print(f"Baro at rest: {p0:.2f} hPa, {t0:.1f} C")

    ref_lat = FALLBACK_REF_LAT * DEG2RAD
    ref_lon = FALLBACK_REF_LON * DEG2RAD
    ref_alt = FALLBACK_REF_ALT
    gps_origin_set = False

    try:
        qnh_offset = ref_alt - isa_alt(p0, t0 + 273.15)
    except ValueError as e:
        print(f"WARNING: bad initial baro reading ({e}); qnh_offset = 0")
        qnh_offset = 0.0

    nom = NomState(p=np.zeros(3), v=np.zeros(3),
                   q=euler_to_quat(roll0, pitch0, 0.0),
                   ba=np.zeros(3), bg=bg0)
    ekf = ESKF(nom, make_P0(), ref_lat=ref_lat,
               sig_accel=SIG_ACCEL, sig_gyro=SIG_GYRO,
               sig_ba_rw=SIG_BA_RW, sig_bg_rw=SIG_BG_RW)

    last_vel_ned = np.zeros(3)
    have_vel = False
    n_gps = n_gps_att = n_baro = n_baro_att = 0
    n_yaw = n_yaw_att = n_mag = n_mag_att = 0
    mag_consec_fail = 0
    gps_serial_fail = 0
    baro_rate_rejects = 0
    last_baro_sample = None
    last_yaw_meas = None
    buf = ""

    # time.monotonic(), not time.time(): eskf.py rejects measurements that
    # go backwards, and NTP can step the wall clock mid-run on a Pi.
    last_t = time.monotonic()
    t_start = last_t
    last_baro_t = last_mag_t = last_print = last_est = 0.0
    step = 0

    print(f"\nFusion running (eskf.py"
          + ("" if no_gps else " + GPS") + (" + Mag" if mag else "")
          + "). Ctrl+C to stop.\n")

    try:
        while True:
            now = time.monotonic()
            dt = now - last_t
            last_t = now
            step += 1

            accel, gyro = read_imu(imu)
            ekf.predict(IMUMeas(dt=dt, accel=accel, gyro=gyro))
            w_imu.writerow([f"{now:.4f}", f"{dt:.5f}",
                            *[f"{v:.5f}" for v in accel],
                            *[f"{v:.6f}" for v in gyro]])

            # ── ZUPT + accel-bias prior ──────────────────────────────
            if step % ZUPT_EVERY == 0:
                ekf.update_zupt(t=now)
                # Only while the detector says we are stopped. At rest a
                # tilt error and a horizontal accel bias are
                # indistinguishable, and ZUPT constrains their SUM while
                # saying nothing about the split -- so the estimate drifts
                # along that null space with no visible symptom, because
                # ZUPT is holding velocity at ~0.01 m/s the whole time.
                #
                # Measured on 20260901_163533: 8 stationary minutes put
                # 0.613 m/s^2 into ba, matching g*sin(3.49 deg) of absorbed
                # tilt to 3%. The walk that followed released it at
                # 0.61 m/s per second and the filter ran to 127 m/s.
                #
                # In motion the bias is genuinely observable, so the prior
                # stands aside. See ESKF.update_accel_bias_prior.
                if ekf.is_static:
                    ekf.update_accel_bias_prior(t=now)

            # ── Baro ─────────────────────────────────────────────────
            if now - last_baro_t >= BARO_PERIOD:
                last_baro_t = now
                p_hpa, t_c = baro.read()

                # Rate check BEFORE the absolute one -- see
                # BARO_MAX_DELTA_HPA. Catches corruption that lands inside
                # the plausible band, which the absolute test cannot.
                rate_bad = False
                if last_baro_sample is not None:
                    d = abs(p_hpa - last_baro_sample[0])
                    if d > BARO_MAX_DELTA_HPA:
                        baro_rate_rejects += 1
                        rate_bad = True
                        if baro_rate_rejects >= BARO_RESYNC_AFTER:
                            # Persistent disagreement means the REFERENCE is
                            # probably the corrupt one, not the stream.
                            # Re-baseline rather than stay blind.
                            print(f"\nBaro: {baro_rate_rejects} consecutive "
                                  f"rate rejections -- re-baselining to "
                                  f"{p_hpa:.2f} hPa.")
                            last_baro_sample = (p_hpa, t_c)
                            baro_rate_rejects = 0
                        elif baro_rate_rejects <= 3 or baro_rate_rejects % 25 == 0:
                            print(f"\nBaro jumped {d:.1f} hPa in one sample "
                                  f"({p_hpa:.2f}) -- rejected as corrupt "
                                  f"({baro_rate_rejects} so far)")
                        n_baro_att += 1

                if not rate_bad:
                    baro_rate_rejects = 0
                try:
                    isa_val = None if rate_bad else isa_alt(p_hpa, t_c + 273.15)
                except ValueError as e:
                    print(f"\nBad baro reading skipped: {e}")
                    n_baro_att += 1
                    isa_val = None
                if isa_val is not None:
                    last_baro_sample = (p_hpa, t_c)
                    alt_ref = isa_val + qnh_offset - ref_alt
                    ok = ekf.update_baro(
                        BaroMeas(altitude=alt_ref, std=BARO_STD), t=now)
                    n_baro_att += 1
                    n_baro += int(ok)
                    w_baro.writerow([f"{now:.4f}", f"{p_hpa:.2f}",
                                     f"{t_c:.1f}", f"{isa_val:.2f}",
                                     f"{qnh_offset:+.2f}", f"{alt_ref:.2f}",
                                     int(ok)])

            # ── Magnetometer ─────────────────────────────────────────
            if mag is not None and now - last_mag_t >= MAG_PERIOD:
                last_mag_t = now
                try:
                    if mag.data_ready():
                        m_raw = np.array(mag.read_raw(), dtype=float)
                        mag_consec_fail = 0
                        if mag_cal is not None:
                            off, scl = mag_cal
                            m_raw = (m_raw - off) * scl

                        # |B| in MICROTESLA for the disturbance check --
                        # eskf.py's sanity band is 20-80 uT and these are
                        # LSB counts until scaled.
                        b_ut = float(np.linalg.norm(m_raw)) * MAG_UT_PER_LSB

                        m_body = mag_axis_map(m_raw)
                        r_now, p_now, _ = quat_to_euler(ekf.nom.q)
                        yaw_true = (tilt_compensated_heading(m_body, r_now,
                                                             p_now)
                                    + MAG_DECLINATION)
                        ok = ekf.update_mag(
                            MagMeas(yaw=yaw_true, std=MAG_YAW_STD,
                                    field_norm=b_ut), t=now)
                        n_mag_att += 1
                        n_mag += int(ok)
                        w_mag.writerow([f"{now:.4f}",
                                        *[f"{v:.1f}" for v in m_raw],
                                        f"{yaw_true / DEG2RAD:.2f}",
                                        f"{quat_to_euler(ekf.nom.q)[2] / DEG2RAD:.2f}",
                                        int(ok)])
                except OSError:
                    mag_consec_fail += 1
                    n_mag_att += 1
                    if mag_consec_fail >= MAG_MAX_CONSEC_FAIL:
                        print(f"\nMagnetometer failed {mag_consec_fail} reads "
                              f"in a row -- disabling for this run.")
                        mag = None

            # ── GPS ──────────────────────────────────────────────────
            try:
                data = ser.read(2048) if ser is not None else b""
            except serial.SerialException as e:
                data = b""
                gps_serial_fail += 1
                if gps_serial_fail == 1 or gps_serial_fail % 50 == 0:
                    print(f"\nGPS serial read failed "
                          f"({gps_serial_fail}x): {e}")
                time.sleep(0.05)
            if data:
                buf += data.decode('ascii', errors='replace')

            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()

                if line.startswith(('$GPRMC', '$GNRMC')):
                    try:
                        msg = pynmea2.parse(line)
                        if (msg.status == 'A'
                                and msg.spd_over_grnd is not None
                                and msg.true_course is not None):
                            spd = float(msg.spd_over_grnd) * KNOTS_TO_MS
                            crs = float(msg.true_course) * DEG2RAD
                            # vD is NOT measured by RMC -- flagged via
                            # vel_valid at the update, never fused as 0.
                            last_vel_ned = np.array(
                                [spd * math.cos(crs), spd * math.sin(crs), 0.0])
                            have_vel = True

                            if spd >= GPS_YAW_MIN_SPEED:
                                yaw_std = max(
                                    math.atan2(GPS_VEL_ACCURACY, spd),
                                    GPS_YAW_STD_MIN)
                                # No field_norm: this is a course-over-ground
                                # heading, not a magnetic reading, so the
                                # disturbance check must not see it.
                                ok = ekf.update_mag(
                                    MagMeas(yaw=crs, std=yaw_std), t=now)
                                n_yaw_att += 1
                                n_yaw += int(ok)
                                last_yaw_meas = (crs, yaw_std, int(ok))
                    except (pynmea2.ParseError, ValueError, TypeError):
                        pass

                elif line.startswith(('$GPGGA', '$GNGGA')):
                    try:
                        msg = pynmea2.parse(line)
                        if not (msg.gps_qual and int(msg.gps_qual) > 0):
                            continue
                        hdop = (float(msg.horizontal_dil)
                                if msg.horizontal_dil else 2.0)
                        pos_std = float(np.clip(GPS_POS_STD_BASE * hdop,
                                                GPS_POS_STD_MIN,
                                                GPS_POS_STD_MAX))
                        vel = last_vel_ned if have_vel else np.zeros(3)

                        if not gps_origin_set:
                            # First fix re-anchors the frame. Attitude and
                            # bias are KEPT -- eskf.py has its own recovery
                            # paths, and the seeded bias is better than any
                            # re-guess.
                            ref_lat = msg.latitude * DEG2RAD
                            ref_lon = msg.longitude * DEG2RAD
                            ref_alt = float(msg.altitude)
                            ekf.ref_lat = ref_lat
                            ekf.nom.p = np.zeros(3)
                            ekf.nom.v = vel.copy()
                            ekf.P[0:3, 0:3] = np.eye(3) * pos_std ** 2
                            ekf.P[3:6, 3:6] = np.eye(3) * 0.5 ** 2
                            if last_baro_sample is not None:
                                pb, tb = last_baro_sample
                                try:
                                    qnh_offset = ref_alt - isa_alt(pb, tb + 273.15)
                                except ValueError:
                                    pass
                            gps_origin_set = True
                            print(f"\nGPS ACQUIRED after {now - t_start:.1f}s "
                                  f"-- {msg.latitude:.6f}, {msg.longitude:.6f}, "
                                  f"{ref_alt:.1f}m (HDOP {hdop:.2f})")
                            n_gps_att += 1
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
                        vel_valid = ((True, True, False) if have_vel
                                     else (False, False, False))
                        ok = ekf.update_gps(GPSMeas(
                            position=pos_ned, velocity=vel,
                            pos_std_h=pos_std,
                            pos_std_v=pos_std * GPS_POS_STD_V_RATIO,
                            vel_std=GPS_VEL_STD,
                            vel_valid=vel_valid), t=now)
                        n_gps_att += 1
                        n_gps += int(ok)

                        # QNH eased toward the fix, and only on ACCEPTED
                        # ones -- a rejected fix is exactly the altitude not
                        # to anchor the vertical reference to.
                        if ok and last_baro_sample is not None:
                            pb, tb = last_baro_sample
                            try:
                                target = (float(msg.altitude)
                                          - isa_alt(pb, tb + 273.15))
                            except ValueError:
                                pass
                            else:
                                qnh_offset += QNH_SMOOTH_ALPHA * (
                                    target - qnh_offset)

                        if last_yaw_meas is None:
                            yaw_cols = ["", "", ""]
                        else:
                            c, s, yok = last_yaw_meas
                            yaw_cols = [f"{c / DEG2RAD:.1f}",
                                        f"{s / DEG2RAD:.2f}", yok]
                        w_gps.writerow([f"{now:.4f}",
                                        f"{msg.latitude:.7f}",
                                        f"{msg.longitude:.7f}",
                                        f"{float(msg.altitude):.2f}",
                                        f"{vel[0]:.3f}", f"{vel[1]:.3f}",
                                        f"{hdop:.2f}", int(ok)] + yaw_cols)
                    except (pynmea2.ParseError, ValueError, TypeError):
                        pass

            # ── Logging / status ─────────────────────────────────────
            if now - last_est >= EST_LOG_PERIOD:
                last_est = now
                lat, lon, alt = ned_to_lla(ekf.nom.p, ref_lat, ref_lon, ref_alt)
                rpy = ekf.euler_deg
                w_est.writerow([f"{now:.4f}",
                                *[f"{v:.3f}" for v in ekf.nom.p],
                                *[f"{v:.3f}" for v in ekf.nom.v],
                                *[f"{v:.2f}" for v in rpy],
                                f"{lat/DEG2RAD:.7f}", f"{lon/DEG2RAD:.7f}",
                                f"{alt:.2f}", n_gps, n_gps_att,
                                n_baro, n_baro_att, n_yaw, n_yaw_att,
                                n_mag, n_mag_att,
                                *[f"{v:.6f}" for v in ekf.nom.ba],
                                *[f"{v:.7f}" for v in ekf.nom.bg]])

                if now - last_print >= PRINT_PERIOD:
                    last_print = now
                    tag = "" if gps_origin_set else "[NO GPS - DR] "
                    print(f"{tag}p=({ekf.nom.p[0]:7.2f},{ekf.nom.p[1]:7.2f},"
                          f"{ekf.nom.p[2]:6.2f})m "
                          f"v=({ekf.nom.v[0]:5.2f},{ekf.nom.v[1]:5.2f},"
                          f"{ekf.nom.v[2]:5.2f})m/s "
                          f"yaw={rpy[2]:6.1f} "
                          f"gps={n_gps}/{n_gps_att} baro={n_baro}/{n_baro_att} "
                          f"mag={n_mag}/{n_mag_att} "
                          f"zupt={ekf.zupt_count}{'*' if ekf.is_static else ''} "
                          f"bg={np.round(np.rad2deg(ekf.nom.bg), 2)}")

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping, closing log files...")
    finally:
        for f in files:
            f.close()
        # Diagnostics that only mean something at the end of a run.
        print(f"\n  ZUPT applied      : {ekf.zupt_count}")
        print(f"  gate recoveries   : {ekf.gate_recoveries}")
        print(f"  gate rejections   : {ekf.gated_count}")
        print(f"  dt clamped        : {ekf.dt_clamped}"
              + (f" (worst {ekf.dt_worst:+.4f}s)" if ekf.dt_clamped else ""))
        print(f"  stale rejects     : {ekf.stale_rejects}")
        print(f"  mag disturbed     : {ekf.mag_disturbed}"
              + ("  (baseline never learned)" if ekf.mag_learn_failed else
                 f"  (baseline {ekf.mag_field_expected:.1f} uT)"
                 if ekf.mag_field_expected else ""))
        if ekf.nonfinite_events:
            print(f"  non-finite events : {len(ekf.nonfinite_events)}")
        if ekf.frozen:
            print("  *** FILTER FROZE -- output past that point is stale ***")
        h = ekf.check_P_health()
        print(f"  P health          : {'ok' if h['ok'] else h['reason']}"
              f"  cond={h['cond']:.2e}")
        print(f"  final gyro bias   : {np.round(np.rad2deg(ekf.nom.bg), 4)} deg/s")
        print("Logs saved.")


if __name__ == "__main__":
    main()
