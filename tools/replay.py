"""
Offline replay -- run the ESKF against logged CSVs instead of live sensors.
==========================================================================

Why this exists
---------------
Without it, testing a tuning change means going outside, running the
hardware, coming back, changing one number, going outside again. With it,
you collect a log once and re-run the whole filter over it in seconds.

That turns tuning from a guess into a search: sweep a parameter across a
decade, run every value against the SAME log, plot NIS or steady-state
error against it. It also makes bugs reproducible -- re-run the exact data
that produced a failure, with a fix applied, and see immediately whether
it fixed it.

What it cannot do: replay reproduces what was RECORDED. If the bug is in
the driver -- a misread register, a wrong axis map, I2C corruption --
replay reproduces the corrupted data faithfully and tells you nothing
about its origin. Hardware-level checks stay necessary.

Design notes
------------
This file does all the I/O; eskf.py does none. That is deliberate: the
same unmodified eskf.py runs here and on the Pi, so replay validates what
actually flies rather than a modified copy.

Baro is replayed from RAW PRESSURE, not from the pre-computed
alt_above_ref_m column, even though the latter is simpler and would
reproduce the live run exactly. The QNH handling is precisely the part
that had a serious bug (hard re-anchoring to noisy GPS altitude), so it
has to be changeable and re-testable against old logs. The logged column
is still read, as a cross-check.

Usage
-----
    python replay.py ~/dronepi-project/logs/20260829_143000
    python replay.py <prefix> --no-gps-after 100 --gps-back-at 160
    python replay.py <prefix> --sweep sig_ba_rw 1e-7 1e-4 8
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

from eskf import (
    ESKF, NomState, IMUMeas, GPSMeas, BaroMeas, MagMeas,
    make_P0, quat_to_euler, seed_static_alignment,
    quat_from_rotvec, quat_mul, quat_normalize, G_MAG,
)

DEG2RAD = math.pi / 180.0

# ISA standard atmosphere, troposphere (0-11 km)
_ISA_T0  = 288.15
_ISA_L   = 0.0065
_ISA_P0  = 1013.25
_ISA_EXP = 287.058 * _ISA_L / 9.80665

# Fraction of the GPS-vs-ISA altitude discrepancy folded into qnh_offset
# per accepted fix. NOT a hard replacement: GPS vertical noise is large
# (tens of metres of spread is normal on a bare receiver), and injecting
# that into the barometer's reference makes every baro measurement look
# like an outlier to the chi2 gate -- destroying the vertical channel the
# barometer exists to provide. Smoothing tracks genuine pressure drift
# (slow) while averaging out GPS altitude noise (fast).
QNH_SMOOTH_ALPHA = 0.03

# IMU samples used for the startup static alignment (see seed_static_alignment).
SEED_SAMPLES = 200


def isa_alt(p_hpa: float, t_k: float) -> float:
    """Pressure + temperature -> geometric altitude, ISA with a
    temperature correction. Raises on implausible pressure so a corrupted
    reading fails here rather than silently returning a complex number."""
    if not (300.0 <= p_hpa <= 1100.0):
        raise ValueError(f"implausible pressure: {p_hpa} hPa")
    h_isa = (_ISA_T0 / _ISA_L) * (1.0 - (p_hpa / _ISA_P0) ** _ISA_EXP)
    t_isa = _ISA_T0 - _ISA_L * max(h_isa, 0.0)
    return h_isa * (t_k / t_isa)


def lla_to_ned(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """Geodetic -> local NED metres. Flat-Earth approximation, valid near
    the reference (which is why the reference is taken from the log's own
    first fix rather than a fixed constant)."""
    R_0, E2 = 6378137.0, 0.00669437999014
    s2 = math.sin(ref_lat) ** 2
    denom = 1.0 - E2 * s2
    R_N = R_0 * (1.0 - E2) / denom ** 1.5
    R_E = R_0 / math.sqrt(denom)
    return np.array([
        (lat - ref_lat) * (R_N + ref_alt),
        (lon - ref_lon) * (R_E + ref_alt) * math.cos(ref_lat),
        -(alt - ref_alt),
    ])


# ===============================================================
#  LOG READING
# ===============================================================

def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(row, key, default=float('nan')):
    """Float from a CSV field, tolerating blanks and junk. Logs from an
    interrupted run routinely have a truncated final line."""
    try:
        v = row.get(key, '')
        return float(v) if v not in ('', None) else default
    except (TypeError, ValueError):
        return default


@dataclass
class Event:
    t: float
    kind: str      # 'imu' | 'gps' | 'baro' | 'mag'
    row: dict


def load_events(prefix: str):
    """Read all five CSVs and merge into one time-ordered event stream.

    Merging by timestamp reproduces the nearest-neighbour scheduling the
    live loop used: the IMU is the master clock and the other sensors are
    folded in whenever they were ready.
    """
    events = []
    counts = {}
    for kind, suffix in (('imu', '_imu.csv'), ('gps', '_gps.csv'),
                         ('baro', '_baro.csv'), ('mag', '_mag.csv')):
        rows = _read_csv(prefix + suffix)
        counts[kind] = len(rows)
        for r in rows:
            t = _f(r, 't')
            if math.isfinite(t):
                events.append(Event(t, kind, r))

    events.sort(key=lambda e: e.t)
    return events, counts


# ===============================================================
#  REPLAY
# ===============================================================

def replay(prefix, sig_accel, sig_gyro, sig_ba_rw, sig_bg_rw,
           gps_std_base=5.3, gps_std_v_ratio=1.95, gps_vel_std=1.10, baro_std=5.0,
           mag_std_deg=5.0, mag_ut_per_lsb=0.15, use_earth_rate=True,
           no_gps_after=None, gps_back_at=None, zupt=True, ba_prior=True, verbose=True):
    """Run the filter over a logged run. Returns a dict of results."""

    events, counts = load_events(prefix)
    if not events:
        sys.exit(f"no events found -- check the prefix: {prefix}")

    if verbose:
        t0, t1 = events[0].t, events[-1].t
        print(f"\n{os.path.basename(prefix)}: {t1 - t0:.1f} s, "
              f"{len(events):,} events")
        print("  " + "  ".join(f"{k}={v:,}" for k, v in counts.items()))

    # Reference frame anchored to the log's own first valid fix, so the
    # flat-Earth conversion is used near where it is valid.
    ref = None
    for e in events:
        if e.kind == 'gps':
            lat, lon, alt = (_f(e.row, 'lat_deg'), _f(e.row, 'lon_deg'),
                             _f(e.row, 'alt_m'))
            if all(map(math.isfinite, (lat, lon, alt))):
                ref = (lat * DEG2RAD, lon * DEG2RAD, alt)
                break
    if ref is None:
        sys.exit("no valid GPS fix in the log -- cannot anchor the NED frame")
    ref_lat, ref_lon, ref_alt = ref

    # Seed gyro bias and level attitude from the opening IMU samples.
    # Without this ZUPT cannot bootstrap: its detector tests |gyro - bg|,
    # and with bg = 0 a consumer MEMS bias exceeds the threshold forever,
    # so ZUPT never fires and never observes the bias that blocks it.
    # (A run WITH GPS hides this, because GPS converges bias the slow way.)
    seed_rows = [e for e in events if e.kind == 'imu'][:SEED_SAMPLES]
    seed_acc = [[_f(r.row, 'ax'), _f(r.row, 'ay'), _f(r.row, 'az')]
                for r in seed_rows]
    seed_gyr = [[_f(r.row, 'gx'), _f(r.row, 'gy'), _f(r.row, 'gz')]
                for r in seed_rows]
    nom = NomState()
    if len(seed_acc) >= 10 and np.all(np.isfinite(seed_acc)) \
            and np.all(np.isfinite(seed_gyr)):
        bg0, roll0, pitch0, seed_ok = seed_static_alignment(seed_acc, seed_gyr)
        if seed_ok:
            nom.bg = bg0
            nom.q = quat_normalize(quat_mul(
                quat_from_rotvec(np.array([0.0, 0.0, 0.0])),
                quat_from_rotvec(np.array([roll0, 0.0, 0.0]))))
            nom.q = quat_normalize(quat_mul(
                nom.q, quat_from_rotvec(np.array([0.0, pitch0, 0.0]))))
            if verbose:
                print(f"  seeded from first {len(seed_acc)} IMU samples: "
                      f"bg = {np.round(np.rad2deg(bg0), 3)} deg/s, "
                      f"roll/pitch = {np.rad2deg(roll0):.2f}/"
                      f"{np.rad2deg(pitch0):.2f} deg")
        elif verbose:
            print("  WARNING: opening samples do not look stationary -- not")
            print("  seeding. ZUPT will not fire until another sensor has")
            print("  converged the gyro bias.")

    ekf = ESKF(nom, make_P0(), ref_lat=ref_lat,
               sig_accel=sig_accel, sig_gyro=sig_gyro,
               sig_ba_rw=sig_ba_rw, sig_bg_rw=sig_bg_rw,
               use_earth_rate=use_earth_rate)

    qnh_offset = 0.0
    last_baro_sample = None
    history = []
    n = {k: [0, 0] for k in ('gps', 'baro', 'mag')}   # [accepted, attempted]
    t_start = events[0].t
    gps_suppressed = 0

    for e in events:
        rel_t = e.t - t_start

        if e.kind == 'imu':
            dt = _f(e.row, 'dt')
            acc = np.array([_f(e.row, 'ax'), _f(e.row, 'ay'), _f(e.row, 'az')])
            gyr = np.array([_f(e.row, 'gx'), _f(e.row, 'gy'), _f(e.row, 'gz')])
            # dt clamping and NaN handling are the filter's job, not the
            # harness's -- pass the logged values through unmodified so
            # replay exercises those code paths exactly as the live run did.
            ekf.predict(IMUMeas(dt=dt, accel=acc, gyro=gyr))
            # ZUPT at 10 Hz. The detector inside the filter decides whether
            # it actually applies, so calling unconditionally is safe.
            if zupt and ekf.step_count % 10 == 0:
                ekf.update_zupt(t=e.t)
                # Accel-bias prior, applied only while the detector says we
                # are stopped. That is exactly when tilt and accel bias are
                # indistinguishable and the estimate is free to wander into
                # the null space between them -- see
                # ESKF.update_accel_bias_prior. In motion the bias becomes
                # genuinely observable, so the prior stands aside.
                if ba_prior and ekf.is_static:
                    ekf.update_accel_bias_prior(t=e.t)
            history.append((rel_t, ekf.position.copy(), ekf.velocity.copy(),
                            ekf.euler_deg.copy(), ekf.pos_1sigma.copy(),
                            ekf.accel_bias.copy(), ekf.gyro_bias.copy()))

        elif e.kind == 'gps':
            # Synthetic outage: drop fixes in a window to test coasting and
            # recovery without needing to physically cover an antenna at
            # the right moment during the original run.
            if (no_gps_after is not None and rel_t >= no_gps_after and
                    (gps_back_at is None or rel_t < gps_back_at)):
                gps_suppressed += 1
                continue

            lat, lon, alt = (_f(e.row, 'lat_deg'), _f(e.row, 'lon_deg'),
                             _f(e.row, 'alt_m'))
            if not all(map(math.isfinite, (lat, lon, alt))):
                continue
            hdop = _f(e.row, 'hdop', 2.0)
            if not math.isfinite(hdop):
                hdop = 2.0
            pos_std = float(np.clip(gps_std_base * hdop, 2.0, 50.0))

            # NMEA RMC carries speed-over-ground and course but no vertical
            # rate, so vD is not measured. Mark it invalid rather than
            # passing 0.0 -- fusing a fabricated zero vertical velocity at
            # vel_std once per fix fights any real climb and removes roughly
            # half the climb rate per update.
            vN, vE = _f(e.row, 'vN', 0.0), _f(e.row, 'vE', 0.0)
            have_vel = math.isfinite(vN) and math.isfinite(vE)
            vel = np.array([vN, vE, 0.0]) if have_vel else np.zeros(3)
            vel_valid = (have_vel, have_vel, False)

            pos_ned = lla_to_ned(lat * DEG2RAD, lon * DEG2RAD, alt,
                                 ref_lat, ref_lon, ref_alt)
            ok = ekf.update_gps(GPSMeas(
                position=pos_ned, velocity=vel,
                pos_std_h=pos_std, pos_std_v=pos_std * gps_std_v_ratio,
                vel_std=gps_vel_std, vel_valid=vel_valid), t=e.t)
            n['gps'][1] += 1
            n['gps'][0] += int(ok)

            # QNH re-anchoring, on ACCEPTED fixes only. A fix the gate just
            # rejected as an outlier is exactly the one whose altitude
            # should not be trusted to set the vertical reference.
            if ok and last_baro_sample is not None:
                p_hpa, t_c = last_baro_sample
                try:
                    target = alt - isa_alt(p_hpa, t_c + 273.15)
                except ValueError:
                    pass
                else:
                    qnh_offset += QNH_SMOOTH_ALPHA * (target - qnh_offset)

        elif e.kind == 'baro':
            p_hpa, t_c = _f(e.row, 'p_hpa'), _f(e.row, 't_c')
            if not (math.isfinite(p_hpa) and math.isfinite(t_c)):
                continue
            last_baro_sample = (p_hpa, t_c)
            try:
                isa_val = isa_alt(p_hpa, t_c + 273.15)
            except ValueError:
                continue
            alt_above_ref = isa_val + qnh_offset - ref_alt
            ok = ekf.update_baro(BaroMeas(altitude=alt_above_ref,
                                          std=baro_std), t=e.t)
            n['baro'][1] += 1
            n['baro'][0] += int(ok)

        elif e.kind == 'mag':
            yaw_deg = _f(e.row, 'yaw_mag_deg')
            if not math.isfinite(yaw_deg):
                continue
            # Field magnitude from the raw 3-axis reading, for the
            # disturbance check -- which needs MICROTESLA, because
            # MAG_FIELD_MIN/MAX in eskf.py are a sanity band on Earth's
            # actual field (20-80 uT).
            #
            # pi_live_nav_baro.py logs calibration-corrected LSB COUNTS, not
            # uT. Passing those straight through put the learned baseline at
            # ~230 "uT", outside the sanity band, so eskf.py concluded the
            # baseline was untrustworthy and disabled disturbance checking
            # for the entire run -- silently, since that path is a warning
            # rather than an error. Measured on 20260827_160750: 229.5 LSB
            # median, which at the LIS2MDL's 0.15 uT/LSB is 34.4 uT and sits
            # comfortably inside the band.
            #
            # mag_ut_per_lsb converts. Pass 1.0 if a logger already writes uT.
            mx, my, mz = (_f(e.row, 'mag_x'), _f(e.row, 'mag_y'),
                          _f(e.row, 'mag_z'))
            bn = (float(np.linalg.norm([mx, my, mz])) * mag_ut_per_lsb
                  if all(map(math.isfinite, (mx, my, mz))) else None)
            if bn is not None and bn <= 0:
                bn = None
            ok = ekf.update_mag(MagMeas(yaw=yaw_deg * DEG2RAD,
                                        std=mag_std_deg * DEG2RAD,
                                        field_norm=bn), t=e.t)
            n['mag'][1] += 1
            n['mag'][0] += int(ok)

    return dict(ekf=ekf, history=history, counts=n,
                gps_suppressed=gps_suppressed,
                ref=(ref_lat, ref_lon, ref_alt))


def report(res, verbose=True):
    """Print the diagnostics that actually tell you whether it worked."""
    ekf = res['ekf']
    n = res['counts']

    print("\n" + "=" * 66)
    print("RESULT")
    print("=" * 66)
    print(f"  final position   : {np.round(ekf.position, 2)} m (NED)")
    print(f"  final velocity   : {np.round(ekf.velocity, 3)} m/s")
    print(f"  final attitude   : {np.round(ekf.euler_deg, 2)} deg (r/p/y)")
    print(f"  position 1-sigma : {np.round(ekf.pos_1sigma, 2)} m")
    print(f"  accel bias       : {np.round(ekf.accel_bias, 4)} m/s^2")
    print(f"  gyro bias        : {np.round(np.rad2deg(ekf.gyro_bias), 4)} deg/s")

    print("\n  acceptance (accepted / attempted):")
    for k in ('gps', 'baro', 'mag'):
        a, t = n[k]
        pct = f"{100.0*a/t:5.1f}%" if t else "   n/a"
        print(f"    {k:5s} {a:6d} / {t:6d}   {pct}")
    if res['gps_suppressed']:
        print(f"    ({res['gps_suppressed']} GPS fixes suppressed by "
              f"the synthetic outage)")

    # NIS is the consistency check: it should average to the number of
    # measurement rows. Much lower means R is too conservative (the filter
    # is being told to expect more disagreement than it actually sees);
    # much higher means the filter is overconfident, or there are real
    # outliers in the data.
    print("\n  NIS (mean, vs expected):")
    # ZUPT included deliberately. It is a SYNTHETIC measurement, so its NIS
    # is the one number that says whether the stationarity detector is
    # telling the truth: a high ZUPT NIS means either the detector fired
    # while moving, or the velocity estimate is genuinely bad. Nothing else
    # in the report distinguishes those from a healthy run.
    for k, rows, series in (('gps', 6, ekf.nis_gps),
                            ('baro', 1, ekf.nis_baro),
                            ('mag', 1, ekf.nis_mag),
                            ('zupt', 3, ekf.nis_zupt)):
        if series:
            m = float(np.mean(series))
            if k == 'zupt':
                # ZUPT NIS is self-referential: the innovation is -v, and
                # ZUPT is what drives v toward zero. A working ZUPT
                # therefore produces tiny innovations by construction, so a
                # LOW value means it is doing its job -- not that R is too
                # conservative, which is what it would mean for a real
                # sensor.
                #
                # The MEAN is the wrong statistic here, and reporting it
                # alone produced a false alarm. On 20260827_160750 it read
                # 124 against an expected 3 -- but the median was 0.52 and
                # p90 was 5.0. Every one of the 245 high values fell in a
                # 50 s window starting inside a physical handling event and
                # decaying afterwards: ZUPT firing with a large innovation
                # because it was CORRECTING the velocity error that the
                # motion had just injected. That is the mechanism working,
                # not misfiring, and a mean-only verdict called it a fault.
                #
                # So: judge on the median, and treat a heavy tail as what it
                # is -- evidence of recovery events worth locating in time,
                # not evidence of a broken detector.
                med = float(np.median(series))
                tail = float(np.mean(np.asarray(series) > 10.0 * rows))
                verdict = ("working (low is expected here)" if med < rows
                           else "median HIGH -- detector may be firing "
                                "while moving")
                print(f"    {k:5s} {med:8.2f}  (median, expect <{rows})"
                      f"   {verdict}")
                if tail > 0.01:
                    print(f"          mean {m:.1f}, {100*tail:.1f}% of "
                          f"updates >{10*rows} -- likely recovery after "
                          f"motion; check when they occur before "
                          f"treating it as a fault")
                continue
            verdict = ("ok" if 0.5 * rows <= m <= 2.0 * rows else
                       "R too conservative" if m < 0.5 * rows else
                       "overconfident or outliers present")
            print(f"    {k:5s} {m:8.2f}  (expect ~{rows})   {verdict}")

    if ekf.gate_recoveries:
        print(f"\n  gate recoveries: {ekf.gate_recoveries}")
        print("    The escape hatch fired -- a sensor was rejected enough times")
        print("    in a row that P was inflated to let it back in. That is a")
        print("    divergence signal, not routine housekeeping: something drove")
        print("    the state far enough from the measurements that the gate")
        print("    locked it out. Check which sensor and what preceded it.")

    if ekf.zupt_count:
        print(f"\n  ZUPT applied {ekf.zupt_count} times "
              f"(currently static: {ekf.is_static})")
        print("    Note: at rest and level, horizontal accel bias and a small")
        print("    tilt error are indistinguishable -- ZUPT constrains velocity")
        print("    but cannot separate them. Expect ba_x/ba_y NOT to converge on")
        print("    a static log; that is observability, not a bug.")
    if ekf.mag_disturbed:
        print(f"\n  magnetometer: {ekf.mag_disturbed} readings rejected as "
              f"disturbed")
        print(f"    baseline |B| = {ekf.mag_field_expected}")
    if ekf.mag_learn_failed:
        print("\n  magnetometer: baseline never learned, disturbance check "
              "was inactive")

    h = ekf.check_P_health()
    print(f"\n  P health: cond={h['cond']:.3e}  min_eig={h['min_eig']:.3e}"
          f"  {'ok' if h['ok'] else 'FAILED: ' + h['reason']}")
    print("    A large condition number is normal -- P mixes metres with")
    print("    rad/s. Watch the trend across runs, not the absolute value.")

    tot_stale = sum(ekf.stale_rejects.values())
    if tot_stale:
        print(f"\n  out-of-order rejects: {ekf.stale_rejects}")
        print("    -- a measurement arrived older than one already fused from")
        print("       that same sensor. Check the parse order against the")
        print("       timestamp order in the raw log.")

    if ekf.dt_clamped:
        print(f"\n  dt clamped {ekf.dt_clamped} times, worst "
              f"{ekf.dt_worst:+.4f} s")
        print("    -- the logging loop was not keeping up; that is a finding")
        print("       about the system, not just a number to absorb.")
    if ekf.nonfinite_events:
        print(f"\n  non-finite rollbacks: {len(ekf.nonfinite_events)}")
        for ev in ekf.nonfinite_events:
            if ev[2] != 'rollback':
                print(f"    step {ev[0]}: {ev[1]} -> {ev[2].upper()}")
    if ekf.frozen:
        print("\n  *** FILTER FROZE -- output past that point is stale ***")


def plot(res, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    h = res['history']
    t = np.array([r[0] for r in h])
    pos = np.array([r[1] for r in h])
    vel = np.array([r[2] for r in h])
    eul = np.array([r[3] for r in h])
    sig = np.array([r[4] for r in h])
    ba = np.array([r[5] for r in h])
    bg = np.array([r[6] for r in h])

    fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    for i, lbl in enumerate('NED'):
        ax[0, 0].plot(t, pos[:, i], label=lbl)
        ax[1, 0].plot(t, vel[:, i], label=lbl)
        # +-2 sigma envelope: if the estimate is consistent, the true
        # error should sit inside this band ~95% of the time.
        ax[0, 1].plot(t, 2 * sig[:, i], label=f'2σ {lbl}')
    for i, lbl in enumerate(('roll', 'pitch', 'yaw')):
        ax[1, 1].plot(t, eul[:, i], label=lbl)
    for i, lbl in enumerate('xyz'):
        ax[2, 0].plot(t, ba[:, i], label=lbl)
        ax[2, 1].plot(t, np.rad2deg(bg[:, i]), label=lbl)

    for a, ttl, yl in ((ax[0, 0], 'Position', 'm'),
                       (ax[0, 1], 'Position 2σ', 'm'),
                       (ax[1, 0], 'Velocity', 'm/s'),
                       (ax[1, 1], 'Attitude', 'deg'),
                       (ax[2, 0], 'Accel bias', 'm/s²'),
                       (ax[2, 1], 'Gyro bias', 'deg/s')):
        a.set_title(ttl); a.set_ylabel(yl)
        a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[2, 0].set_xlabel('t (s)'); ax[2, 1].set_xlabel('t (s)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\n  plot -> {path}")
    print("    Bias panels are the ones to check first: if those states sit")
    print("    frozen at their initial values, the filter looks healthy on")
    print("    position while doing none of the work it exists to do.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prefix', help='log prefix, e.g. logs/20260829_143000')
    # 400 ug/sqrt(Hz), MPU6050-class. (300e-6 * G_MAG already converts
    # ug/sqrt(Hz) to m/s^2/sqrt(Hz) -- do not scale by g a second time.)
    ap.add_argument('--sig-accel', type=float, default=400e-6 * G_MAG)
    # Consumer-MEMS defaults. The previous sig_gyro (deg2rad(0.01)/60 =
    # 2.9e-6) is a TACTICAL-grade figure -- an MPU6050 is nearer 5e-4, two
    # orders of magnitude away. Defaults this wrong are worse than none,
    # because a plausible number invites trust. Run allan.py and pass the
    # measured values; these are only a starting point.
    ap.add_argument('--sig-gyro', type=float, default=5e-4)
    ap.add_argument('--sig-ba-rw', type=float, default=1e-4)
    ap.add_argument('--sig-bg-rw', type=float, default=1e-6)
    ap.add_argument('--gps-std', type=float, default=5.3,
                    help='horizontal GPS 1-sigma at HDOP=1 (site-specific: '
                         'the dominant error is multipath, so measure it '
                         'where you actually fly)')
    ap.add_argument('--baro-std', type=float, default=5.0)
    ap.add_argument('--mag-ut-per-lsb', type=float, default=0.15,
                    help='converts logged mag_x/y/z into microtesla for the '
                         'disturbance check (LIS2MDL = 0.15; pass 1.0 if the '
                         'logger already wrote uT). Wrong units here disable '
                         'the check silently.')
    ap.add_argument('--no-earth-rate', action='store_true',
                    help='drop Coriolis/Earth-rate/transport-rate -- run '
                         'both ways on the same log to see how much they '
                         'actually matter at your scale')
    ap.add_argument('--no-gps-after', type=float, metavar='SEC',
                    help='synthetic outage: suppress GPS from this time')
    ap.add_argument('--gps-back-at', type=float, metavar='SEC',
                    help='restore GPS at this time')
    ap.add_argument('--no-zupt', action='store_true',
                    help='disable zero-velocity updates')
    ap.add_argument('--plot', metavar='PNG')
    ap.add_argument('--sweep', nargs=4,
                    metavar=('PARAM', 'LO', 'HI', 'N'),
                    help='sweep a parameter log-spaced over N values and '
                         'report NIS for each')
    args = ap.parse_args()

    kw = dict(sig_accel=args.sig_accel, sig_gyro=args.sig_gyro,
              sig_ba_rw=args.sig_ba_rw, sig_bg_rw=args.sig_bg_rw,
              gps_std_base=args.gps_std, baro_std=args.baro_std,
              mag_ut_per_lsb=args.mag_ut_per_lsb,
              use_earth_rate=not args.no_earth_rate,
              no_gps_after=args.no_gps_after, gps_back_at=args.gps_back_at,
              zupt=not args.no_zupt)

    if args.sweep:
        param, lo, hi, count = args.sweep
        # Log-spaced, because the right value is usually an order of
        # magnitude away from a first guess, not a few percent.
        values = np.logspace(np.log10(float(lo)), np.log10(float(hi)),
                             int(count))
        print(f"\nSweeping {param} over {len(values)} log-spaced values\n")
        print(f"{param:>14s} {'GPS NIS':>10s} {'baro NIS':>10s} "
              f"{'pos 1σ':>10s} {'accept%':>9s}")
        for v in values:
            kw[param] = float(v)
            r = replay(args.prefix, verbose=False, **kw)
            ekf = r['ekf']
            a, t = r['counts']['gps']
            print(f"{v:14.4e} "
                  f"{np.mean(ekf.nis_gps) if ekf.nis_gps else float('nan'):10.2f} "
                  f"{np.mean(ekf.nis_baro) if ekf.nis_baro else float('nan'):10.2f} "
                  f"{np.linalg.norm(ekf.pos_1sigma):10.2f} "
                  f"{100.0*a/t if t else float('nan'):8.1f}%")
        print("\n  Target: GPS NIS near 6, baro NIS near 1. Pick the value")
        print("  that lands closest, then check it on a second log before")
        print("  trusting it -- one log can be tuned to by coincidence.")
        return

    res = replay(args.prefix, **kw)
    report(res)
    if args.plot:
        plot(res, args.plot)


if __name__ == '__main__':
    main()
