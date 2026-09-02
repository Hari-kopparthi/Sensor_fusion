#!/usr/bin/env python3
"""
BMP180 barometer isolation test.

Written to answer one question the live nav logs cannot: when a pressure
reading comes back implausible (-69 hPa, 173 hPa, 254 hPa were all seen on
this build), is the BMP180 itself/its wiring at fault, or is it collateral
damage from other traffic on the shared I2C bus?

Deliberately imports BMP180 and isa_alt from pi_live_nav_baro rather than
reimplementing them. A fresh driver that happened to work would prove
nothing about why the real one fails -- this has to exercise the same code
path that is actually producing bad readings.

Three failure classes are counted SEPARATELY, because the live nav script's
baro=X/Y counter conflates them and that ambiguity is what made this hard
to diagnose in the first place:

    OK          plausible pressure, sane altitude
    IMPLAUSIBLE read succeeded but the value is physically impossible,
                i.e. the I2C transaction returned corrupted bytes
    I2C ERROR   the transaction itself failed (OSError from smbus2)

Usage on the Pi:
    cd ~/dronepi-project && source venv/bin/activate

    # baseline: barometer alone, nothing else touching the bus
    python3 test_bmp180.py

    # same test, but with the IMU and magnetometer hammered concurrently.
    # If corruption appears ONLY here, the fault is bus contention, not
    # the barometer.
    python3 test_bmp180.py --stress

    python3 test_bmp180.py --duration=120     # default 60 s

Run the baseline FIRST and let it finish before trying --stress, so the two
numbers are comparable.
"""
import statistics
import sys
import time

from pi_live_nav_baro import BMP180, isa_alt

# Physically possible range at any inhabited altitude, same bounds isa_alt()
# enforces. Anything outside this is corrupted bytes, not weather.
P_MIN_HPA = 300.0
P_MAX_HPA = 1100.0
# Temperature sanity: the BMP180 is rated -40..+85 C. Outside that the
# temperature bytes are corrupt too, which matters because temperature
# feeds the altitude computation.
T_MIN_C = -40.0
T_MAX_C = 85.0


def read_duration_arg() -> float:
    for a in sys.argv:
        if a.startswith("--duration="):
            return float(a.split("=", 1)[1])
    return 60.0


def open_stress_devices():
    """Open the other two I2C devices so they can be polled concurrently.

    Returns (imu, mag), either of which may be None if unavailable -- a
    missing device makes the stress test less thorough but should not stop
    it, since the point is to add bus traffic, not to test those sensors.
    """
    imu = mag = None
    try:
        from mpu6050 import mpu6050
        imu = mpu6050(0x68)
        print("  stress: MPU6050 (0x68) opened")
    except Exception as e:
        print(f"  stress: MPU6050 unavailable ({e.__class__.__name__}) -- "
              f"continuing without it")
    try:
        from lis2mdl_find_placement import LIS2MDL
        mag = LIS2MDL()
        print("  stress: LIS2MDL (0x1E) opened")
    except Exception as e:
        print(f"  stress: LIS2MDL unavailable ({e.__class__.__name__}) -- "
              f"continuing without it")
    return imu, mag


def poke_stress_devices(imu, mag, seconds: float):
    """Hammer the other I2C devices for `seconds`, errors ignored.

    Must run at the LIVE LOOP's rate, not the barometer's. An earlier
    version of this poked each device once per barometer read (~8 Hz) and
    reported a clean bus -- but pi_live_nav_baro.py reads the MPU6050 on
    every iteration of a loop that spins at roughly 200 Hz (two I2C
    transactions each), and the magnetometer's data_ready() was being hit
    at the same rate by the busy-poll bug. That is ~600 transactions/sec
    against the ~16/sec the gentle version produced, so it was not
    reproducing the condition under test at all.

    Their own reliability is not what is being measured; they exist only to
    occupy the bus while the barometer's multi-byte conversion sequence is
    in progress.
    """
    deadline = time.time() + seconds
    while time.time() < deadline:
        if imu is not None:
            try:
                imu.get_accel_data()
                imu.get_gyro_data()
            except Exception:
                pass
        if mag is not None:
            try:
                mag.data_ready()
                mag.read_raw()
            except Exception:
                pass
        time.sleep(0.005)      # matches the nav loop's own sleep


def main():
    stress = "--stress" in sys.argv
    duration = read_duration_arg()

    print("BMP180 isolation test")
    print("=" * 70)
    print(f"Mode      : {'STRESS (IMU + mag polled concurrently)' if stress else 'BASELINE (barometer alone)'}")
    print(f"Duration  : {duration:.0f} s")
    print(f"Plausible : {P_MIN_HPA:.0f}-{P_MAX_HPA:.0f} hPa, {T_MIN_C:.0f}..{T_MAX_C:.0f} C")
    print()

    try:
        baro = BMP180()
    except Exception as e:
        print(f"Could not open BMP180: {e}")
        print("If this fails outright the wiring is broken, not marginal --")
        print("check VCC/GND/SDA/SCL before anything else.")
        sys.exit(1)

    imu = mag = None
    if stress:
        imu, mag = open_stress_devices()
        print()

    n_ok = n_implausible = n_i2c_err = 0
    pressures = []
    temps = []
    bad_samples = []          # (elapsed_s, p_hpa, t_c) for the summary
    last_p = None
    n_frozen = 0              # identical consecutive readings

    t_start = time.time()
    print(f"  {'t (s)':>7}  {'p (hPa)':>9}  {'T (C)':>7}  {'alt (m)':>9}  status")
    print("  " + "-" * 56)

    try:
        while time.time() - t_start < duration:
            elapsed = time.time() - t_start
            try:
                p_hpa, t_c = baro.read()
            except OSError as e:
                n_i2c_err += 1
                print(f"  {elapsed:7.1f}  {'--':>9}  {'--':>7}  {'--':>9}  "
                      f"I2C ERROR: {e}")
                if stress:
                    poke_stress_devices(imu, mag, 0.125)
                else:
                    time.sleep(0.125)
                continue

            plausible = (P_MIN_HPA <= p_hpa <= P_MAX_HPA
                         and T_MIN_C <= t_c <= T_MAX_C)
            if not plausible:
                n_implausible += 1
                bad_samples.append((elapsed, p_hpa, t_c))
                print(f"  {elapsed:7.1f}  {p_hpa:9.2f}  {t_c:7.1f}  "
                      f"{'--':>9}  IMPLAUSIBLE (corrupted bytes)")
            else:
                n_ok += 1
                pressures.append(p_hpa)
                temps.append(t_c)
                alt = isa_alt(p_hpa, t_c + 273.15)
                # A sensor whose value never changes at all is as broken as
                # one returning garbage -- the STATUS bit says "ready" but
                # the register is stale. Same frozen-axis check the
                # magnetometer static test uses.
                if last_p is not None and p_hpa == last_p:
                    n_frozen += 1
                last_p = p_hpa
                # Only print occasionally when healthy, so corruption stands
                # out instead of scrolling past in a wall of good readings.
                if n_ok % 8 == 1:
                    print(f"  {elapsed:7.1f}  {p_hpa:9.2f}  {t_c:7.1f}  "
                          f"{alt:9.2f}  OK")

            # ~8 Hz overall, matching BARO_PERIOD in the nav script. In
            # stress mode the gap is spent saturating the bus rather than
            # idling, which is the whole point of the mode.
            if stress:
                poke_stress_devices(imu, mag, 0.125)
            else:
                time.sleep(0.125)

    except KeyboardInterrupt:
        print("\n  (interrupted)")

    # ── Summary ──────────────────────────────────────────────────────────
    total = n_ok + n_implausible + n_i2c_err
    print("\n" + "=" * 70)
    print("RESULTS\n")
    if total == 0:
        print("  No reads completed.")
        return

    print(f"  total reads      {total:6d}")
    print(f"  OK               {n_ok:6d}  ({100.0*n_ok/total:5.1f}%)")
    print(f"  IMPLAUSIBLE      {n_implausible:6d}  ({100.0*n_implausible/total:5.1f}%)")
    print(f"  I2C ERROR        {n_i2c_err:6d}  ({100.0*n_i2c_err/total:5.1f}%)")

    if n_ok >= 2:
        p_std = statistics.pstdev(pressures)
        print(f"\n  pressure    mean {statistics.mean(pressures):8.2f} hPa   "
              f"sd {p_std:6.3f}   range {min(pressures):.2f}..{max(pressures):.2f}")
        print(f"  temperature mean {statistics.mean(temps):8.2f} C     "
              f"range {min(temps):.1f}..{max(temps):.1f}")
        # 0.03 hPa is the datasheet noise at OSS=3 (~0.25 m). Meaningfully
        # more than that on a stationary bench means electrical noise, not
        # weather.
        if p_std > 0.15:
            print(f"  ** pressure noise ({p_std:.3f} hPa) is well above the "
                  f"~0.03 hPa datasheet\n     figure for OSS=3. Suggests "
                  f"electrical noise rather than a clean signal.")
        if n_frozen > n_ok * 0.5:
            print(f"  ** {n_frozen}/{n_ok} readings identical to the previous "
                  f"one -- the register\n     may be stale rather than "
                  f"genuinely updating.")

    if bad_samples:
        print(f"\n  First few corrupted values (what the bytes decoded to):")
        for elapsed, p_hpa, t_c in bad_samples[:8]:
            print(f"    t={elapsed:6.1f}s   p={p_hpa:10.2f} hPa   T={t_c:7.1f} C")

    print("\n" + "-" * 70)
    print("HOW TO READ THIS\n")
    if n_implausible == 0 and n_i2c_err == 0:
        print("  Clean run. If the live nav script still shows corruption,")
        print("  the barometer is fine on its own and the fault is")
        print("  interaction with other bus traffic -- re-run with --stress.")
    else:
        pct = 100.0 * (n_implausible + n_i2c_err) / total
        print(f"  {pct:.1f}% of reads failed.")
        if not stress:
            print("  Corruption with NOTHING else on the bus points at the")
            print("  BMP180 itself or its wiring -- reseat VCC/GND/SDA/SCL")
            print("  at both ends and re-run before changing any software.")
        else:
            print("  Compare against the baseline run. Substantially worse")
            print("  here means bus contention; roughly the same means the")
            print("  barometer/wiring is the problem regardless of traffic.")
    print("=" * 70)


if __name__ == "__main__":
    main()
