#!/usr/bin/env python3
"""
Hard-iron / soft-iron calibration for the LIS2MDL magnetometer.

Why this is mandatory, not optional -- see mag_calibrate.py's docstring
(written for the LSM303D) for the full explanation; the physics is
identical, only the chip underneath changed. In short: an uncalibrated
magnetometer measures Earth's field PLUS a constant offset from every
piece of ferrous metal / current-carrying wire fixed to the vehicle. That
offset is constant in the BODY frame, so it distorts heading by an amount
that varies with heading -- an error the Kalman filter cannot detect on
its own, because it looks like a real measurement.

  hard iron  : constant additive offset. Corrected by subtracting the
               centre of the swept sphere.
  soft iron  : direction-dependent distortion. This script applies the
               diagonal approximation (per-axis gain equalisation), which
               handles axis-gain differences but not cross-axis coupling.

Method: rotate through as many orientations as possible, track the min
and max reached on each axis, then
    offset[i] = (max[i] + min[i]) / 2          <- hard iron
    scale[i]  = mean_radius / radius[i]        <- soft iron (diagonal)

No --range flag needed here (unlike mag_calibrate.py for the LSM303D):
the LIS2MDL has a single fixed +-50 gauss scale, so there is nothing to
select and saturation should not occur under normal use.

Usage on the Pi:
    cd ~/dronepi-project && source venv/bin/activate
    python3 lis2mdl_calibrate.py

Rotate the sensor slowly through EVERY orientation you can -- tumble it
about all three axes, not just spin it flat. Flat rotation alone leaves
the Z axis uncalibrated. Aim for 60 seconds or so, then press Ctrl+C.

Writes lis2mdl_calibration.json, which pi_live_nav_baro.py loads at
startup once wired up for this chip.
"""
import json
import os
import sys
import time

from lis2mdl_find_placement import LIS2MDL, UT_PER_LSB, SATURATION_LSB

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "lis2mdl_calibration.json")
MIN_SAMPLES = 300
# Each axis must swing by at least this much for its calibration to mean
# anything. An axis that never moved has min == max, and its computed
# offset would be an arbitrary single point rather than a sphere centre.
MIN_SPAN_LSB = 500


def main():
    print("LIS2MDL hard-iron calibration")
    print("=" * 70)
    print(f"Fixed full scale: +-50 gauss ({UT_PER_LSB:.3f} uT/LSB)")
    mag = LIS2MDL()
    print()
    print("Rotate the sensor slowly through EVERY orientation -- tumble it")
    print("about all three axes. Flat spinning alone will not calibrate Z.")
    print("Press Ctrl+C when the min/max values stop changing.\n")

    lo = [10**9] * 3
    hi = [-10**9] * 3
    n = 0
    n_failed = 0
    n_saturated = 0
    saturated_axes = set()

    try:
        while True:
            if not mag.data_ready():
                time.sleep(0.005)
                continue
            try:
                m = mag.read_raw()
            except OSError:
                n_failed += 1
                time.sleep(0.02)
                continue

            if any(abs(v) >= SATURATION_LSB for v in m):
                # A clipped sample's extreme is the range ceiling, not the
                # true field -- feeding it into lo/hi would silently bake
                # a wrong sphere centre in. Drop it, same as a failed read.
                n_saturated += 1
                for i, v in enumerate(m):
                    if abs(v) >= SATURATION_LSB:
                        saturated_axes.add("XYZ"[i])
                time.sleep(0.02)
                continue

            for i in range(3):
                lo[i] = min(lo[i], m[i])
                hi[i] = max(hi[i], m[i])
            n += 1

            spans = [hi[i] - lo[i] for i in range(3)]
            print(f"  samples {n:5d} (dropped {n_failed:4d}, "
                  f"saturated {n_saturated:4d})  "
                  f"spans  X {spans[0]:6d}  Y {spans[1]:6d}  Z {spans[2]:6d}",
                  end="\r")
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n")

    if n_saturated:
        print(f"  ** {n_saturated} sample(s) saturated on axis "
              f"{'/'.join(sorted(saturated_axes))} and were DROPPED, not "
              f"used. Unexpected at +-50 gauss -- check for a strong "
              f"nearby source (motor magnet, speaker) during the sweep.\n")

    if n < MIN_SAMPLES:
        print(f"Only {n} samples collected (need {MIN_SAMPLES}). "
              f"Run for longer.")
        sys.exit(1)

    spans = [hi[i] - lo[i] for i in range(3)]
    axes  = "XYZ"

    print("=" * 70)
    print("COVERAGE\n")
    poor = []
    for i in range(3):
        ok = spans[i] >= MIN_SPAN_LSB
        print(f"  {axes[i]}:  {lo[i]:7d} .. {hi[i]:7d}   "
              f"span {spans[i]:6d} LSB   {'OK' if ok else 'INSUFFICIENT'}")
        if not ok:
            poor.append(axes[i])

    if poor:
        print(f"\n  Axes {', '.join(poor)} did not move enough. Their offsets")
        print("  would be guesses, not sphere centres. Re-run and make sure to")
        print("  tumble the sensor about every axis, not just spin it flat.")
        sys.exit(1)

    # ── Hard iron: centre of the swept sphere ────────────────────────────
    offset = [(hi[i] + lo[i]) / 2.0 for i in range(3)]
    # ── Soft iron: equalise the per-axis radii (diagonal approximation) ──
    radius = [spans[i] / 2.0 for i in range(3)]
    mean_r = sum(radius) / 3.0
    scale  = [mean_r / radius[i] for i in range(3)]

    print("\n" + "=" * 70)
    print("CALIBRATION\n")
    for i in range(3):
        print(f"  {axes[i]}:  offset {offset[i]:9.1f} LSB "
              f"({offset[i] * UT_PER_LSB:7.2f} uT)   "
              f"scale {scale[i]:.4f}")

    field_uT = mean_r * UT_PER_LSB
    print(f"\n  Calibrated field magnitude: {field_uT:.1f} uT")
    if 40.0 <= field_uT <= 60.0:
        print("  Consistent with Earth's field at mid-latitudes -- good.")
    else:
        print("  Outside the ~40-60 uT expected at mid-latitudes. Either the")
        print("  coverage was incomplete, or something strongly magnetic was")
        print("  moving relative to the sensor during calibration.")

    cal = {
        "offset_lsb":  offset,
        "scale":       scale,
        "field_uT":    field_uT,
        "samples":     n,
        "dropped":     n_failed,
        "saturated":   n_saturated,
        "created":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
