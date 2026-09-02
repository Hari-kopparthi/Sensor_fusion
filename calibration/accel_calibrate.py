#!/usr/bin/env python3
"""
Six-position accelerometer calibration for the MPU-6050.

Why this is needed here, specifically
-------------------------------------
Measured on this unit: at rest the accelerometer reports a gravity
magnitude of 9.193 m/s^2 against a true 9.80665 -- 6.26% low. The filter
adds a gravity model of exactly 9.80665 when computing

    a_nav = C_bn @ f_meas + g_nav

so a shortfall of 0.614 m/s^2 in f_meas appears as a genuine, permanent
downward acceleration. Integrated, that predicts a vertical velocity ramp
of 0.31, 0.61, 0.92, 1.23 m/s at half-second intervals; the live run
produced 0.34, 0.63, 0.88, 1.18. The error accounts for essentially all of
the observed velocity divergence while sitting still.

Bias vs scale, and why one orientation is not enough
---------------------------------------------------
A single at-rest reading cannot tell a zero offset from a gain error:
both make gravity read 9.19 instead of 9.81. Measuring each axis with
gravity pointing BOTH ways separates them, because bias adds the same
way in both orientations while scale does not:

    m_down = -g * scale + bias
    m_up   = +g * scale + bias
      =>  bias  = (m_up + m_down) / 2
          scale = (m_up - m_down) / (2g)

Correction applied at runtime:  true = (measured - bias) / scale

Usage on the Pi:
    cd ~/dronepi-project && source venv/bin/activate
    python3 accel_calibrate.py

You will be prompted for six orientations. Rest the board on a flat, firm
surface for each one and keep it still -- hand-holding introduces tremor
far larger than the effect being measured. Orientations are described in
the FRD body frame the filter uses (x forward, y right, z down), which is
what axis_map() produces, not the raw chip axes.

Writes accel_calibration.json, loaded automatically by pi_live_nav_baro.py.
"""
import json
import os
import statistics
import sys
import time

import numpy as np
from mpu6050 import mpu6050

from pi_live_nav_baro import axis_map, read_imu

G = 9.80665
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "accel_calibration.json")
N_SAMPLES = 300
SAMPLE_DT = 0.01

# (axis index, expected sign of the FRD reading, human description)
#
# At rest the accelerometer measures specific force f = -g_body. So when an
# axis points DOWN, gravity lies along +axis in the body frame and that
# axis reads NEGATIVE. Hence "z down" (the normal flat orientation) reads
# about -9.81 on z, which matches the at-rest print in pi_live_nav_baro.py.
ORIENTATIONS = [
    (2, -1, "FLAT, normal orientation (board level, as it will be mounted)"),
    (2, +1, "UPSIDE DOWN (flipped over, still level)"),
    (0, -1, "NOSE DOWN (front edge on the table, x axis pointing at floor)"),
    (0, +1, "NOSE UP (back edge on the table, x axis pointing at ceiling)"),
    (1, -1, "RIGHT SIDE DOWN (y axis pointing at the floor)"),
    (1, +1, "LEFT SIDE DOWN (y axis pointing at the ceiling)"),
]


def collect(imu, desc):
    input(f"\n  Place the board: {desc}\n  Press Enter when it is still...")
    print("  sampling", end="", flush=True)
    samples = []
    for i in range(N_SAMPLES):
        accel, _ = read_imu(imu)
        samples.append(accel)
        if i % 60 == 0:
            print(".", end="", flush=True)
        time.sleep(SAMPLE_DT)
    print(" done")
    arr = np.array(samples)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    print(f"    mean {np.round(mean, 3)}  std {np.round(std, 3)}")
    return mean, std


def main():
    print("MPU-6050 six-position accelerometer calibration")
    print("=" * 70)
    imu = mpu6050(0x68)

    # measured[axis][sign] = mean reading of that axis in that orientation
    measured = {0: {}, 1: {}, 2: {}}

    for axis, sign, desc in ORIENTATIONS:
        mean, std = collect(imu, desc)

        # Sanity check: the axis under test should dominate, and the other
        # two should be near zero. If not, the board is not in the
        # orientation the prompt asked for and the calibration would be
        # silently wrong.
        expected = sign * G
        if abs(mean[axis] - expected) > 3.0:
            print(f"    ** WARNING: axis {'xyz'[axis]} reads "
                  f"{mean[axis]:+.2f}, expected about {expected:+.2f}.")
            print("       The board is probably not in the requested "
                  "orientation.")
            if input("       Continue anyway? [y/N] ").strip().lower() != "y":
                sys.exit(1)
        if max(std) > 0.5:
            print("    ** WARNING: high variance -- the board was moving. "
                  "Redo this orientation for a good result.")

        measured[axis][sign] = mean[axis]

    # ── Solve bias and scale per axis ────────────────────────────────────
    bias = [0.0, 0.0, 0.0]
    scale = [1.0, 1.0, 1.0]
    print("\n" + "=" * 70)
    print("CALIBRATION\n")
    for axis in range(3):
        m_up   = measured[axis][+1]
        m_down = measured[axis][-1]
        bias[axis]  = (m_up + m_down) / 2.0
        scale[axis] = (m_up - m_down) / (2.0 * G)
        print(f"  {'xyz'[axis]}:  bias {bias[axis]:+7.4f} m/s^2   "
              f"scale {scale[axis]:.5f}   "
              f"({(scale[axis] - 1.0) * 100:+.2f}% gain error)")

    # ── Verify: what would the flat orientation now read? ────────────────
    flat_raw = measured[2][-1]
    flat_corrected = (flat_raw - bias[2]) / scale[2]
    print(f"\n  Check: flat z-axis raw {flat_raw:.4f} -> corrected "
          f"{flat_corrected:.4f} m/s^2")
    print(f"         (should be very close to -{G:.4f})")

    err = abs(abs(flat_corrected) - G)
    if err > 0.05:
        print(f"  ** Residual error {err:.3f} m/s^2 is larger than expected.")
        print("     Check that each orientation was actually flat and still.")
    else:
        print(f"  Residual {err:.4f} m/s^2 -- good.")

    cal = {
        "bias":    bias,
        "scale":   scale,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note":    "FRD body frame, applied AFTER axis_map(). "
                   "true = (measured - bias) / scale",
    }
    with open(OUT_PATH, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\n  Saved to {OUT_PATH}")
    print("  pi_live_nav_baro.py will load this automatically.")
    print("=" * 70)


if __name__ == "__main__":
    main()
