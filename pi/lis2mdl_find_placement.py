#!/usr/bin/env python3
"""
Magnetometer placement finder -- LIS2MDL version.

Same purpose as mag_find_placement.py (which was written for the LSM303D):
answer "how far from the Pi/frame is far enough?" with a measured number.
Interference from nearby current-carrying electronics adds to Earth's real
field, so |B| reads HIGH close to the source and falls toward Earth's true
value as you move away.

Why a separate file instead of reusing mag_find_placement.py: the LIS2MDL
is a different chip with a different register map, fixed I2C address, and
-- importantly -- a fixed +-50 gauss full scale (no --range selection like
the LSM303D had). That headroom is wide enough that the saturation problem
chased on the old sensor is very unlikely to recur here, but the driver
underneath still has to match the actual chip.

Does not require calibration first -- reads raw |B| magnitude only, which
is insensitive to axis mapping/heading, only to how much extra field is
being added by nearby interference. Run a hard-iron calibration afterwards,
once a mounting position is picked and fixed.

Usage on the Pi:
    cd ~/dronepi-project && source venv/bin/activate
    python3 lis2mdl_find_placement.py [--lat 52.0]

Controls while running:
    Enter        log the CURRENT reading under a label you type
    Ctrl+C       stop and print a summary table of everything logged
"""
import statistics
import struct
import sys
import time

import smbus2

I2C_BUS  = 1
ADDR     = 0x1E          # fixed on LIS2MDL -- no SA0 ambiguity like LSM303D
WHO_AM_I_REG   = 0x4F
WHO_AM_I_VAL   = 0x40
CFG_REG_A      = 0x60
CFG_REG_B      = 0x61
CFG_REG_C      = 0x62
STATUS_REG     = 0x67
OUTX_L_REG_M   = 0x68     # X/Y/Z low/high, 6 bytes, auto-increment with 0x80

# CFG_REG_A: COMP_TEMP_EN(1) | ODR=100Hz(11) | MD=continuous(00)
CFG_A_VALUE = 0x8C
# CFG_REG_B: OFF_CANC(1) -- continuous hard-iron offset cancellation, a
# feature this chip has that the LSM303D did not.
CFG_B_VALUE = 0x02
# CFG_REG_C: BDU(1) -- block data update, avoids reading a torn sample
# straddling two ODR ticks.
CFG_C_VALUE = 0x10

UT_PER_LSB = 0.15         # fixed sensitivity: 1.5 mgauss/LSB = 0.15 uT/LSB
SATURATION_LSB = 32760    # 16-bit signed ADC ceiling, same as any 16-bit mag


class LIS2MDL:
    """Minimal driver: configure once, poll STATUS_REG, read 6 bytes."""

    def __init__(self, bus: int = I2C_BUS, addr: int = ADDR):
        self._bus = smbus2.SMBus(bus)
        self._addr = addr

        who = self._bus.read_byte_data(addr, WHO_AM_I_REG)
        if who != WHO_AM_I_VAL:
            raise OSError(f"WHO_AM_I mismatch: got 0x{who:02X}, "
                           f"expected 0x{WHO_AM_I_VAL:02X} -- wrong chip "
                           f"or bad wiring")

        self._bus.write_byte_data(addr, CFG_REG_A, CFG_A_VALUE)
        self._bus.write_byte_data(addr, CFG_REG_B, CFG_B_VALUE)
        self._bus.write_byte_data(addr, CFG_REG_C, CFG_C_VALUE)

    def data_ready(self) -> bool:
        status = self._bus.read_byte_data(self._addr, STATUS_REG)
        return bool(status & 0x08)   # ZYXDA

    def read_raw(self):
        """Returns (x, y, z) as signed 16-bit LSB counts."""
        # 0x80 OR'd into the start register enables register auto-increment
        # for a multi-byte block read (same trick as the LSM303D driver).
        d = self._bus.read_i2c_block_data(self._addr, OUTX_L_REG_M | 0x80, 6)
        x, y, z = struct.unpack('<hhh', bytes(d))
        return x, y, z


def earth_field_estimate(lat_deg: float) -> float:
    """Rough total-field estimate by latitude band, display only -- see
    mag_find_placement.py's version of this function for the same caveat."""
    lat = abs(lat_deg)
    if lat < 15:
        return 32.0
    if lat < 35:
        return 38.0
    if lat < 55:
        return 49.0
    if lat < 70:
        return 56.0
    return 60.0


def read_latitude_arg() -> float:
    for i, a in enumerate(sys.argv):
        if a == "--lat" and i + 1 < len(sys.argv):
            return float(sys.argv[i + 1])
    return 52.0   # default: UK


def classify(b_ut: float, earth_ut: float) -> str:
    """Deviation in EITHER direction counts -- interference can add to the
    field or cancel part of it. See mag_find_placement.py's classify() for
    the full rationale."""
    ratio = b_ut / earth_ut
    deviation = abs(ratio - 1.0)
    if deviation <= 0.15:
        return "GOOD "
    if deviation <= 0.35:
        return "OK   "
    if deviation <= 1.0:
        return "WEAK "
    return "POOR "


def main():
    earth_ut = earth_field_estimate(read_latitude_arg())
    print("LIS2MDL magnetometer placement finder")
    print("=" * 70)
    print(f"Fixed full scale: +-50 gauss ({UT_PER_LSB:.3f} uT/LSB) -- much")
    print(f"wider headroom than the LSM303D, saturation should be rare.")
    print(f"Reference: Earth's field here is approximately {earth_ut:.0f} uT.")
    print("Move the sensor around while watching |B|. Lower is better --")
    print("it means less interference is being added to the real field.")
    print()
    print("Press ENTER at any point to log the current position under a")
    print("label (e.g. 'on frame', '5cm stalk', '10cm stalk'). Ctrl+C to")
    print("finish and see a comparison table.\n")

    try:
        mag = LIS2MDL()
    except Exception as e:
        print(f"Could not open magnetometer: {e}")
        sys.exit(1)

    logged = []          # (label, mean_uT, std_uT)
    window = []          # rolling recent samples for a stable live readout
    WINDOW_N = 25

    import select
    print(f"  {'|B| uT':>10}  {'vs Earth':>10}  status")
    print("  " + "-" * 40)

    try:
        while True:
            if not mag.data_ready():
                time.sleep(0.005)
                continue
            try:
                x, y, z = mag.read_raw()
            except OSError:
                time.sleep(0.02)
                continue

            saturated = any(abs(v) >= SATURATION_LSB for v in (x, y, z))
            if saturated:
                print(f"  {'SATURATED':>10}  {'--':>10}  "
                      f"axis clipped -- move away from the source", end="\r")
                time.sleep(0.05)
                continue

            b_ut = ((x * UT_PER_LSB) ** 2 + (y * UT_PER_LSB) ** 2
                     + (z * UT_PER_LSB) ** 2) ** 0.5
            window.append(b_ut)
            if len(window) > WINDOW_N:
                window.pop(0)

            smoothed = statistics.mean(window)
            ratio = smoothed / earth_ut
            status = classify(smoothed, earth_ut)

            print(f"  {smoothed:10.1f}  {ratio:9.2f}x  {status}", end="\r")

            if select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                label = input("\n  Label this position: ").strip() or "unlabeled"
                logged.append((label, smoothed, statistics.pstdev(window)))
                print(f"  Logged '{label}': {smoothed:.1f} uT\n")
                print(f"  {'|B| uT':>10}  {'vs Earth':>10}  status")
                print("  " + "-" * 40)

            time.sleep(0.02)

    except KeyboardInterrupt:
        pass

    print("\n\n" + "=" * 70)
    if not logged:
        print("No positions logged. Re-run and press Enter to record some.")
        return

    print("LOGGED POSITIONS -- closest to Earth's field is best\n")
    logged.sort(key=lambda r: r[1])
    print(f"  {'label':<20} {'|B| uT':>10} {'vs Earth':>10}  status")
    print("  " + "-" * 56)
    for label, mean_ut, std_ut in logged:
        ratio = mean_ut / earth_ut
        print(f"  {label:<20} {mean_ut:10.1f} {ratio:9.2f}x  "
              f"{classify(mean_ut, earth_ut)}  (+-{std_ut:.1f})")

    best = logged[0]
    print(f"\n  Best measured: '{best[0]}' at {best[1]:.1f} uT "
          f"({best[1] / earth_ut:.2f}x Earth's field)")
    print("\n  GOOD  (within 15%)   interference negligible, use this position")
    print("  OK    (within 35%)   acceptable, calibration will absorb the rest")
    print("  WEAK  (within 100%)  move further if you reasonably can")
    print("  POOR  (>100% off)    too close -- calibration may not hold steady")
    print("=" * 70)


if __name__ == "__main__":
    main()
