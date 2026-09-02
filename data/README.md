# Recorded runs

## `20260901_174748` — 31-minute backpack walk

The reference run. All fixes applied, sensor carried on the back rather
than in hand (which matters — see the work log on why hand-carrying at
199 deg/s breaks the mechanisation).

- GNSS 97.4% accepted, HDOP 1.03 outdoors
- Position NIS 2.86 against a theoretical 3.0
- 3.19 m RMS against the receiver's own 8.86 m

Five CSVs: `_imu`, `_gps`, `_baro`, `_mag`, `_est`.

## Not committed

A **2.7-hour stationary log** (`20260901_132235`) produced the four measured
noise parameters via Allan variance. It is 56 MB, which is more than belongs
in a git repository. `tools/allan.py` is the analysis that consumed it, and
the resulting values are recorded in the top-level README.

To generate an equivalent yourself: leave the unit powered and completely
still on a solid surface for 3+ hours, discarding the first 30 minutes as
thermal warm-up, then run `allan.py` against the IMU log with the
**measured** sample rate.
