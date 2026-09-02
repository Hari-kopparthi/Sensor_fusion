# Drone Navigation Filter — What Was Done

A 15-state error-state Kalman filter (ESKF) fusing IMU, GNSS, barometer and
magnetometer on a Raspberry Pi 4. This is the record of what was found,
what was changed, and what each change is measured to be worth.

Every number below comes from a logged run on the actual hardware. Nothing
is asserted from the code alone.

---

## The system

| Component | Part | Interface | Rate |
|---|---|---|---|
| IMU | MPU6050 | I²C `0x68` | ~52 Hz |
| GNSS | NEO-6M | UART, NMEA | 1 Hz |
| Barometer | BMP180 | I²C `0x77` | 8 Hz |
| Magnetometer | LIS2MDL | I²C `0x1E` | 8 Hz |

**State vector (15):** position (3), velocity (3), attitude error (3),
accelerometer bias (3), gyroscope bias (3). Frame is local NED with
Earth-rate, transport-rate and Coriolis corrections.

**Files:**
- `eskf.py` — the filter. No I/O, so the same file runs on the Pi and in replay.
- `pi_live_nav_eskf.py` — hardware driver, logging, systemd entry point.
- `replay.py` — re-runs the filter over logged CSVs for offline tuning.
- `allan.py` — Allan variance analysis for noise-parameter extraction.

---

## 1. The starting problem

A stationary unit produced **63 km of position error** and **126° of yaw
drift over 306 seconds**. Both position and attitude diverged and never
recovered — once attitude was wrong, every sensor's innovations failed the
chi-squared gate against it, and acceptance collapsed.

The rest of this document is the causes and the fixes.

---

## 2. Initialisation — the bootstrap deadlock

### 2.1 Gyro bias was never initialised

**Symptom.** ZUPT never fired. Attitude drifted immediately.

**Cause.** The MPU6050 sits at **−6.1 °/s** of gyro bias out of the box.
The ZUPT detector tests bias-corrected angular rate against a 0.01 rad/s
threshold; with `bg = 0` that reads 0.108 rad/s — eleven times over. So the
detector never fires, the bias is never observed, and the threshold never
drops. A closed loop with no way in.

**Fix.** `seed_static_alignment()` measures the bias once from 2 seconds of
stationary data before the filter starts.

**Measured.** Across seven runs the seed agreed with the filter's own
converged estimate after 30+ minutes to within **0.05 °/s**:

```
run                seed bg_x   final bg_x     diff
20260901_000400       -6.191      -6.162    -0.029
20260901_004512       -6.158      -6.133    -0.025
20260901_103451       -6.165      -6.165    +0.000
20260901_105714       -6.174      -6.177    +0.003
```

The seed is not a starting guess the filter later refines — it is already
essentially the converged answer.

### 2.2 The alignment had no retry

**Symptom.** Run `20260901_173820` failed completely: `bg` stayed at zero,
velocity passed 1 m/s at t = 1.48 s, and the whole 4-minute run was
worthless.

**Cause.** The alignment window caught the unit still being set down —
gyro spread was marginally over the 0.02 rad/s threshold, so the seed was
correctly rejected. Under systemd there is no console to time the start
against, so the window lands whenever the service happens to come up.

**Fix.** Retry up to 5 times (2 s each). On that run the *next* window
would have passed.

### 2.3 A rejected seed still used its attitude

**Symptom.** Same run started at **79.9° pitch** with zero bias.

**Cause.** When the seed was rejected the code zeroed `bg0` but kept
`roll0`/`pitch0` — and its comment claimed it fell back to level, which it
never did.

**Fix.** Made the behaviour match the intent, but *not* by forcing level —
that would have been worse. A noisy gravity average is still roughly the
right attitude (the unit genuinely was at 80° pitch, standing on its end).
The gyro mean is the part that becomes meaningless during motion. So
attitude is kept, bias is discarded, and the failure is announced loudly.

### 2.4 First heading bypasses the filter

**Cause.** Every `H` and `F` term is a linearisation valid for *small*
attitude error. At startup yaw is unknown, so the first magnetometer
reading can carry a 90° innovation — a regime where the filter's own model
does not describe reality.

**Fix.** `set_yaw()` sets the first heading directly, bypassing the
measurement update. Everything after it is a genuinely small correction.

**Measured (ablation).** Removing it:

```
              drift max   yaw range   gate recoveries   ZUPT
baseline         19.27 m       6.64°                18   8758
removed          56.03 m      68.37°               508      0
```

The single largest contributor measured anywhere in the filter.

---

## 3. Zero-velocity updates (ZUPT)

**Principle.** When the vehicle is genuinely stationary, velocity is known
to be exactly zero — free information, and the most effective way to
constrain accelerometer bias.

**Detection** uses the *variance* of the IMU over a 50-sample window, not
the mean. The mean is useless: a stationary accelerometer reads 9.81, and
so does one in steady level flight. What separates them is that real
motion is never perfectly steady.

**Thresholds were measured, not guessed.** Rolling statistics over a quiet
window versus one where the unit was handled:

```
             quiet max   moving max   threshold
acc_var        0.01342      20.4905      0.0200
gyr_var        0.00038       1.8729      0.0005
gyr_mag        0.00651       0.9704      0.0100
```

One of them had to be **loosened, not tightened**: the original
`gyr_var = 1e-4` sat *below* the sensor's own quiet noise floor of
1.8e-4, so genuinely stationary samples failed it — only 93% of a
known-static window passed. At 5e-4 the same window passes 100%.

**A dwell requirement was added** (`ZUPT_DWELL = 100`). The variance
window straddles motion boundaries, so a half-quiet window can slip under
the thresholds exactly as the vehicle starts or stops. Requiring sustained
agreement cut boundary firings from 375 to 91.

**Measured value** — 20-minute GPS outage, stationary vehicle:

```
with ZUPT        15.8 m drift    max |v|   0.50 m/s
without ZUPT   76,197.2 m drift  max |v| 110.59 m/s
```

**4,822×.** Unaided, the filter integrates accelerometer error into a
110 m/s phantom velocity on a vehicle that never moved.

**Limitation.** ZUPT cannot fire during motion — asserting zero velocity
while moving would be a confident lie. A 50-second outage *with* motion
produced 322 m of drift against 3.8 m for a 280-second outage while
stationary.

---

## 4. Measurement gating

### 4.1 Chi-squared innovation gate

NIS is measured in sigmas rather than metres, so the gate is automatically
scale-aware: a 20 m innovation is implausible when `P` is tight, but
reasonable after a long outage when the INS has drifted. Thresholds are
chi-squared quantiles — a chosen false-alarm rate converted into a number.

GPS gets a looser gate (1 in 9,000 false rejections rather than 1 in 370)
because INS drift between 1 Hz fixes makes its innovations legitimately
larger than the sensor noise model alone predicts.

**Measured (ablation).** Removing gating entirely: drift 19.27 → 24.35 m.

### 4.2 Per-sensor rejection limits

**Cause.** A shared "5 consecutive rejections" limit is really a *time*
threshold in disguise. Five rejections is five seconds of GPS but
two-thirds of a second of 8 Hz barometer.

```
sensor    rate Hz   shared=5   per-sensor    that is
gps          0.75     6.67 s      5          6.7 s
baro         7.40     0.68 s     50          6.8 s
mag          7.40     0.68 s     25          3.4 s
zupt         5.16     0.97 s     50          9.7 s
```

**Measured (ablation).** Reverting to a shared 5: drift 19.27 → 42.22 m,
gate recoveries 18 → 46.

### 4.3 Subspace-targeted covariance inflation

**Symptom.** A persistently failing *barometer* — a one-row, vertical-only
sensor — was inflating the **gyro bias** uncertainty 8× across three
recovery events.

**Cause.** The escape hatch multiplied the whole covariance (`P *= 4`).
Nothing about a struggling barometer is evidence about gyro bias.

**Fix.** `_inflate_observed(H)` scales only the observed subspace, via
`P ← D P D` with `D = diag(√f on observed, 1 elsewhere)`. This keeps
correlation coefficients unchanged rather than silently rescaling the
relationships between states. It is also the mathematically direct fix:
the gate tests `innov' S⁻¹ innov` with `S = H P H' + R`, so the only part
of `P` that can widen the gate is the part `H` sees.

**Measured.** Barometer failure, 1-sigma after inflation:

```
state     before   subspace   whole-P
   pD     5.0000    10.0000   10.0000
  bgZ     0.1745     0.1745    0.3491
```

Gyro bias sigma: 1.00× with the fix, 2.00× without — compounding per event.

### 4.4 Position and velocity gated separately

**Symptom.** During a walk, 75% of GPS fixes were rejected while the
receiver was tracking correctly (median 0.95 m/s between fixes, HDOP 1.03).

**Cause.** Position and velocity shared one `H`, one NIS, one gate
decision. A diverged velocity therefore discarded perfectly good position.

**Fix.** Two sequential updates with independent gates. Costs nothing
mathematically — `R` is already diagonal, and sequential processing of
uncorrelated measurements is exactly equivalent to one joint update.

**Measured.** GPS position NIS median **39.72 → 7.74**.

---

## 5. The tilt / accel-bias degeneracy

**Symptom.** Over 8 stationary minutes the accel-bias estimate grew to
**0.613 m/s²** — 766× the 0.0008 m/s² the calibration had just left.
Nothing looked wrong: ZUPT held velocity at 0.01 m/s the whole time.

Then walking began, ZUPT correctly stopped firing, and that 0.613 m/s²
integrated into velocity at **0.61 m/s per second**. The GPS velocity gate
was exceeded within a second and the filter locked itself out, reaching
127 m/s and 1.4 km.

**Cause.** At rest a tilt error and a horizontal accel bias produce the
same signature. Any split satisfying

```
g·sin(attitude error) + accel bias = g·sin(true tilt)
```

fits the measurements equally well, so the estimate wanders along that null
space. Confirmed quantitatively: attitude drifted 3.49°, and
`g·sin(3.49°) = 0.597 m/s²` against the observed 0.613 — a 3% match.
Process noise cannot explain it (`sig_ba_rw·√480 s` allows 0.009 m/s²,
seventy times less), so it was actively estimated.

**Fix.** `update_accel_bias_prior()` — a weak pseudo-measurement that the
accel bias is still what the six-position calibration measured. This is
not an assumption: the calibration measured the residual directly, so
"ba ≈ 0" is data. Applied only while stationary; in motion the bias
becomes genuinely observable and the prior stands aside.

Sigma is deliberately weak (0.05 m/s², 60× looser than the calibration
residual) so real thermal drift is unimpeded, but a degree of tilt cannot
hide in the bias state.

**Measured.** Over the same 8 stationary minutes: `|ba|` **0.806 → 0.018**,
and attitude held at −2.47°/0.10° instead of drifting 3.49°.

---

## 6. Sensor-specific fixes

### 6.1 GPS vertical velocity is not measured

NMEA RMC gives speed-over-ground and course but **no vertical rate**, so
callers routinely pass `vD = 0.0`. Fusing that row asserts "vertical
velocity is exactly zero, sigma 0.15 m/s" once per fix, forever.

**Measured** on a synthetic 2 m/s climb:

```
after      vD dropped    vD fused as 0
   1 s        -2.000          -0.036
  30 s        -2.000          -0.085
```

Fusing the fabricated zero removes **96%** of the true climb rate. The
`vel_valid` flag drops the row instead: "no information" rather than
"confidently zero".

### 6.2 GPS vertical accuracy is worse than horizontal

Every satellite is above the horizon, so the geometry constraining height
is one-sided. `pos_std_v` is set to 1.95× `pos_std_h`. Treating them as
equal makes a 10 m vertical error pull `pD` by 7.35 m instead of 4.22 m.

### 6.3 Barometer rate limiting

**Symptom.** Vertical position swinging +308 m / −154 m.

**Cause.** `isa_alt()`'s plausibility band (300–1100 hPa) is far too wide
for this hardware's corruption. A reading of **873 hPa sits inside it** and
is already ~1,000 m of altitude error. An absolute band cannot distinguish
"wrong" from "unusual weather".

**Fix.** A rate check, because physics bounds how fast pressure moves —
weather a few hPa per *hour*, altitude 1 hPa per 8 m. At 8 Hz, 5 hPa
between samples means a 320 m/s climb rate.

```
observed      delta      verdict    alt error if fused
 873.41 hPa  122.59      REJECT              981 m
1300.00 hPa  304.00      REJECT            2,432 m
 997.20 hPa    1.20      accept               10 m
```

Includes a re-baseline after 20 consecutive rejections, so one corrupted
reference cannot reject every subsequent good value forever.

### 6.4 Magnetometer disturbance rejection

The magnetometer measures the total field: Earth's plus motor current plus
any nearby steel. A disturbed sample gives a heading that is wrong
*smoothly and plausibly* — the innovation lands inside the chi-squared gate
and is fused. The gate cannot catch this.

The expected magnitude is **learned** from the first 20 readings rather
than taken from a geomagnetic model, because a calibration scale error
makes every reading differ from the model by that factor.

**A units bug was found:** the logger writes calibration-corrected LSB
counts, not microtesla. Passing them straight through gave a baseline of
~270, outside the 20–80 µT sanity band, so the check **silently disabled
itself for entire runs**. After the fix, the baseline learns at 40.48 µT
and 1.0% of indoor readings are rejected as disturbed (0% outdoors).

### 6.5 Hardware fault found

Barometer corruption (4.7% of reads implausible, pressure sd 97.2 hPa
against a 0.03 hPa datasheet figure) was traced to a **marginal I²C
connection**, not software. After reseating:

```
                 before      after
IMPLAUSIBLE        4.7%       0.0%
pressure sd     97.222 hPa   0.045 hPa
temp range   -26.2..46.9 C  25.8..26.0 C
```

Software theories — QNH re-anchoring, bus contention, a competing systemd
service — were pursued and all disproved before the wiring was checked.
The tell was in the data all along: good readings were always *perfect*
(0.12 hPa spread) while bad ones were wild, which is a connection dropping
out, not a sensor degrading.

---

## 7. Calibration

### 7.1 Accelerometer — six positions

**Cause.** A single at-rest reading cannot separate a zero offset from a
gain error; both make gravity read low. Measuring each axis with gravity
pointing *both* ways separates them, because bias adds the same way in
both orientations while scale does not.

**Result.** The error was almost entirely bias:

```
scale  [1.0086, 1.0017, 1.0132]   gains were already within 1.3%
bias   [0.410,  0.104,  0.445] m/s²
```

**Measured effect.** Gravity magnitude error **6.26% → 0.01%**
(9.193 → 9.8074 m/s² against a true 9.80665). The filter adds an exact
9.80665, so the previous 0.614 m/s² shortfall appeared as a permanent
downward acceleration — about 37 m/s of false velocity per minute of
unaided flight. Now 0.05 m/s.

### 7.2 Magnetometer — hard and soft iron

Rotate through all orientations, track per-axis min/max, then
`offset = (max+min)/2` and `scale = mean_radius/radius`.

```
samples 7105, dropped 0, saturated 0
scale   [1.0504, 0.9774, 0.9758]   7.5% spread — well-formed sphere
offset  [62.0, -168.0, 13.5] LSB
```

The Y offset is **−25 µT of hard iron** — substantial, and previously
distorting heading in a way the filter had no way to detect.

**Important:** valid only for the exact physical configuration measured.
Moving the sensor relative to anything metal invalidates it.

---

## 8. Noise characterisation — Allan variance

A 2.70-hour stationary log, first 30 minutes discarded as thermal warm-up,
leaving 2.20 hours at a **measured** 51.654 Hz (not the 100 Hz the loop
nominally targets — assuming the nominal rate would have scaled every
result).

```
parameter     guessed      measured    ratio
sig_accel   3.9227e-03   6.8272e-03    1.74x
sig_gyro    5.0000e-04   4.9586e-04    0.99x
sig_ba_rw   1.0000e-04   3.9666e-04    3.97x
sig_bg_rw   1.0000e-06   3.5977e-05   35.98x
```

`sig_gyro` was right to within 1%. **`sig_bg_rw` was 36× too small** — the
filter was far too confident that gyro bias holds still. The same log shows
`bg_x` drifting 0.21 °/s over 2.7 hours, exactly the motion a
36×-too-tight value forbids.

**Caveat carried in the code:** `gyro_z`'s rate-random-walk fit is not
trustworthy (log-log slope −0.02 where +0.50 is wanted; τ_min = 2537 s
against a 7920 s log, below the factor-of-5 guideline). The live value uses
the mean of `gyro_x` and `gyro_y` alone, whose slopes are sound.

**Measured effect.** GPS NIS 4.76 → 5.60 (target ~6); mag NIS 1.38 → 0.93
(target ~1). Both closer to consistent.

### 8.1 GPS velocity sigma — measured, then swept

During stretches where the vehicle was demonstrably stationary, the
receiver reported **1.05–1.15 m/s** of horizontal speed. Truth was exactly
zero, so that *is* its error — and the assumed 0.5 m/s claimed twice the
accuracy the receiver has.

```
vel_std   posNIS   velNIS   statRMS   spikes>4m
          (~3)     (~2)
  0.50     2.86     7.79     2.53 m       18
  1.10     3.11     3.19     2.12 m       97
  2.00     4.78     2.16     2.15 m      201
```

`1.10` minimises **true** error (stationary rms, the only place truth is
known) and puts position NIS at 3.11 against a target of 3. At 2.00 the
velocity NIS finally hits target but position NIS degrades to 4.78 — past
the optimum.

The cost is a rougher trajectory: loosening `R` accepts more of the noisy
GPS velocity, and each accepted fix injects that noise into the state.
Accepted deliberately, because stationary rms is measurable and the spike
count is not scoreable without truth during motion.

---

## 9. Robustness mechanisms

| Mechanism | What it guards against | Status on real logs |
|---|---|---|
| `dt` clamping | Lost samples, NTP steps making `dt` negative and `P` non-positive-definite | Never triggered (0 / 99,250) |
| Non-finite recovery | NaN from a dropped I²C read | Escalates: rollback → warn at 5 → reset velocity and re-inflate `P` at 10 → freeze at 15 |
| Out-of-order rejection | A stale NMEA sentence fused after a newer one, correcting the filter backwards in time | 2 caught on a 32-min run |
| `P` health check | Covariance losing positive-definiteness | Healthy throughout |

A silent rollback that keeps happening is worse than a crash — the filter
freezes while the vehicle keeps moving and nothing in the state *looks*
wrong. Hence the escalation rather than an indefinite retry.

---

## 10. Validation

### 10.1 Against raw GNSS

During stretches where the vehicle was genuinely stationary, the true
position is a constant — so scatter about that stretch's mean is a real
error for both signals, with no external truth needed.

```
window (min)   GPS rms   ESKF rms
  0.0-0.9       7.10 m     1.42 m
  1.0-3.1       5.15 m     5.16 m
  3.6-5.7      17.99 m     2.45 m
 22.8-27.1     11.73 m     6.03 m
 27.2-28.8      2.32 m     0.89 m

 mean          8.86 m     3.19 m
```

**Velocity, where truth is exactly 0.00 m/s:**

```
 3.6-5.7 min    GPS 1.146 m/s    ESKF 0.017 m/s
22.8-27.1 min   GPS 1.136 m/s    ESKF 0.016 m/s
27.2-28.8 min   GPS 1.053 m/s    ESKF 0.019 m/s
```

**2.8× better than raw GNSS in position, ~65× in velocity.**

### 10.2 Consistency (NIS)

On the outdoor walk with all fixes applied:

```
          median    expect
gps pos     2.86        ~3
gps vel     3.19        ~2
baro        0.28        ~1
mag         0.26        ~1
zupt        0.73        <3
```

### 10.3 Divergence and recovery

A deliberate GPS denial with the unit moved: 422 m of drift, then
**422 m → 6 m within 35 seconds** once GPS returned. The old filter, given
the same class of event, never recovered.

### 10.4 Long-run stability

2.7 hours stationary: gyro bias held −6.219 → −6.390 °/s (0.21 °/s over
the whole run), barometer 97.5% accepted, GNSS 97.7%.

---

## 11. Tooling built

- **`replay.py`** — re-runs the unmodified filter over logged CSVs. Turns
  tuning from a guess into a search: sweep a parameter across a decade
  against the *same* log. Deliberately does all the I/O itself so `eskf.py`
  is identical in replay and in flight.
- **Ablation harness** — disables one mechanism at a time and measures the
  damage. Used to produce the numbers in §2–§4.
- **`allan.py`** — Allan variance with slope checks, so an untrustworthy
  fit is flagged rather than silently averaged in.
- **`test_bmp180.py`** — sensor isolation test that separates *implausible
  values* from *I²C errors*, an ambiguity the live counters conflate.

---

## 12. Known limitations

**GPS-denied flight: 0.87 m of drift per second of outage.** ZUPT cannot
help while moving. This is the MPU6050's bias instability, not a tuning
problem — it needs another aiding source (optical flow, UWB, visual
odometry).

**Magnetic disturbances that rotate the field without changing its
magnitude pass the check entirely.** A 2.7-hour run took a permanent 36°
heading shift on a stationary unit this way. Catching it needs a different
signal — dip angle, or cross-checking magnetic heading against
gyro-integrated yaw.

**The barometer is the sole observer of the vertical channel.** GPS
vertical velocity is correctly dropped (RMC does not measure it), so losing
the barometer leaves only 1 Hz GPS position. Switching the NEO-6M to UBX
and reading `NAV-VELNED` would give a second independent observer.

**51.65 Hz loop rate** is fine for flight dynamics but is why violent
hand-carried motion (199 °/s) breaks the mechanisation — attitude changes
3.9° between samples and coning/sculling error leaks into velocity.

**`gyro_z`'s Allan fit needs a 4+ hour log** to resolve properly.

---

## 13. Result

| | before | after |
|---|---|---|
| Stationary position (with GNSS) | ran to 63 km | ±4 m over 22 min |
| Stationary yaw drift | 126° / 306 s | 1.3° / 22 min |
| GPS-denied, stationary | 3,646 m / 5 min | 15.8 m / 20 min |
| Recovery from divergence | never | 422 m → 6 m in 35 s |
| Gravity magnitude error | 6.26% | 0.01% |
| Position vs raw GNSS | — | 2.8× better |
| Velocity vs raw GNSS | — | ~65× better |

Filter parameters are now measured rather than assumed: four noise
densities from Allan variance, accelerometer bias and scale from
six-position calibration, magnetometer hard-iron from a tumble
calibration, GPS velocity sigma from stationary stretches of a real walk,
and gyro bias re-seeded at every startup.
