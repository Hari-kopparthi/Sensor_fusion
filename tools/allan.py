"""
Allan variance analysis -- extract ESKF noise parameters from a
stationary IMU log.
=====================================================================

Produces the four sigmas that eskf.py needs:
    sig_accel   accel white noise      m/s^2/sqrt(Hz)
    sig_gyro    gyro  white noise      rad/s/sqrt(Hz)
    sig_ba_rw   accel bias random walk m/s^3/sqrt(Hz)
    sig_bg_rw   gyro  bias random walk rad/s^2/sqrt(Hz)

Log requirements -- these matter more than the processing:
  * 2-3 hours MINIMUM. Bias instability sits at the bottom of the curve,
    typically tau = 100-1000 s for MEMS. A 10-minute log cannot resolve
    a feature at tau = 1000 s; you get the white-noise slope and nothing
    else, and the bias numbers you read off will be fiction.
  * Absolutely stationary. Concrete floor beats a desk. Footsteps on a
    suspended floor show up in the data.
  * Thermally settled: power on and wait 20-30 min BEFORE logging. The
    warm-up transient looks like enormous bias random walk that does not
    exist in steady operation.
  * Stable ambient temperature -- not a room where the heating cycles.
  * Raw and unfiltered, at full rate. No smoothing, no decimation.
  * Verify the actual sample rate from timestamps. Allan variance assumes
    uniform sampling, and a jittery Pi loop is not uniform.

Usage:
    python allan.py imu_log.csv --rate 100
    python allan.py imu_log.csv --rate 100 --plot allan.png

Expected CSV: one row per sample, columns
    ax, ay, az, gx, gy, gz
with an optional leading timestamp column (use --timestamp-col).
Accel in m/s^2, gyro in rad/s (see --accel-g / --gyro-dps to convert).
"""

import argparse
import sys
import numpy as np

try:
    import allantools
except ImportError:
    sys.exit("allantools not installed.  pip install allantools")

G = 9.80665


# ---------------------------------------------------------------
#  Reading the three regions of the Allan deviation curve
# ---------------------------------------------------------------

def white_noise(taus, adev):
    """Angle/velocity random walk -- the tau^(-1/2) region.

    Read the deviation at tau = 1 s. By the definition of the ARW/VRW
    coefficient, adev(1 s) IS the white-noise root-PSD directly, in
    units/sqrt(Hz). No scaling factor.

    Interpolated in log-log space because the tau grid rarely lands
    exactly on 1.0.
    """
    return float(np.exp(np.interp(np.log(1.0), np.log(taus), np.log(adev))))


def bias_instability(taus, adev):
    """The flat minimum of the curve.

    Convention: B = adev_min / 0.664. The 0.664 comes from the definition
    of bias instability as a flicker-noise floor -- it is the value the
    Allan deviation takes for flicker noise of coefficient B.

    Returns (B, tau_at_minimum). Check tau_min against your log length:
    if tau_min is within a factor of ~5 of total duration, the curve has
    not actually reached its minimum and this number is not trustworthy.
    """
    i = int(np.argmin(adev))
    return float(adev[i] / 0.664), float(taus[i])


def rate_random_walk(taus, adev):
    """The tau^(+1/2) rising region at long averaging times.

    IEEE 952 defines rate random walk by

        sigma(tau) = K * sqrt(tau / 3)      =>      K = sigma(tau) / sqrt(tau/3)

    so at tau = 3 s, sigma(3) = K exactly -- no scaling factor at all.
    (The convention is often quoted as "read K at tau = 3 s", which is
    where that special case comes from. It is NOT "sigma(3) * sqrt(3)".)

    Reading at tau = 3 s only works if 3 s is actually ON the rising
    slope, which for MEMS it usually is not -- at 3 s you are typically
    still in the white-noise region. So instead: find where the log-log
    slope is closest to +1/2 and invert the relation there.

    Returns (K, tau_used, slope_found). A slope far from +0.5 means the
    log is too short to reach this region and the value is not
    trustworthy.
    """
    logt, loga = np.log(taus), np.log(adev)
    slopes = np.gradient(loga, logt)

    # only look in the second half of the tau range -- RRW is a long-tau
    # phenomenon, and short-tau slopes are dominated by white noise
    start = len(taus) // 2
    j = start + int(np.argmin(np.abs(slopes[start:] - 0.5)))

    K = float(adev[j] / np.sqrt(taus[j] / 3.0))
    return K, float(taus[j]), float(slopes[j])


def analyse_channel(x, rate, name):
    taus, adev, _, _ = allantools.oadev(np.asarray(x, dtype=float),
                                        rate=rate, data_type='freq',
                                        taus='octave')
    ok = np.isfinite(adev) & (adev > 0)
    taus, adev = taus[ok], adev[ok]

    N = white_noise(taus, adev)
    B, tau_B = bias_instability(taus, adev)
    K, tau_K, slope_K = rate_random_walk(taus, adev)

    return dict(name=name, taus=taus, adev=adev,
                N=N, B=B, tau_B=tau_B, K=K, tau_K=tau_K, slope_K=slope_K)


# ---------------------------------------------------------------
#  Main
# ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--rate', type=float, required=True,
                    help='sample rate in Hz (measure it, do not assume)')
    ap.add_argument('--delimiter', default=',')
    ap.add_argument('--skip-rows', type=int, default=0,
                    help='header rows to skip')
    ap.add_argument('--timestamp-col', action='store_true',
                    help='first column is a timestamp, not ax '
                         '(shorthand for --lead-cols 1)')
    ap.add_argument('--lead-cols', type=int, default=None, metavar='N',
                    help='number of columns before ax. Needed because '
                         '--timestamp-col assumes exactly one, and '
                         'pi_live_nav_baro.py writes TWO (t, dt, ax, ...). '
                         'With the wrong count the accel columns silently '
                         'become [dt, ax, ay] -- mean |accel| reads 0.25 '
                         'instead of ~9.81, which the sanity check below '
                         'catches, but only if you read the warning.')
    ap.add_argument('--accel-g', action='store_true',
                    help='accel columns are in g, convert to m/s^2')
    ap.add_argument('--gyro-dps', action='store_true',
                    help='gyro columns are in deg/s, convert to rad/s')
    ap.add_argument('--plot', metavar='PNG',
                    help='write the Allan deviation plot here')
    args = ap.parse_args()

    data = np.loadtxt(args.csv, delimiter=args.delimiter,
                      skiprows=args.skip_rows)
    if data.ndim != 2:
        sys.exit(f"expected a 2-D array, got shape {data.shape}")

    c0 = args.lead_cols if args.lead_cols is not None else \
        (1 if args.timestamp_col else 0)
    need = c0 + 6
    if data.shape[1] < need:
        sys.exit(f"need {need} columns, found {data.shape[1]}")

    accel = data[:, c0:c0+3].copy()
    gyro  = data[:, c0+3:c0+6].copy()
    if args.accel_g:
        accel *= G
    if args.gyro_dps:
        gyro = np.deg2rad(gyro)

    n = len(data)
    duration = n / args.rate
    print(f"\n{n:,} samples at {args.rate} Hz  =  {duration/3600:.2f} hours")
    if duration < 2*3600:
        print("  WARNING: under 2 hours. Bias instability and random-walk")
        print("  numbers below are unlikely to be meaningful.")

    # sanity check: is it actually stationary?
    a_mean = accel.mean(axis=0)
    print(f"\nmean accel: [{a_mean[0]:+.4f} {a_mean[1]:+.4f} {a_mean[2]:+.4f}] m/s^2"
          f"   |a| = {np.linalg.norm(a_mean):.4f}")
    if abs(np.linalg.norm(a_mean) - G) > 0.5:
        print(f"  WARNING: |accel| is {np.linalg.norm(a_mean):.2f}, expected ~{G:.2f}."
              " Check units and that the unit was at rest.")
    g_mean = gyro.mean(axis=0)
    print(f"mean gyro:  [{g_mean[0]:+.2e} {g_mean[1]:+.2e} {g_mean[2]:+.2e}] rad/s"
          f"   ({np.rad2deg(np.linalg.norm(g_mean))*3600:.1f} deg/hr)")
    print("  (nonzero mean gyro IS your static bias estimate -- Earth rate")
    print("   is 15 deg/hr, so anything much larger than that is sensor bias)")

    results = []
    for i, ax in enumerate('xyz'):
        results.append(analyse_channel(accel[:, i], args.rate, f'accel_{ax}'))
    for i, ax in enumerate('xyz'):
        results.append(analyse_channel(gyro[:, i], args.rate, f'gyro_{ax}'))

    print("\n" + "="*74)
    print("PER-AXIS RESULTS")
    print("="*74)
    print(f"{'channel':10s} {'white noise':>14s} {'bias instab':>14s} "
          f"{'tau_min':>9s} {'RRW':>12s} {'slope':>7s}")
    for r in results:
        print(f"{r['name']:10s} {r['N']:14.4e} {r['B']:14.4e} "
              f"{r['tau_B']:8.1f}s {r['K']:12.4e} {r['slope_K']:+7.2f}")

    print("\n  slope should be near +0.50 for a trustworthy RRW number.")
    print("  tau_min well below the log duration means the curve reached")
    print("  its minimum; close to it means the log is too short.")

    # axis-averaged -- the filter uses one isotropic value per quantity
    acc = results[:3]
    gyr = results[3:]
    sig_accel = float(np.mean([r['N'] for r in acc]))
    sig_gyro  = float(np.mean([r['N'] for r in gyr]))
    sig_ba_rw = float(np.mean([r['K'] for r in acc]))
    sig_bg_rw = float(np.mean([r['K'] for r in gyr]))

    print("\n" + "="*74)
    print("ESKF PARAMETERS  (axis-averaged)")
    print("="*74)
    print(f"""
    ekf = ESKF(nom, P0, ref_lat,
               sig_accel={sig_accel:.4e},   # m/s^2/sqrt(Hz)
               sig_gyro ={sig_gyro:.4e},   # rad/s/sqrt(Hz)
               sig_ba_rw={sig_ba_rw:.4e},   # m/s^3/sqrt(Hz)
               sig_bg_rw={sig_bg_rw:.4e})   # rad/s^2/sqrt(Hz)
""")
    print(f"  for reference: sig_accel = {sig_accel/G*1e6:.0f} ug/sqrt(Hz)")
    print(f"                 sig_gyro  = {np.rad2deg(sig_gyro)*60:.4f} deg/sqrt(hr)")
    print("""
  These are BENCH numbers -- stationary, still air. Motor vibration
  raises effective accel noise substantially, often by an order of
  magnitude. Correct for validating filter mechanics on a static test,
  and the right starting point, but expect to need a looser Q once the
  motors spin.

  The bias random-walk values are the least reliable of the four: they
  need the longest data and are the most sensitive to temperature drift
  contaminating the log. Treat them as a starting point to tune from.
""")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for group, ax, title, unit in ((acc, axes[0], 'Accelerometer', 'm/s²'),
                                       (gyr, axes[1], 'Gyroscope', 'rad/s')):
            for r in group:
                ax.loglog(r['taus'], r['adev'], label=r['name'])
            ax.axvline(1.0, color='grey', ls=':', lw=0.8)
            ax.set_xlabel('averaging time τ (s)')
            ax.set_ylabel(f'Allan deviation ({unit})')
            ax.set_title(title)
            ax.grid(True, which='both', alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"  plot written to {args.plot}")
        print("  check: -1/2 slope at short tau, a visible minimum, then")
        print("  a +1/2 rise. If you cannot see the minimum, log for longer.")


if __name__ == '__main__':
    main()
