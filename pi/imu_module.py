"""
Tactical-grade IMU error model
Python port of Paul Groves' IMU_model.m
"Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems", 2nd Ed.

Error model (applied in order):
  1. Scale factor and cross-coupling errors  (M_a, M_g)
  2. Constant biases                         (b_a, b_g)
  3. Gyro g-dependent biases                 (G_g * f_true)
  4. White noise                             (root-PSD / sqrt(dt))
  5. Quantization                            (mid-tread round, carry residual)

Tactical-grade defaults match Groves' Inertial_Demo_1ECEF configuration.
"""

import numpy as np

# ─── Unit conversion constants ───────────────────────────────────────────────
DEG2RAD    = np.pi / 180.0
MICRO_G_TO_MS2 = 9.80665e-6    # 1 µg → m/s²


# ─── Default tactical-grade IMU error parameters ─────────────────────────────
# These match Table in Groves' Inertial_Demo_1ECEF exactly.
def tactical_imu_errors() -> dict:
    """
    Tactical-grade IMU error configuration (Groves Inertial_Demo_1ECEF).

    Accelerometer biases:  ~900 µg  (vs MEMS ~50 µg)
    Gyro biases:           ~10 °/hr (vs MEMS ~360 °/hr for low-cost)
    Includes scale factor, cross-coupling, g-dependent gyro bias, quantization.
    """
    return {
        # Accelerometer biases (m/s²)
        'b_a': np.array([900, -1300, 800]) * MICRO_G_TO_MS2,

        # Gyro biases (rad/s)
        'b_g': np.array([-9, 13, -8]) * DEG2RAD / 3600,

        # Accelerometer scale factor and cross-coupling (dimensionless, 3x3)
        'M_a': np.array([[ 500, -300,  200],
                         [-150, -600,  250],
                         [-250,  100,  450]]) * 1e-6,

        # Gyro scale factor and cross-coupling (dimensionless, 3x3)
        'M_g': np.array([[ 400, -300,  250],
                         [   0, -300, -150],
                         [   0,    0, -350]]) * 1e-6,

        # Gyro g-dependent bias (rad·s/m, 3x3)
        # omega_g_dep = G_g @ f_true
        'G_g': np.array([[ 0.9, -1.1, -0.6],
                         [-0.5,  1.9, -1.6],
                         [ 0.3,  1.1, -1.3]]) * DEG2RAD / (3600 * 9.80665),

        # Accelerometer noise root-PSD (m s^{-1.5}  i.e. m/s² per √Hz)
        'accel_noise_root_PSD': 100 * MICRO_G_TO_MS2,

        # Gyro noise root-PSD (rad s^{-0.5}  i.e. rad/s per √Hz)
        'gyro_noise_root_PSD':  0.01 * DEG2RAD / 60,

        # Quantization levels
        'accel_quant_level': 1e-2,    # m/s²
        'gyro_quant_level':  2e-4,    # rad/s
    }


# ─── IMU model ────────────────────────────────────────────────────────────────
def imu_model(
    dt:              float,
    f_true:          np.ndarray,      # true specific force, body frame (3,)
    omega_true:      np.ndarray,      # true angular rate,  body frame (3,)
    IMU_errors:      dict,
    quant_residuals: np.ndarray,      # (6,) carry-over from previous epoch
) -> tuple:
    """
    Apply full tactical-grade IMU error model to true specific force and
    angular rate.  Direct port of IMU_model.m (Groves, 2012).

    Error chain:
      accel: (I + M_a) * f_true  +  b_a  +  noise  →  quantize
      gyro:  (I + M_g) * w_true  +  b_g  +  G_g*f  +  noise  →  quantize

    Parameters
    ----------
    dt              : time step (s)
    f_true          : true specific force in body frame (m/s²)
    omega_true      : true angular rate  in body frame (rad/s)
    IMU_errors      : dict from tactical_imu_errors() or custom
    quant_residuals : (6,) accumulated quantization carry-overs

    Returns
    -------
    f_meas          : measured specific force  (m/s²)
    omega_meas      : measured angular rate    (rad/s)
    quant_residuals : updated (6,) carry-overs
    """
    quant_residuals = quant_residuals.copy()

    # ── Accelerometer ────────────────────────────────────────────────────
    f_meas = ((np.eye(3) + IMU_errors['M_a']) @ f_true
              + IMU_errors['b_a']
              + (IMU_errors['accel_noise_root_PSD'] / np.sqrt(dt))
                * np.random.randn(3))

    # ── Gyroscope (includes g-dependent bias) ────────────────────────────
    omega_meas = ((np.eye(3) + IMU_errors['M_g']) @ omega_true
                  + IMU_errors['b_g']
                  + IMU_errors['G_g'] @ f_true
                  + (IMU_errors['gyro_noise_root_PSD'] / np.sqrt(dt))
                    * np.random.randn(3))

    # ── Quantization — mid-tread round() quantizer, carry residual ───────
    # Matches IMU_model.m exactly: quantize the measurement itself (not an
    # accumulated delta-v/delta-theta), using round-to-nearest so the
    # quantization error is unbiased (+/- level/2), rather than floor()
    # which truncates toward -inf and injects a constant -level/2 bias.
    ql_a = IMU_errors['accel_quant_level']
    n_a = np.round((f_meas + quant_residuals[:3]) / ql_a)
    quant_residuals[:3] = f_meas + quant_residuals[:3] - n_a * ql_a
    f_meas = n_a * ql_a

    ql_g = IMU_errors['gyro_quant_level']
    n_g = np.round((omega_meas + quant_residuals[3:]) / ql_g)
    quant_residuals[3:] = omega_meas + quant_residuals[3:] - n_g * ql_g
    omega_meas = n_g * ql_g

    return f_meas, omega_meas, quant_residuals


# ─── Convenience: MEMS-grade parameters for comparison ───────────────────────
def mems_imu_errors() -> dict:
    """
    Low-cost MEMS-grade IMU (no scale factor / cross-coupling / g-dependent terms).
    Equivalent to the simple model used in earlier navigation.py.
    """
    zero3 = np.zeros((3, 3))
    return {
        'b_a':  np.array([0.02, -0.01,  0.015]),   # m/s²
        'b_g':  np.array([5e-4, -3e-4,  4e-4]),    # rad/s  (~100 °/hr)
        'M_a':  zero3,
        'M_g':  zero3,
        'G_g':  zero3,
        'accel_noise_root_PSD': 0.05,               # m/s²/√Hz
        'gyro_noise_root_PSD':  0.005,              # rad/s/√Hz
        'accel_quant_level':    1e-4,               # m/s²
        'gyro_quant_level':     1e-6,               # rad/s
    }
