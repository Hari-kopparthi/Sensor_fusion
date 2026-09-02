"""Geodetic helpers used by pi_live_nav_eskf.py.

Extracted verbatim from the project's navigation.geodetic module so this
archive runs standalone. Only the four functions the live driver actually
needs are here -- lla_to_ned and ned_to_lla for the flat-Earth position
conversion, euler_to_quat to build the initial attitude from the static
alignment, and wgs84_radii which the first two depend on.
"""
from typing import Tuple

import numpy as np

# WGS-84 ellipsoid, taken from the project's gnss_module so this file has
# no cross-module dependency.
WGS84_A  = 6378137.0        # equatorial radius, m
WGS84_E2 = 6.694379990e-3   # first eccentricity squared

DEG2RAD   = np.pi / 180.0

RAD2DEG   = 180.0 / np.pi



def wgs84_radii(lat_rad: float) -> Tuple[float, float]:
    """Meridian radius M and normal (transverse) radius N at given latitude."""
    sin_lat = np.sin(lat_rad)
    denom   = np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    M = WGS84_A * (1.0 - WGS84_E2) / denom**3
    N = WGS84_A / denom
    return M, N


def lla_to_ned(lat: float, lon: float, alt: float,
               ref_lat: float, ref_lon: float, ref_alt: float) -> np.ndarray:
    M, N = wgs84_radii(ref_lat)
    return np.array([(lat - ref_lat) * M,
                     (lon - ref_lon) * N * np.cos(ref_lat),
                     ref_alt - alt])


def ned_to_lla(ned: np.ndarray,
               ref_lat: float, ref_lon: float, ref_alt: float
               ) -> Tuple[float, float, float]:
    M, N = wgs84_radii(ref_lat)
    return (ref_lat + ned[0] / M,
            ref_lon + ned[1] / (N * np.cos(ref_lat)),
            ref_alt - ned[2])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    q = np.array([cr*cp*cy + sr*sp*sy,
                  sr*cp*cy - cr*sp*sy,
                  cr*sp*cy + sr*cp*sy,
                  cr*cp*sy - sr*sp*cy])
    return q / np.linalg.norm(q)

