"""
Shared math utilities for field_nav nodes.
"""

import math
import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def latlon_to_local_xy(
    lat: float, lon: float,
    origin_lat: float, origin_lon: float,
) -> tuple[float, float]:
    """
    Convert a WGS-84 lat/lon to a flat local East-North frame [m]
    using an equirectangular approximation.
    Accurate to ~1 cm over distances < 1 km — sufficient for field use.

    Parameters
    ----------
    lat, lon         : target point
    origin_lat, lon  : local frame origin (e.g. field entry gate)

    Returns
    -------
    (x_east, y_north) in metres
    """
    R_EARTH = 6_371_000.0   # mean Earth radius [m]

    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    cos_lat = math.cos(math.radians(origin_lat))

    x = dlon * cos_lat * R_EARTH   # East
    y = dlat * R_EARTH             # North
    return x, y


def rotation_matrix_2d(angle: float) -> np.ndarray:
    """2-D rotation matrix for the given angle [rad]."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s],
                     [s,  c]])


def mahalanobis_distance(
    z: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> float:
    """Scalar Mahalanobis distance — useful for outlier rejection."""
    diff = z - mu
    return float(math.sqrt(diff @ np.linalg.inv(cov) @ diff))
