"""Observer position and line-of-sight geometry in the GCRS."""

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from okapi.orbits.twobody import propagate


def observer_position(site: EarthLocation, t: Time) -> np.ndarray:
    """Geocentric GCRS position of the site, km."""
    pos, _ = site.get_gcrs_posvel(t)
    return np.array(pos.xyz.to_value(u.km), dtype=float)


def unit_vector(ra: float, dec: float) -> np.ndarray:
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def radec_of(direction: np.ndarray) -> tuple[float, float]:
    d = direction / np.linalg.norm(direction)
    return float(np.arctan2(d[1], d[0])) % (2 * np.pi), float(np.arcsin(np.clip(d[2], -1, 1)))


def tangent_offset(direction: np.ndarray, ra0: float, dec0: float) -> tuple[float, float]:
    """(east, north) offsets in radians of ``direction`` from (ra0, dec0), gnomonic projection."""
    d = direction / np.linalg.norm(direction)
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.array([-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)])
    centre = unit_vector(ra0, dec0)
    k = float(np.dot(d, centre))
    return float(np.dot(d, east) / k), float(np.dot(d, north) / k)


def topocentric(
    r0: np.ndarray, v0: np.ndarray, t_s: float, site: EarthLocation, epoch: Time
) -> np.ndarray:
    """Vector from the site to a two-body object ``t_s`` seconds after ``epoch`` (km, GCRS)."""
    r, _ = propagate(r0, v0, t_s)
    return r - observer_position(site, epoch + t_s * u.s)
