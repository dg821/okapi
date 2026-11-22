"""Simulate what the camera would measure of a satellite, to test the whole orbit chain."""

from typing import Any, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import ITRS, EarthLocation, SkyCoord
from astropy.time import Time

from okapi.astrometry.observation import Observation
from okapi.orbits import MU_EARTH
from okapi.orbits.geometry import observer_position, radec_of, tangent_offset, topocentric


def geo_state(
    longitude_deg: float, epoch: Time, mu: float = MU_EARTH
) -> tuple[np.ndarray, np.ndarray]:
    """A circular equatorial orbit at 24 sidereal-hour period over the given longitude."""
    a = (mu * (86164.0905 / (2 * np.pi)) ** 2) ** (1 / 3)
    lon = np.radians(longitude_deg)

    def to_gcrs(x: float, y: float, z: float) -> np.ndarray:
        itrs = ITRS(x=x * u.km, y=y * u.km, z=z * u.km, obstime=epoch)
        return np.array(cast(Any, SkyCoord(itrs)).gcrs.cartesian.xyz.to_value(u.km), dtype=float)

    r = to_gcrs(a * np.cos(lon), a * np.sin(lon), 0.0)
    axis = to_gcrs(0.0, 0.0, 1.0)  # Earth's rotation axis in the GCRS (not exactly the z axis)
    axis /= np.linalg.norm(axis)
    rn = float(np.linalg.norm(r))
    v = np.sqrt(mu / rn) * np.cross(axis, r / rn)  # circular at the actual radius
    return r, v


def overhead_leo_state(
    site: EarthLocation, epoch: Time, altitude_km: float = 600.0, heading_deg: float = 90.0
) -> tuple[np.ndarray, np.ndarray]:
    """A circular orbit passing through the site's zenith at ``epoch``, heading ``heading_deg``
    (clockwise from north) there."""
    r_site = observer_position(site, epoch)
    up = r_site / np.linalg.norm(r_site)
    r = up * (np.linalg.norm(r_site) + altitude_km)
    north = np.array([0.0, 0.0, 1.0]) - up * up[2]
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    h = np.radians(heading_deg)
    v = np.sqrt(MU_EARTH / np.linalg.norm(r)) * (np.cos(h) * north + np.sin(h) * east)
    return r, v


def observe(
    r0: np.ndarray,
    v0: np.ndarray,
    epoch: Time,
    site: EarthLocation,
    frame_starts: list[float],
    exposure: u.Quantity,
    *,
    kind: str = "point",
    noise: u.Quantity = 1 * u.arcsec,
    timing_sigma: u.Quantity = 0.05 * u.s,
    timing_error: u.Quantity = 0 * u.s,
    seed: int = 0,
) -> list[Observation]:
    """One observation per frame (start offsets in seconds from ``epoch``). For ``kind="point"``
    the streak is inside the frame; for ``"line"`` it runs off the frame, so the observation is
    the track's closest point to the frame centre, the object having crossed it at an unknown
    moment in the exposure. ``timing_error`` is a constant shutter-timing bias applied to the
    recorded frame times (the truth moves, the record does not)."""
    rng = np.random.default_rng(seed)
    e = float(exposure.to_value(u.s))
    out = []
    for fs in frame_starts:
        mid = fs + e / 2
        d_mid = topocentric(r0, v0, mid, site, epoch)
        d_a = topocentric(r0, v0, fs, site, epoch)
        d_b = topocentric(r0, v0, fs + e, site, epoch)
        ra, dec = radec_of(d_mid)
        east_a, north_a = tangent_offset(d_a, ra, dec)
        east_b, north_b = tangent_offset(d_b, ra, dec)
        pa = np.arctan2(east_b - east_a, north_b - north_a)
        length = float(np.hypot(east_b - east_a, north_b - north_a))
        recorded_mid = epoch + (mid + float(timing_error.to_value(u.s))) * u.s
        sig = float(noise.to_value(u.rad))
        e_n, n_n = rng.normal(0, sig, 2)
        # observed position: offset by noise in the tangent plane at (ra, dec)
        ra_o = ra + e_n / np.cos(dec)
        dec_o = dec + n_n
        coord = SkyCoord(ra_o * u.rad, dec_o * u.rad, frame="icrs")
        pa_o = pa + rng.normal(0, np.sqrt(2) * sig / length)
        if kind == "point":
            rate = length / e * (1 + rng.normal(0, np.sqrt(2) * sig / length))
            out.append(
                Observation(
                    "point",
                    recorded_mid,
                    coord,
                    noise,
                    timing_sigma,
                    rate * u.rad / u.s,
                    (pa_o * u.rad) % (np.pi * u.rad),
                    1.0,
                    length * u.rad,
                )
            )
        else:
            out.append(
                Observation(
                    "line",
                    recorded_mid,
                    coord,
                    noise,
                    exposure / 2,
                    None,
                    (pa_o * u.rad) % (np.pi * u.rad),
                    1.0,
                    0.9 * u.deg,
                )
            )
    return out
