"""Things to point at. A target predicts where it appears in the sky from the site at a time;
satellites also have a motion, which is what identification and aiming rely on."""

from dataclasses import dataclass
from typing import Any, Protocol, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, TEME, CartesianRepresentation, EarthLocation, SkyCoord
from astropy.time import Time
from sgp4.api import Satrec

from okapi.orbits.geometry import observer_position, radec_of, topocentric
from okapi.orbits.simulate import geo_state


class TargetError(RuntimeError):
    """A target's position could not be predicted (decayed satellite, bad element set)."""


class Target(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_satellite(self) -> bool: ...

    def predict(self, site: EarthLocation, t: Time) -> SkyCoord:
        """Topocentric ICRS direction at time ``t``."""
        ...


def _coord(ra: float, dec: float) -> SkyCoord:
    return SkyCoord(ra * u.rad, dec * u.rad, frame="icrs")


@dataclass(frozen=True)
class FixedTarget:
    """A star or other fixed sky position (ICRS)."""

    name: str
    coord: SkyCoord
    is_satellite: bool = False

    def predict(self, site: EarthLocation, t: Time) -> SkyCoord:
        return self.coord


@dataclass(frozen=True)
class GeoSlotTarget:
    """A nominal geostationary slot: a circular equatorial orbit that stays over one longitude."""

    name: str
    longitude: u.Quantity
    is_satellite: bool = True

    def predict(self, site: EarthLocation, t: Time) -> SkyCoord:
        r, _ = geo_state(float(self.longitude.to_value(u.deg)), t)
        ra, dec = radec_of(r - observer_position(site, t))
        return _coord(ra, dec)


@dataclass(frozen=True)
class StateTarget:
    """A two-body orbit from a state vector (km, km/s, GCRS) at ``epoch``: for example a fitted
    orbit, to re-observe the same object another night (valid for hours, not days)."""

    name: str
    r: np.ndarray
    v: np.ndarray
    epoch: Time
    is_satellite: bool = True

    def predict(self, site: EarthLocation, t: Time) -> SkyCoord:
        dt = float((t - self.epoch).to_value(u.s))
        ra, dec = radec_of(topocentric(self.r, self.v, dt, site, self.epoch))
        return _coord(ra, dec)


def _tle_checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line) % 10


class TleTarget:
    """A satellite from a two-line element set, propagated with SGP4."""

    is_satellite = True

    def __init__(self, name: str, line1: str, line2: str) -> None:
        self.name = name
        self.line1, self.line2 = line1.rstrip(), line2.rstrip()
        if not (self.line1.startswith("1 ") and self.line2.startswith("2 ")):
            raise TargetError(f"{name}: lines must start with '1 ' and '2 '")
        for line in (self.line1, self.line2):
            if len(line) != 69:
                raise TargetError(f"{name}: a TLE line is 69 characters, got {len(line)}")
            if _tle_checksum(line[:68]) != int(line[68]):
                raise TargetError(f"{name}: checksum mismatch in line {line[0]}")
        try:
            self._sat = Satrec.twoline2rv(self.line1, self.line2)
        except Exception as e:  # sgp4 raises ValueError for malformed lines
            raise TargetError(f"{name}: unreadable element set: {e}") from e

    def predict(self, site: EarthLocation, t: Time) -> SkyCoord:
        utc = t.utc
        err, r, _v = self._sat.sgp4(float(utc.jd1), float(utc.jd2))
        if err != 0:
            raise TargetError(f"{self.name}: SGP4 error code {err} (decayed or bad elements)")
        teme = TEME(CartesianRepresentation(*(x * u.km for x in r)), obstime=t)
        xyz = cast(Any, teme.transform_to(GCRS(obstime=t))).cartesian.xyz.to_value(u.km)
        ra, dec = radec_of(np.asarray(xyz, dtype=float) - observer_position(site, t))
        return _coord(ra, dec)


def parse_tles(text: str) -> list[TleTarget]:
    """Two-line or three-line (name first) element sets, blank lines ignored."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out: list[TleTarget] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = f"NORAD {lines[i][2:7].strip()}"
            out.append(TleTarget(name, lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append(TleTarget(lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            raise TargetError(f"unexpected line in element sets: {lines[i][:40]!r}")
    return out
