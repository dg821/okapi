"""GPS interface: UTC time and geodetic position for the nightly calibration."""

from dataclasses import dataclass
from typing import Protocol

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time


class GpsError(RuntimeError):
    """No usable fix (no satellites, no data, corrupt data)."""


@dataclass(frozen=True)
class GpsFix:
    time: Time  # UTC, to about a second (NMEA sentences carry no finer time)
    lat: u.Quantity  # geodetic, +N
    lon: u.Quantity  # +E
    height: u.Quantity  # above the ellipsoid
    satellites: int

    @property
    def site(self) -> EarthLocation:
        return EarthLocation.from_geodetic(lon=self.lon, lat=self.lat, height=self.height)


class GpsProtocol(Protocol):
    def read_fix(self) -> GpsFix:
        """Block until a valid fix arrives or give up. Raises ``GpsError``."""
        ...
