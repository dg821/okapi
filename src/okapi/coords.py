"""Coordinate conversions between the sky (RA/Dec, ICRS) and the mount (Az/Alt).

Azimuth is measured from north through east, 0-360 deg. Altitude is 0-90 deg. No atmospheric
refraction is applied: the handset does not model it either, and the plate-solve loop absorbs it.
"""

from dataclasses import dataclass
from typing import cast

import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

# The Pi is offline in the field. Without this astropy tries to download Earth-orientation data
# and fails for dates past its bundled table. The cost is a UT1-UTC error below one second,
# which is a few arcseconds of pointing: far below the mount's own accuracy.
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "ignore"


@dataclass(frozen=True)
class AltAzPosition:
    az: u.Quantity
    alt: u.Quantity

    def __post_init__(self) -> None:
        object.__setattr__(self, "az", self.az.to(u.deg) % (360 * u.deg))
        object.__setattr__(self, "alt", self.alt.to(u.deg))

    def separation(self, other: "AltAzPosition") -> u.Quantity:
        a = SkyCoord(az=self.az, alt=self.alt, frame=AltAz())
        b = SkyCoord(az=other.az, alt=other.alt, frame=AltAz())
        return a.separation(b).to(u.deg)


def site_from_geodetic(lat_deg: float, lon_deg: float, height_m: float = 0.0) -> EarthLocation:
    """Build a site from geodetic latitude (+N), longitude (+E) and height above the ellipsoid."""
    if not -90 <= lat_deg <= 90:
        raise ValueError(f"latitude {lat_deg} outside [-90, 90]")
    if not -180 <= lon_deg <= 360:
        raise ValueError(f"longitude {lon_deg} outside [-180, 360]")
    return EarthLocation.from_geodetic(
        lon=lon_deg * u.deg, lat=lat_deg * u.deg, height=height_m * u.m
    )


def radec_to_altaz(coord: SkyCoord, site: EarthLocation, t: Time) -> AltAzPosition:
    aa = coord.transform_to(AltAz(obstime=t, location=site))
    return AltAzPosition(az=aa.az, alt=aa.alt)


def altaz_to_radec(pos: AltAzPosition, site: EarthLocation, t: Time) -> SkyCoord:
    aa = SkyCoord(az=pos.az, alt=pos.alt, frame=AltAz(obstime=t, location=site))
    return cast(SkyCoord, aa.icrs)
