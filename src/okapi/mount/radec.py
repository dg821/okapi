"""RA/Dec slews for mounts that only understand Az/Alt."""

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord

from okapi.clock import Clock
from okapi.coords import AltAzPosition, radec_to_altaz


class AltAzMount:
    """What ``slew_radec_via_altaz`` needs from a mount."""

    def get_altaz(self) -> AltAzPosition: ...

    def slew_altaz(self, target: AltAzPosition) -> None: ...


def slew_radec_via_altaz(
    mount: AltAzMount,
    target: SkyCoord,
    site: EarthLocation,
    clock: Clock,
    assumed_rate: u.Quantity,
) -> None:
    """Aim where ``target`` will be when we arrive; the arrival time is estimated from the
    angular distance and an assumed slew rate. The estimate only needs to be roughly right:
    the centering loop and a second short slew remove the remainder."""
    dest = radec_to_altaz(target, site, clock.now())
    for _ in range(3):
        lead = dest.separation(mount.get_altaz()) / assumed_rate
        dest = radec_to_altaz(target, site, clock.now() + lead)
    mount.slew_altaz(dest)
