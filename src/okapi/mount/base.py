"""Mount interface. All sequencing code depends on this, never on a serial port directly."""

from typing import Protocol, runtime_checkable

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.coords import AltAzPosition
from okapi.pointing import PointingModel


class MountError(RuntimeError):
    """The mount refused or failed a command."""


class MountLimitError(MountError):
    """The requested position is outside the mount's safe range."""


MIN_ALTITUDE = 5 * u.deg
MAX_ALTITUDE = 89 * u.deg


@runtime_checkable
class PointingModelAware(Protocol):
    """Mounts that compute their own tracking rates need to know the pointing model."""

    def set_pointing_model(self, model: PointingModel) -> None: ...


class MountProtocol(Protocol):
    def set_site(self, site: EarthLocation) -> None:
        """Send geodetic location to the mount."""
        ...

    def set_time(self, t: Time) -> None:
        """Send UTC date and time to the mount."""
        ...

    def get_altaz(self) -> AltAzPosition:
        """Where the mount reports it is pointing (not necessarily where it truly points)."""
        ...

    def get_radec(self) -> SkyCoord:
        """Same, as ICRS RA/Dec."""
        ...

    def slew_altaz(self, target: AltAzPosition) -> None:
        """Start a slew and return at once. Raises ``MountLimitError`` outside altitude limits."""
        ...

    def slew_radec(self, target: SkyCoord) -> None:
        """Start a slew to an ICRS position and return at once."""
        ...

    def nudge(self, d_az: u.Quantity, d_alt: u.Quantity) -> None:
        """Start a small relative move (used by the centering loop)."""
        ...

    def is_slewing(self) -> bool: ...

    def wait_until_settled(self, timeout: u.Quantity) -> None:
        """Block until a slew finishes. Raises ``MountError`` on timeout."""
        ...

    def abort(self) -> None:
        """Stop all motion immediately."""
        ...

    def start_tracking(self) -> None:
        """Follow the sidereal rate from the current position."""
        ...

    def stop_tracking(self) -> None: ...

    @property
    def is_tracking(self) -> bool: ...

    def sync_radec(self, target: SkyCoord) -> None:
        """Tell the mount it is *actually* pointing at ``target`` (one-star alignment point)."""
        ...
