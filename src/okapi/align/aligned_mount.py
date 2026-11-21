"""A mount wrapped with a Pi-side pointing model, so callers see true positions.

Everything that takes or returns a position (slews, reads) is translated; everything else is
passed straight through. Sequencing code uses it exactly like a raw mount.
"""

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.clock import Clock
from okapi.coords import AltAzPosition, altaz_to_radec
from okapi.mount.base import MountError, MountProtocol, PointingModelAware
from okapi.mount.radec import slew_radec_via_altaz
from okapi.pointing import PointingModel


class AlignedMount:
    def __init__(
        self,
        inner: MountProtocol,
        model: PointingModel,
        *,
        site: EarthLocation,
        clock: Clock,
        assumed_slew_rate: u.Quantity = 2 * u.deg / u.s,
    ) -> None:
        self._inner = inner
        self._model = model
        self._site = site
        self._clock = clock
        self._rate = assumed_slew_rate
        # A mount that drives its own tracking (the SynScan driver) needs the model to aim right.
        if isinstance(inner, PointingModelAware):
            inner.set_pointing_model(model)

    # -- positions are translated through the model ------------------------------------
    def get_altaz(self) -> AltAzPosition:
        return self._model.to_true(self._inner.get_altaz())

    def get_radec(self) -> SkyCoord:
        return altaz_to_radec(self.get_altaz(), self._site, self._clock.now())

    def slew_altaz(self, target: AltAzPosition) -> None:
        self._inner.slew_altaz(self._model.to_reported(target))

    def slew_radec(self, target: SkyCoord) -> None:
        slew_radec_via_altaz(self, target, self._site, self._clock, self._rate)

    def sync_radec(self, target: SkyCoord) -> None:
        raise MountError(
            "sync would shift the mount's frame under the pointing model; refit the model instead"
        )

    # -- everything else passes through --------------------------------------------------
    def set_site(self, site: EarthLocation) -> None:
        self._site = site
        self._inner.set_site(site)

    def set_time(self, t: Time) -> None:
        self._inner.set_time(t)

    def nudge(self, d_az: u.Quantity, d_alt: u.Quantity) -> None:
        self._inner.nudge(d_az, d_alt)

    def is_slewing(self) -> bool:
        return self._inner.is_slewing()

    def wait_until_settled(self, timeout: u.Quantity) -> None:
        self._inner.wait_until_settled(timeout)

    def abort(self) -> None:
        self._inner.abort()

    def start_tracking(self) -> None:
        self._inner.start_tracking()

    def stop_tracking(self) -> None:
        self._inner.stop_tracking()

    @property
    def is_tracking(self) -> bool:
        return self._inner.is_tracking
