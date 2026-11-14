"""Simulated GPS for tests and dry runs."""

import astropy.units as u
from astropy.coordinates import EarthLocation

from okapi.clock import Clock
from okapi.sensors.base import GpsError, GpsFix


class FakeGps:
    def __init__(
        self,
        clock: Clock,
        site: EarthLocation,
        *,
        clock_error: u.Quantity = 0 * u.s,
        fail_first: int = 0,
        always_fail: bool = False,
        satellites: int = 9,
    ) -> None:
        """``clock_error`` is how far the *system* clock is ahead of GPS time."""
        self._clock = clock
        self._site = site
        self._clock_error = clock_error
        self._fail_first = fail_first
        self._always_fail = always_fail
        self._satellites = satellites
        self.reads = 0

    def read_fix(self) -> GpsFix:
        self.reads += 1
        if self._always_fail or self.reads <= self._fail_first:
            raise GpsError("simulated: no fix")
        return GpsFix(
            time=self._clock.now() - self._clock_error,
            lat=self._site.lat,
            lon=self._site.lon,
            height=self._site.height,
            satellites=self._satellites,
        )
