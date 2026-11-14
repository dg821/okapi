"""Simulated alt-az mount for tests and dry runs.

Models what the real telescope does that the software has to cope with:

* axes slew at a finite rate, so a slew takes time (advanced on the injected clock);
* the mount's reported position differs from where it truly points by a hidden alignment
  error, which ``sync_radec`` removes -- this is what the centering loop has to find;
* when tracking, the true position follows the sky at the sidereal rate.
"""

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.clock import Clock
from okapi.coords import AltAzPosition, altaz_to_radec, radec_to_altaz, wrap180
from okapi.mount.base import MAX_ALTITUDE, MIN_ALTITUDE, MountError, MountLimitError


@dataclass
class _Slew:
    start_mono: float
    end_mono: float
    origin: AltAzPosition
    dest: AltAzPosition


class FakeMount:
    def __init__(
        self,
        clock: Clock,
        *,
        alignment_error: tuple[u.Quantity, u.Quantity] = (0 * u.deg, 0 * u.deg),
        slew_rate: u.Quantity = 3 * u.deg / u.s,
    ) -> None:
        self._clock = clock
        self._slew_rate = slew_rate
        self._offset_az, self._offset_alt = alignment_error  # reported = true + offset
        self._true = AltAzPosition(0 * u.deg, 30 * u.deg)
        self._slew: _Slew | None = None
        self._tracking = False
        self._target: SkyCoord | None = None
        self._site: EarthLocation | None = None
        self.mount_time: Time | None = None

    # -- setup ---------------------------------------------------------------------------
    def set_site(self, site: EarthLocation) -> None:
        self._site = site

    def set_time(self, t: Time) -> None:
        self.mount_time = t

    def _require_site(self) -> EarthLocation:
        if self._site is None:
            raise MountError("site not set; call set_site first")
        return self._site

    # -- true position (simulation only; public so the fake solver and tests can read it) --
    def _settle(self) -> None:
        """Finish a slew whose time has elapsed."""
        s = self._slew
        if s is not None and self._clock.monotonic() >= s.end_mono:
            self._true = s.dest
            self._slew = None
            if self._tracking:
                self._lock_target()

    def _lock_target(self) -> None:
        self._target = altaz_to_radec(self._true, self._require_site(), self._clock.now())

    def true_altaz(self) -> AltAzPosition:
        self._settle()
        s = self._slew
        if s is not None:
            frac = (self._clock.monotonic() - s.start_mono) / (s.end_mono - s.start_mono)
            d_az = wrap180(s.dest.az - s.origin.az)
            return AltAzPosition(
                s.origin.az + frac * d_az, s.origin.alt + frac * (s.dest.alt - s.origin.alt)
            )
        if self._tracking and self._target is not None:
            return radec_to_altaz(self._target, self._require_site(), self._clock.now())
        return self._true

    # -- queries -------------------------------------------------------------------------
    def get_altaz(self) -> AltAzPosition:
        t = self.true_altaz()
        return AltAzPosition(t.az + self._offset_az, t.alt + self._offset_alt)

    def true_radec(self) -> SkyCoord:
        """Where the telescope truly points. Simulation only: a real mount cannot tell you."""
        return altaz_to_radec(self.true_altaz(), self._require_site(), self._clock.now())

    def get_radec(self) -> SkyCoord:
        return altaz_to_radec(self.get_altaz(), self._require_site(), self._clock.now())

    def is_slewing(self) -> bool:
        self._settle()
        return self._slew is not None

    @property
    def is_tracking(self) -> bool:
        return self._tracking

    # -- motion --------------------------------------------------------------------------
    @staticmethod
    def _check_limits(pos: AltAzPosition) -> None:
        if not MIN_ALTITUDE <= pos.alt <= MAX_ALTITUDE:
            raise MountLimitError(
                f"altitude {pos.alt:.1f} outside [{MIN_ALTITUDE}, {MAX_ALTITUDE}]"
            )

    def _begin_slew(self, true_dest: AltAzPosition) -> None:
        """Start moving the true position to ``true_dest``."""
        origin = self.true_altaz()
        duration = (
            max(abs(wrap180(true_dest.az - origin.az)), abs(true_dest.alt - origin.alt))
            / self._slew_rate
        )
        now = self._clock.monotonic()
        self._true = origin
        self._slew = _Slew(now, now + max(float(duration.to_value(u.s)), 1e-6), origin, true_dest)

    def _to_true(self, reported: AltAzPosition) -> AltAzPosition:
        return AltAzPosition(reported.az - self._offset_az, reported.alt - self._offset_alt)

    def slew_altaz(self, target: AltAzPosition) -> None:
        self._check_limits(target)
        self._tracking_hold()
        self._begin_slew(self._to_true(target))

    def _tracking_hold(self) -> None:
        # a slew supersedes the tracked target; it is re-locked on arrival if tracking is on
        self._true = self.true_altaz()
        self._target = None

    def slew_radec(self, target: SkyCoord) -> None:
        site = self._require_site()
        # The sky moves while we slew, so aim where the target will be on arrival.
        arrival = self._clock.now()
        dest = radec_to_altaz(target, site, arrival)
        for _ in range(3):
            self._check_limits(dest)
            origin = self.true_altaz()
            true_dest = self._to_true(dest)
            dur = (
                max(abs(wrap180(true_dest.az - origin.az)), abs(true_dest.alt - origin.alt))
                / self._slew_rate
            )
            arrival = self._clock.now() + dur
            dest = radec_to_altaz(target, site, arrival)
        self._check_limits(dest)
        self._tracking_hold()
        self._begin_slew(self._to_true(dest))

    def nudge(self, d_az: u.Quantity, d_alt: u.Quantity) -> None:
        cur = self.get_altaz()
        self.slew_altaz(AltAzPosition(cur.az + d_az, cur.alt + d_alt))

    def wait_until_settled(self, timeout: u.Quantity) -> None:
        limit = self._clock.monotonic() + float(timeout.to_value(u.s))
        while self.is_slewing():
            if self._clock.monotonic() >= limit:
                raise MountError("timed out waiting for the slew to finish")
            self._clock.sleep(0.1)

    def abort(self) -> None:
        self._true = self.true_altaz()
        self._slew = None
        if self._tracking:
            self._lock_target()

    # -- tracking and alignment ------------------------------------------------------------
    def start_tracking(self) -> None:
        self._require_site()
        self._true = self.true_altaz()
        self._tracking = True
        if self._slew is None:
            self._lock_target()

    def stop_tracking(self) -> None:
        self._true = self.true_altaz()
        self._tracking = False
        self._target = None

    def sync_radec(self, target: SkyCoord) -> None:
        site = self._require_site()
        claimed = radec_to_altaz(target, site, self._clock.now())
        true = self.true_altaz()
        self._offset_az = wrap180(claimed.az - true.az)
        self._offset_alt = claimed.alt - true.alt
