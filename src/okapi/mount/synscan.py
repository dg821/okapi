"""SynScan hand-control driver (Orion SynScan AZ and other Sky-Watcher SynScan handsets).

Design (docs/design/synscan-protocol-notes.md, ADR 0004):

* **Az/Alt only.** The handset's RA/Dec is meaningless until it has been aligned, and it cannot be
  aligned over serial. RA/Dec is computed here, from Az/Alt, the site and the clock.
* **The Pi drives tracking.** Handset tracking is switched off; the Pi sends variable-rate slews
  on both axes, re-aimed every ``track_period`` from the measured position (closed loop), aimed
  through the pointing model. If the measured position strays from the aim by more than
  ``MAX_TRACK_ERROR`` the motors are stopped and ``MountError`` is raised: a backwards axis sign
  or a stuck motor cannot run away.
* **Nothing is sent blindly.** Every command waits for its reply (up to 5.5 s, the document's
  worst case). Read-only queries are retried; commands that move or configure the mount are not.

Known risk: the handset has no dead-man timer, so axis rates keep running if the Pi dies
mid-track. Use the mount as a context manager, or call ``close()``, so the rates are zeroed.
"""

import contextlib
import threading
from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.clock import Clock
from okapi.coords import (
    AltAzPosition,
    altaz_to_radec,
    radec_to_altaz,
    wrap180,
)
from okapi.mount.base import MAX_ALTITUDE, MIN_ALTITUDE, MountError, MountLimitError
from okapi.mount.radec import slew_radec_via_altaz
from okapi.mount.synscan_protocol import (
    ALT_AXIS,
    AZ_AXIS,
    ProtocolError,
    Version,
    decode_location,
    decode_time,
    encode_location,
    encode_position_command,
    encode_rate_slew,
    encode_time,
    parse_position,
    parse_version,
    signed,
)
from okapi.mount.transport import Transport, TransportTimeoutError
from okapi.pointing import PointingModel

REPLY_TIMEOUT_S = 5.5  # the document: be prepared to wait up to 5 s
MAX_TRACK_ERROR = 0.25 * u.deg  # farther than this from the aim and tracking is declared lost
MAX_TRACK_RATE = 1.0 * u.deg / u.s
MIN_RATE_FIRMWARE = {3: Version(3, 3, 0), 4: Version(4, 1, 1)}  # variable-rate slew
RECOMMENDED_FIRMWARE = {3: Version(3, 38, 0), 4: Version(4, 38, 6)}  # protocol v3.3 baseline
SETTLE_POLL_S = 0.25


@dataclass(frozen=True)
class HandsetInfo:
    version: Version
    model: int
    aligned: bool

    @property
    def model_name(self) -> str:
        if 128 <= self.model <= 143:
            return "AZ GOTO series"
        if 144 <= self.model <= 159:
            return "DOB GOTO series"
        return f"model {self.model} (not an alt-az series: is this the right handset?)"


@dataclass(frozen=True)
class HandsetReport:
    """Everything we can read from the handset without changing anything."""

    info: HandsetInfo
    altaz: AltAzPosition
    tracking_mode: int
    location: tuple[float, float]  # (lat, lon) degrees
    time_utc: Time | None  # None if the handset clock was never set; else per our reading of
    # the document's offset/DST convention


class SynScanMount:
    def __init__(
        self,
        transport: Transport,
        clock: Clock,
        *,
        assumed_slew_rate: u.Quantity = 2 * u.deg / u.s,
        track_period: u.Quantity = 1 * u.s,
        background_tracking: bool = False,
        az_rate_sign: int = 1,
        alt_rate_sign: int = 1,
    ) -> None:
        """``background_tracking`` starts a thread that re-aims every ``track_period``; leave it
        off in simulation and call ``track_step`` yourself. ``az_rate_sign`` / ``alt_rate_sign``
        flip an axis whose positive rate turns out to move the wrong way (see the hardware test)."""
        self._t = transport
        self._clock = clock
        self._rate = assumed_slew_rate
        self._period = track_period
        self._background = background_tracking
        self._signs = {AZ_AXIS: az_rate_sign, ALT_AXIS: alt_rate_sign}
        self._lock = threading.RLock()
        self._site: EarthLocation | None = None
        self._model = PointingModel.identity()
        self._tracking = False
        self._resume = False  # a goto is running; start tracking when it ends
        self._target: SkyCoord | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.tracking_error: MountError | None = None  # set if the background thread gave up
        self.warnings: list[str] = []
        self.info = self._handshake()

    # -- low level ---------------------------------------------------------------------------
    def _exchange(self, cmd: bytes, *, length: int | None = None, retries: int = 0) -> bytes:
        with self._lock:
            for _ in range(retries + 1):
                self._t.write(cmd)
                try:
                    return self._t.read_reply(REPLY_TIMEOUT_S, length)
                except TransportTimeoutError:
                    continue
        raise MountError(f"hand control did not answer {cmd[:1]!r}")

    def _ack(self, cmd: bytes) -> None:
        reply = self._exchange(cmd)
        if reply != b"#":
            raise ProtocolError(f"expected '#' after {cmd[:1]!r}, got {reply!r}")

    def _handshake(self) -> HandsetInfo:
        echo = self._exchange(b"Kx", length=2, retries=2)
        if echo != b"x#":
            raise ProtocolError(f"echo test failed: {echo!r}")
        version = parse_version(self._exchange(b"V", retries=2))
        model = self._exchange(b"m", length=2, retries=2)[0]
        aligned = self._exchange(b"J", length=2, retries=2)[0] == 1
        info = HandsetInfo(version, model, aligned)
        baseline = RECOMMENDED_FIRMWARE.get(version.major)
        if baseline is not None and version < baseline:
            self.warnings.append(
                f"firmware {version} is older than {baseline}, which protocol v3.3 describes; "
                "some commands may differ"
            )
        if not 128 <= model <= 159:
            self.warnings.append(info.model_name)
        return info

    def _require_site(self) -> EarthLocation:
        if self._site is None:
            raise MountError("site not set; call set_site first")
        return self._site

    def read_report(self) -> HandsetReport:
        """Read-only snapshot for bring-up (``okapi mount-info``): sends no motion or settings."""
        mode = self._exchange(b"t", length=2, retries=2)[0]
        location = decode_location(self._exchange(b"w", length=9, retries=2))
        try:
            time_utc = decode_time(self._exchange(b"h", length=9, retries=2)).to_utc()
        except ProtocolError:
            time_utc = None  # never set: a finding to report, not a failure
        return HandsetReport(self.info, self.get_altaz(), mode, location, time_utc)

    # -- setup -------------------------------------------------------------------------------
    def set_pointing_model(self, model: PointingModel) -> None:
        with self._lock:
            self._model = model
            if self._tracking and self._target is not None and not self._slew_in_progress():
                self._lock_target()  # re-aim the running track through the new model

    def set_site(self, site: EarthLocation) -> None:
        lat, lon = float(site.lat.to_value(u.deg)), float(site.lon.to_value(u.deg))
        self._ack(encode_location(lat, lon))
        got_lat, got_lon = decode_location(self._exchange(b"w", length=9, retries=2))
        if abs(got_lat - lat) > 2 / 3600 or abs(wrap180((got_lon - lon) * u.deg).value) > 2 / 3600:
            raise MountError(
                f"location read-back mismatch: sent {lat:.5f},{lon:.5f}, "
                f"got {got_lat:.5f},{got_lon:.5f}"
            )
        self._site = site

    def set_time(self, t: Time) -> None:
        self._ack(encode_time(t))
        back = decode_time(self._exchange(b"h", length=9, retries=2)).to_utc()
        if abs(float((back - t).to_value(u.s))) > 3.0:
            raise MountError(f"time read-back mismatch: sent {t.utc.isot}, got {back.isot}")

    # -- position ----------------------------------------------------------------------------
    def get_altaz(self) -> AltAzPosition:
        az, alt = parse_position(self._exchange(b"z", retries=2))
        return AltAzPosition(az * u.deg, signed(alt) * u.deg)

    def get_radec(self) -> SkyCoord:
        """Computed from Az/Alt, assuming the handset's frame is the true horizon frame."""
        return altaz_to_radec(self.get_altaz(), self._require_site(), self._clock.now())

    # -- motion ------------------------------------------------------------------------------
    @staticmethod
    def _check_limits(pos: AltAzPosition) -> None:
        if not MIN_ALTITUDE <= pos.alt <= MAX_ALTITUDE:
            raise MountLimitError(
                f"altitude {pos.alt:.1f} outside [{MIN_ALTITUDE}, {MAX_ALTITUDE}]"
            )

    def slew_altaz(self, target: AltAzPosition) -> None:
        self._check_limits(target)
        with self._lock:
            self._stop_rates()
            self._ack(
                encode_position_command("b", target.az.to_value(u.deg), target.alt.to_value(u.deg))
            )
            self._resume = self._tracking

    def slew_radec(self, target: SkyCoord) -> None:
        slew_radec_via_altaz(self, target, self._require_site(), self._clock, self._rate)

    def nudge(self, d_az: u.Quantity, d_alt: u.Quantity) -> None:
        cur = self.get_altaz()
        self.slew_altaz(AltAzPosition(cur.az + d_az, cur.alt + d_alt))

    def _slew_in_progress(self) -> bool:
        reply = self._exchange(b"L", retries=2)
        if reply not in (b"0#", b"1#"):
            raise ProtocolError(f"bad goto-in-progress reply {reply!r}")
        return reply == b"1#"

    def is_slewing(self) -> bool:
        with self._lock:
            if self._slew_in_progress():
                return True
            if self._resume:
                self._resume = False
                self._lock_target()
            return False

    def wait_until_settled(self, timeout: u.Quantity) -> None:
        limit = self._clock.monotonic() + float(timeout.to_value(u.s))
        while self.is_slewing():
            if self._clock.monotonic() >= limit:
                raise MountError("timed out waiting for the slew to finish")
            self._clock.sleep(SETTLE_POLL_S)

    def abort(self) -> None:
        with self._lock:
            self._ack(b"M")
            self._stop_rates()
            if self._tracking:
                self._lock_target()

    def sync_radec(self, target: SkyCoord) -> None:
        raise MountError(
            "sync is not used: the handset cannot be aligned over serial and the Pi's pointing "
            "model corrects pointing instead (ADR 0002)"
        )

    # -- tracking (Pi-driven) ----------------------------------------------------------------
    @property
    def is_tracking(self) -> bool:
        return self._tracking

    def _require_rate_firmware(self) -> None:
        minimum = MIN_RATE_FIRMWARE.get(self.info.version.major)
        if minimum is not None and self.info.version < minimum:
            raise MountError(
                f"firmware {self.info.version} cannot do variable-rate slews (needs {minimum}); "
                "update the hand control firmware"
            )

    def start_tracking(self) -> None:
        with self._lock:
            self._require_site()
            self._require_rate_firmware()
            self._ack(b"T\x00")  # handset tracking off: we drive the axes ourselves
            self._tracking = True
            self.tracking_error = None
            if self._slew_in_progress():
                self._resume = True
            else:
                self._lock_target()
            if self._background and (self._thread is None or not self._thread.is_alive()):
                self._stop.clear()
                self._thread = threading.Thread(target=self._track_loop, daemon=True)
                self._thread.start()

    def stop_tracking(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=float(self._period.to_value(u.s)) * 3 + 1)
        with self._lock:
            self._tracking = False
            self._resume = False
            self._target = None
            self._stop_rates()

    def _lock_target(self) -> None:
        """Track whatever sky position the telescope is pointing at now."""
        reported = self.get_altaz()
        true = self._model.to_true(reported)
        self._target = altaz_to_radec(true, self._require_site(), self._clock.now())
        self.track_step()

    def track_step(self) -> None:
        """Re-aim both axes so the telescope reaches where the target will be one period from now.

        Closed loop on the measured position, so rate quantization (0.25 arcsec/s) and model
        errors do not accumulate. Called every ``track_period`` by the background thread, or by
        hand in simulation."""
        with self._lock:
            if not self._tracking or self._target is None:
                return
            site, now = self._require_site(), self._clock.now()
            measured = self.get_altaz()
            aim_now = self._model.to_reported(radec_to_altaz(self._target, site, now))
            if aim_now.separation(measured) > MAX_TRACK_ERROR:
                self._tracking = False
                self._stop_rates()
                raise MountError(
                    f"tracking lost: {measured.separation(aim_now):.2f} from where the target "
                    "should be (axis sign backwards? motor stuck?). Motors stopped."
                )
            period = self._period
            aim = self._model.to_reported(radec_to_altaz(self._target, site, now + period))
            az_rate = wrap180(aim.az - measured.az) / period
            alt_rate = (aim.alt - measured.alt) / period
            if max(abs(az_rate), abs(alt_rate)) > MAX_TRACK_RATE:
                self._tracking = False
                self._stop_rates()
                raise MountError(
                    "tracking rate too high (target near the zenith?). Motors stopped."
                )
            self._set_rates(az_rate.to_value(u.arcsec / u.s), alt_rate.to_value(u.arcsec / u.s))

    def _set_rates(self, az_arcsec_s: float, alt_arcsec_s: float) -> None:
        self._ack(encode_rate_slew(AZ_AXIS, self._signs[AZ_AXIS] * az_arcsec_s))
        self._ack(encode_rate_slew(ALT_AXIS, self._signs[ALT_AXIS] * alt_arcsec_s))

    def _stop_rates(self) -> None:
        self._set_rates(0.0, 0.0)

    def _track_loop(self) -> None:
        while not self._stop.is_set():
            self._clock.sleep(float(self._period.to_value(u.s)))
            if self._stop.is_set():
                return
            try:
                with self._lock:
                    if self._tracking and not self._slew_in_progress():
                        self.track_step()
            except MountError as e:
                self.tracking_error = e
                self._tracking = False
                with contextlib.suppress(MountError):  # link down: nothing more we can do
                    self._stop_rates()
                return

    # -- lifecycle ---------------------------------------------------------------------------
    def close(self) -> None:
        """Zero the axis rates (they would otherwise keep running) and close the port."""
        try:
            self.stop_tracking()
        finally:
            self._t.close()

    def __enter__(self) -> "SynScanMount":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
