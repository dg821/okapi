"""A simulated SynScan hand control: speaks the serial protocol byte for byte.

It sits on top of ``FakeMount`` for the physics (slewing, tilt error, rate-driven motion), and
turns protocol commands into calls on it. Use it as the ``Transport`` of ``SynScanMount`` to test
the driver, framing, and timeouts without hardware. Behaviors the document leaves open are
configurable so tests can exercise both possibilities (see ``tracking_when_unaligned``).
"""

import contextlib

import astropy.units as u
from astropy.time import Time, TimeDelta

from okapi.clock import Clock
from okapi.coords import AltAzPosition, site_from_geodetic
from okapi.mount.base import MountError
from okapi.mount.fake import FakeMount
from okapi.mount.synscan_protocol import (
    ALT_AXIS,
    AZ_AXIS,
    HandsetTime,
    ProtocolError,
    Version,
    command_length,
    decode_angle,
    decode_location,
    decode_rate_slew,
    decode_time,
    encode_angle,
    signed,
)
from okapi.mount.transport import TransportTimeoutError

DEFAULT_FIRMWARE = Version(3, 38, 0)  # the first version the v3.3 document applies to


class FakeHandset:
    def __init__(
        self,
        mount: FakeMount,
        clock: Clock,
        *,
        firmware: Version = DEFAULT_FIRMWARE,
        model: int = 145,  # 144-159 is the DOB GOTO series
        aligned: bool = False,
        tracking_when_unaligned: bool = False,
        silent: bool = False,
    ) -> None:
        self.mount = mount
        self._clock = clock
        self.firmware = firmware
        self.model = model
        self.aligned = aligned
        self.tracking_when_unaligned = tracking_when_unaligned
        self.silent = silent  # never answers: for timeout tests
        self.tracking_mode = 0
        self.commands: list[bytes] = []  # everything received, for assertions
        self._in = bytearray()
        self._out = bytearray()
        self._location = bytes(8)
        self._time: tuple[Time, HandsetTime] | None = None
        self._time_set_mono = 0.0
        self._rates = {AZ_AXIS: 0.0, ALT_AXIS: 0.0}  # signed arcsec/s per axis
        self.closed = False

    # -- Transport ---------------------------------------------------------------------------
    def write(self, data: bytes) -> None:
        self._out.clear()  # like the real transport: stale replies are dropped
        self._in.extend(data)
        while self._in:
            n = command_length(self._in[0])
            if len(self._in) < n:
                break
            cmd = bytes(self._in[:n])
            del self._in[:n]
            self.commands.append(cmd)
            if not self.silent:
                self._out.extend(self._handle(cmd))

    def read_reply(self, timeout: float, length: int | None = None) -> bytes:
        if length:
            if len(self._out) < length:
                raise TransportTimeoutError("simulated handset: no reply")
            reply = bytes(self._out[:length])
        else:
            end = self._out.find(b"#")
            if end < 0:
                raise TransportTimeoutError("simulated handset: no reply")
            reply = bytes(self._out[: end + 1])
        del self._out[: len(reply)]
        return reply

    def close(self) -> None:
        self.closed = True

    # -- command handling --------------------------------------------------------------------
    def _handle(self, cmd: bytes) -> bytes:
        letter = chr(cmd[0])
        if letter == "K":
            return cmd[1:2] + b"#"
        if letter == "V":
            f = self.firmware
            return f"{f.major:02X}{f.minor:02X}{f.patch:02X}#".encode()
        if letter == "m":
            return bytes([self.model]) + b"#"
        if letter == "J":
            return bytes([int(self.aligned)]) + b"#"
        if letter in "zZ":
            return self._position_reply()
        if letter in "bB":
            return self._goto(cmd)
        if letter == "L":
            return (b"1" if self.mount.is_slewing() else b"0") + b"#"
        if letter == "M":
            self.mount.abort()
            return b"#"
        if letter == "t":
            return bytes([self.tracking_mode]) + b"#"
        if letter == "T":
            return self._set_tracking(cmd[1])
        if letter == "W":
            self._location = cmd[1:9]
            lat, lon = decode_location(self._location + b"#")
            self.mount.set_site(site_from_geodetic(lat, lon, 0.0))
            return b"#"
        if letter == "w":
            return self._location + b"#"
        if letter == "H":
            return self._set_time(cmd)
        if letter == "h":
            return self._get_time()
        if letter == "P":
            return self._rate_command(cmd)
        return b""  # unknown commands get no answer, like a deaf handset

    def _position_reply(self) -> bytes:
        pos = self.mount.get_altaz()
        az, alt = encode_angle(pos.az.to_value(u.deg)), encode_angle(pos.alt.to_value(u.deg))
        return f"{az},{alt}#".encode()

    def _goto(self, cmd: bytes) -> bytes:
        # Out of range or malformed: the handset just does not move.
        with contextlib.suppress(UnicodeDecodeError, ValueError, ProtocolError, MountError):
            first, second = cmd[1:].decode("ascii").split(",")
            az, alt = decode_angle(first), signed(decode_angle(second))
            self.mount.slew_altaz(AltAzPosition(az * u.deg, alt * u.deg))
        return b"#"

    def _set_tracking(self, mode: int) -> bytes:
        self.tracking_mode = mode
        if mode == 0:
            self.mount.stop_tracking()
        elif self.aligned or self.tracking_when_unaligned:
            self.mount.start_tracking()
        return b"#"

    def _set_time(self, cmd: bytes) -> bytes:
        fields = decode_time(cmd[1:9] + b"#")
        utc = fields.to_utc()
        self._time = (utc, fields)
        self._time_set_mono = self._clock.monotonic()
        self.mount.set_time(utc)
        return b"#"

    def _get_time(self) -> bytes:
        if self._time is None:
            return bytes(8) + b"#"
        utc0, fields = self._time
        utc = utc0 + TimeDelta((self._clock.monotonic() - self._time_set_mono) * u.s)
        local = utc + TimeDelta((fields.offset_h + int(fields.dst)) * u.hour)
        dt = local.datetime
        offset = fields.offset_h % 256
        return (
            bytes(
                [
                    dt.hour,
                    dt.minute,
                    dt.second,
                    dt.month,
                    dt.day,
                    dt.year % 100,
                    offset,
                    int(fields.dst),
                ]
            )
            + b"#"
        )

    def _rate_command(self, cmd: bytes) -> bytes:
        try:
            axis, rate = decode_rate_slew(cmd)
        except ProtocolError:
            return b"#"  # fixed-rate and other P commands are accepted and ignored
        self._rates[axis] = rate
        with contextlib.suppress(MountError):  # busy with a goto: the handset ignores it
            self.mount.set_axis_rates(
                self._rates[AZ_AXIS] / 3600.0 * u.deg / u.s,
                self._rates[ALT_AXIS] / 3600.0 * u.deg / u.s,
            )
        return b"#"
