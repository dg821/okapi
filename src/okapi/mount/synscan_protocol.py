"""Encoding and decoding for the SynScan hand-control serial protocol (v3.3).

Pure functions, no I/O. See docs/design/synscan-protocol-notes.md for the command table and what
is and is not verified. Angles travel as hexadecimal fractions of a full turn; location, time and
slew rates travel as raw binary bytes.
"""

from dataclasses import dataclass

import astropy.units as u
from astropy.time import Time, TimeDelta

from okapi.mount.base import MountError

AZ_AXIS = 16
ALT_AXIS = 17
FULL_TURN_24 = 1 << 24
FULL_TURN_16 = 1 << 16


class ProtocolError(MountError):
    """The handset sent something that does not match the protocol."""


# -- angles ------------------------------------------------------------------------------------
def encode_angle(deg: float) -> str:
    """Precise 8-hex-digit angle: 24 bits of a turn plus two ignored ``00`` digits."""
    n = round((deg % 360.0) / 360.0 * FULL_TURN_24) % FULL_TURN_24
    return f"{n:06X}00"


def decode_angle(field: str) -> float:
    """Degrees in [0, 360) from a 4-digit (standard) or 8-digit (precise) hex field."""
    try:
        if len(field) == 8:
            return int(field[:6], 16) / FULL_TURN_24 * 360.0
        if len(field) == 4:
            return int(field, 16) / FULL_TURN_16 * 360.0
    except ValueError:
        pass
    raise ProtocolError(f"bad angle field {field!r}")


def signed(deg: float) -> float:
    """Map [0, 360) to [-180, 180): negative altitudes arrive as large fractions (assumed)."""
    return deg - 360.0 if deg >= 180.0 else deg


def encode_position_command(letter: str, first_deg: float, second_deg: float) -> bytes:
    """e.g. ``b`` + azimuth + ``,`` + altitude."""
    return f"{letter}{encode_angle(first_deg)},{encode_angle(second_deg)}".encode("ascii")


def parse_position(reply: bytes) -> tuple[float, float]:
    """``b'12AB0500,40000500#'`` -> (az deg in [0,360), alt deg in [0,360)). Apply ``signed``."""
    if not reply.endswith(b"#"):
        raise ProtocolError(f"position reply not terminated: {reply!r}")
    try:
        first, second = reply[:-1].decode("ascii").split(",")
    except (UnicodeDecodeError, ValueError):
        raise ProtocolError(f"malformed position reply {reply!r}") from None
    return decode_angle(first), decode_angle(second)


# -- version -----------------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor:02d}.{self.patch:02d}"


def parse_version(reply: bytes) -> Version:
    """Six ASCII hex digits then ``#``: ``042507#`` is 4.37.07."""
    if len(reply) != 7 or not reply.endswith(b"#"):
        raise ProtocolError(f"bad version reply {reply!r}")
    try:
        digits = reply[:6].decode("ascii")
        return Version(int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    except ValueError:
        raise ProtocolError(f"bad version reply {reply!r}") from None


# -- location and time ---------------------------------------------------------------------------
def _dms(value: float) -> tuple[int, int, int]:
    d = int(value)
    rem = (value - d) * 60.0
    m = int(rem)
    s = round((rem - m) * 60.0)
    if s == 60:
        s, m = 0, m + 1
    if m == 60:
        m, d = 0, d + 1
    return d, m, s


def encode_location(lat_deg: float, lon_deg: float) -> bytes:
    """``W`` + lat d,m,s, N=0/S=1, lon d,m,s, E=0/W=1 (whole arcseconds)."""
    if not -90.0 <= lat_deg <= 90.0:
        raise ValueError(f"latitude {lat_deg} out of range")
    lon = ((lon_deg + 180.0) % 360.0) - 180.0
    lat_d, lat_m, lat_s = _dms(abs(lat_deg))
    lon_d, lon_m, lon_s = _dms(abs(lon))
    return b"W" + bytes([lat_d, lat_m, lat_s, int(lat_deg < 0), lon_d, lon_m, lon_s, int(lon < 0)])


def decode_location(reply: bytes) -> tuple[float, float]:
    """(lat, lon) in degrees from the 8 bytes + ``#`` that ``w`` returns."""
    if len(reply) != 9 or not reply.endswith(b"#"):
        raise ProtocolError(f"bad location reply {reply!r}")
    ld, lm, ls, ls_sign, od, om, os_, os_sign = reply[:8]
    lat = ld + lm / 60 + ls / 3600
    lon = od + om / 60 + os_ / 3600
    return (-lat if ls_sign else lat), (-lon if os_sign else lon)


def encode_time_fields(
    hour: int, minute: int, second: int, month: int, day: int, year: int, offset_h: int, dst: int
) -> bytes:
    """``H`` + fields as the document gives them; negative zone offsets are sent as 256 - n."""
    return b"H" + bytes(
        [hour, minute, second, month, day, year % 100, offset_h % 256, 1 if dst else 0]
    )


def encode_time(t: Time) -> bytes:
    """Send UTC as the handset's 'local' time with a zero offset and no daylight saving."""
    rounded = t.utc + TimeDelta(0.5 * u.s)  # to the nearest second
    dt = rounded.datetime
    return encode_time_fields(dt.hour, dt.minute, dt.second, dt.month, dt.day, dt.year, 0, 0)


@dataclass(frozen=True)
class HandsetTime:
    hour: int
    minute: int
    second: int
    month: int
    day: int
    year: int  # two digits, 20xx assumed
    offset_h: int
    dst: bool

    def to_utc(self) -> Time:
        """UTC, taking the fields as local clock time. ``ProtocolError`` if they are not a valid
        date (a handset whose clock was never set sends zeros)."""
        try:
            local = Time(
                f"{2000 + self.year:04d}-{self.month:02d}-{self.day:02d}T"
                f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}",
                scale="utc",
            )
        except ValueError:
            raise ProtocolError(
                f"handset clock holds an invalid date/time "
                f"({self.year:02d}-{self.month:02d}-{self.day:02d} "
                f"{self.hour:02d}:{self.minute:02d}:{self.second:02d})"
            ) from None
        return local - TimeDelta((self.offset_h + int(self.dst)) * u.hour)


def decode_time(reply: bytes) -> HandsetTime:
    if len(reply) != 9 or not reply.endswith(b"#"):
        raise ProtocolError(f"bad time reply {reply!r}")
    h, mi, s, mo, d, y, off, dst = reply[:8]
    return HandsetTime(h, mi, s, mo, d, y, off - 256 if off > 127 else off, bool(dst))


# -- rates -------------------------------------------------------------------------------------
MAX_RATE_ARCSEC_S = 65535 / 4


def encode_rate_slew(axis: int, rate_arcsec_s: float) -> bytes:
    """Variable-rate slew on one axis; the sign selects the direction. Resolution 0.25 arcsec/s."""
    if axis not in (AZ_AXIS, ALT_AXIS):
        raise ValueError(f"unknown axis {axis}")
    n = round(abs(rate_arcsec_s) * 4)
    if n > 65535:
        raise ProtocolError(f"rate {rate_arcsec_s} arcsec/s exceeds {MAX_RATE_ARCSEC_S:.0f}")
    direction = 6 if rate_arcsec_s >= 0 else 7
    return b"P" + bytes([3, axis, direction, n // 256, n % 256, 0, 0])


def decode_rate_slew(command: bytes) -> tuple[int, float]:
    """(axis, signed arcsec/s) from a variable-rate slew command, or ProtocolError."""
    if len(command) != 8 or command[0:2] != b"P\x03" or command[3] not in (6, 7):
        raise ProtocolError(f"not a variable-rate slew: {command!r}")
    axis, direction, hi, lo = command[2], command[3], command[4], command[5]
    rate = (hi * 256 + lo) / 4.0
    return axis, (rate if direction == 6 else -rate)


# -- command framing (used by the simulated handset) ------------------------------------------
_FIXED_LENGTHS = {
    **dict.fromkeys("VmJzZeELMtwhp", 1),
    **dict.fromkeys("bres", 18),
    **dict.fromkeys("BRS", 10),
    "K": 2,
    "T": 2,
    "W": 9,
    "H": 9,
    "P": 8,
}


def command_length(first_byte: int) -> int:
    """Total bytes of the command that starts with ``first_byte`` (1 for unknown letters)."""
    return _FIXED_LENGTHS.get(chr(first_byte), 1)
