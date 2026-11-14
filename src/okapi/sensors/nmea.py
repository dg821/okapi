"""NMEA 0183 parsing and a GPS built on any source of NMEA lines (serial port, gpsd, file).

Only GGA (position, altitude, satellites) and RMC (date, time, validity) are needed. A fix
requires a checksum-valid RMC with status A and a GGA with quality > 0 for the same second.
"""

from collections.abc import Callable
from dataclasses import dataclass

import astropy.units as u
from astropy.time import Time

from okapi.sensors.base import GpsError, GpsFix


def checksum_ok(sentence: str) -> bool:
    """True if the sentence has no checksum or carries a correct one."""
    s = sentence.strip()
    if not s.startswith("$"):
        return False
    if "*" not in s:
        return True
    body, _, given = s[1:].partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(given[:2], 16)
    except ValueError:
        return False


def _latlon(value: str, hemi: str, deg_digits: int) -> float:
    if not value:
        raise ValueError("empty coordinate")
    deg = float(value[:deg_digits])
    minutes = float(value[deg_digits:])
    out = deg + minutes / 60.0
    return -out if hemi in ("S", "W") else out


@dataclass(frozen=True)
class _Gga:
    time: str
    lat: float
    lon: float
    quality: int
    satellites: int
    height_m: float  # above the ellipsoid (MSL altitude + geoid separation)


@dataclass(frozen=True)
class _Rmc:
    time: str
    valid: bool
    date: str


def _parse_gga(f: list[str]) -> _Gga | None:
    try:
        quality = int(f[6] or 0)
        if quality == 0:
            return None
        sep = float(f[11]) if len(f) > 11 and f[11] else 0.0
        return _Gga(
            time=f[1].split(".")[0],
            lat=_latlon(f[2], f[3], 2),
            lon=_latlon(f[4], f[5], 3),
            quality=quality,
            satellites=int(f[7] or 0),
            height_m=float(f[9]) + sep,
        )
    except (ValueError, IndexError):
        return None


def _parse_rmc(f: list[str]) -> _Rmc | None:
    try:
        return _Rmc(time=f[1].split(".")[0], valid=f[2] == "A", date=f[9])
    except IndexError:
        return None


def _utc(date: str, hhmmss: str) -> Time:
    dd, mm, yy = int(date[0:2]), int(date[2:4]), int(date[4:6])
    year = 1900 + yy if yy >= 80 else 2000 + yy  # NMEA has 2-digit years; GPS epoch is 1980
    return Time(
        f"{year:04d}-{mm:02d}-{dd:02d}T{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}", scale="utc"
    )


class NmeaGps:
    """``next_line`` returns the next NMEA line, or raises ``EOFError``/returns '' when done."""

    def __init__(self, next_line: Callable[[], str], *, max_lines: int = 400) -> None:
        self._next_line = next_line
        self._max_lines = max_lines

    def read_fix(self) -> GpsFix:
        gga: _Gga | None = None
        rmc: _Rmc | None = None
        for _ in range(self._max_lines):
            try:
                line = self._next_line()
            except EOFError:
                break
            if not line:
                break
            line = line.strip()
            if not checksum_ok(line):
                continue
            fields = line.split("*")[0].split(",")
            kind = fields[0][3:]
            if kind == "GGA":
                gga = _parse_gga(fields) or gga
            elif kind == "RMC":
                parsed = _parse_rmc(fields)
                rmc = parsed if parsed and parsed.valid else rmc
            if gga and rmc and gga.time == rmc.time:
                return GpsFix(
                    time=_utc(rmc.date, rmc.time),
                    lat=gga.lat * u.deg,
                    lon=gga.lon * u.deg,
                    height=gga.height_m * u.m,
                    satellites=gga.satellites,
                )
        raise GpsError("no valid GGA+RMC fix in the NMEA stream")
