import astropy.units as u
import pytest
from astropy.time import Time

from okapi.clock import SimClock
from okapi.sensors.base import GpsError
from okapi.sensors.fake import FakeGps
from okapi.sensors.nmea import NmeaGps, checksum_ok


def with_checksum(body: str) -> str:
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    return f"${body}*{calc:02X}"


# The canonical examples from the NMEA 0183 documentation.
GGA = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
RMC = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"


def source(*lines: str):
    it = iter(lines)

    def nxt() -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return nxt


def test_documented_examples_have_valid_checksums():
    assert checksum_ok(GGA)
    assert checksum_ok(RMC)
    assert not checksum_ok(GGA.replace("545.4", "545.5"))
    assert not checksum_ok("GPGGA,no,dollar")
    assert checksum_ok("$GPGGA,no,checksum")  # checksum is optional


def test_parses_position_height_time_and_satellites():
    fix = NmeaGps(source(GGA, RMC)).read_fix()
    assert fix.lat.to_value(u.deg) == pytest.approx(48 + 7.038 / 60)
    assert fix.lon.to_value(u.deg) == pytest.approx(11 + 31.0 / 60)
    assert fix.height.to_value(u.m) == pytest.approx(545.4 + 46.9)  # MSL + geoid separation
    assert fix.satellites == 8
    assert fix.time.isot.startswith("1994-03-23T12:35:19")


def test_order_does_not_matter_and_noise_is_skipped():
    fix = NmeaGps(source("$GPGSV,junk*00", RMC, "garbage", GGA)).read_fix()
    assert fix.satellites == 8


def test_southern_and_western_hemispheres_are_negative():
    gga = with_checksum("GPGGA,040000,3351.000,S,07009.000,W,1,07,1.0,100.0,M,30.0,M,,")
    rmc = with_checksum("GPRMC,040000,A,3351.000,S,07009.000,W,0.0,0.0,210626,,")
    fix = NmeaGps(source(gga, rmc)).read_fix()
    assert fix.lat.to_value(u.deg) == pytest.approx(-(33 + 51 / 60))
    assert fix.lon.to_value(u.deg) == pytest.approx(-(70 + 9 / 60))
    assert fix.time.isot.startswith("2026-06-21T04:00:00")  # 2-digit year 26 -> 2026


def test_void_status_is_not_a_fix():
    void = with_checksum("GPRMC,123519,V,,,,,,,230394,,")
    with pytest.raises(GpsError):
        NmeaGps(source(GGA, void)).read_fix()


def test_zero_quality_is_not_a_fix():
    nofix = with_checksum("GPGGA,123519,,,,,0,00,,,M,,M,,")
    with pytest.raises(GpsError):
        NmeaGps(source(nofix, RMC)).read_fix()


def test_corrupt_sentences_are_ignored():
    corrupt = GGA.replace("4807.038", "4807.039")
    with pytest.raises(GpsError):
        NmeaGps(source(corrupt, RMC)).read_fix()


def test_gga_and_rmc_from_different_seconds_do_not_pair():
    later = with_checksum("GPRMC,123520,A,4807.038,N,01131.000,E,0,0,230394,,")
    with pytest.raises(GpsError):
        NmeaGps(source(GGA, later)).read_fix()


def test_gives_up_after_max_lines():
    junk = with_checksum("GPGSV,1,1,00")
    with pytest.raises(GpsError):
        NmeaGps(lambda: junk, max_lines=10).read_fix()


def test_fake_gps_reports_site_and_clock_error(site):
    clock = SimClock(Time("2026-06-21T04:00:00", scale="utc"))
    fix = FakeGps(clock, site, clock_error=10 * u.s).read_fix()
    assert fix.lat.to_value(u.deg) == pytest.approx(40.0)
    assert (clock.now() - fix.time).to_value(u.s) == pytest.approx(10.0)
    assert fix.site.lon.to_value(u.deg) == pytest.approx(-105.0)


def test_fake_gps_failures(site, clock):
    gps = FakeGps(clock, site, fail_first=2)
    for _ in range(2):
        with pytest.raises(GpsError):
            gps.read_fix()
    assert gps.read_fix().satellites == 9
    with pytest.raises(GpsError):
        FakeGps(clock, site, always_fail=True).read_fix()
