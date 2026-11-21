"""Bring-up checks against the real hand control. Skipped unless OKAPI_TEST_PORT is set.

    OKAPI_TEST_PORT=/dev/ttyUSB0 uv run pytest -m hardware tests/test_synscan_hardware.py -s

Stage 1 (always runs when the port is set) only reads, plus writes site and time to the handset,
which is harmless. Stage 2 MOVES THE TELESCOPE and runs only if you also set OKAPI_TEST_MOTION=1.
Before that: clear the area, make sure the tube is balanced and cables are free, and have a hand
on the power switch. Stage 2 answers PROJECT_PLAN assumptions A6 (does Pi-driven tracking work)
and finds the sign of each axis.
"""

import os
import time

import astropy.units as u
import pytest
from astropy.time import Time

from okapi.clock import SystemClock
from okapi.coords import AltAzPosition, site_from_geodetic
from okapi.mount.synscan import SynScanMount
from okapi.mount.synscan_protocol import ALT_AXIS, AZ_AXIS, encode_rate_slew
from okapi.mount.transport import SerialTransport

PORT = os.environ.get("OKAPI_TEST_PORT")
MOTION = os.environ.get("OKAPI_TEST_MOTION") == "1"

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(not PORT, reason="set OKAPI_TEST_PORT to the hand control's serial port"),
]


@pytest.fixture
def mount():
    m = SynScanMount(SerialTransport(PORT or ""), SystemClock())
    yield m
    m.close()


def test_stage1_handshake_and_read_only_report(mount):
    report = mount.read_report()
    print(
        f"\nfirmware {report.info.version}, model {report.info.model_name}, "
        f"aligned={report.info.aligned}, az/alt={report.altaz}, "
        f"tracking mode {report.tracking_mode}, handset time {report.time_utc.isot}"
    )
    assert report.info.version.major in (3, 4)
    assert 128 <= report.info.model <= 159, "not an alt-az handset series"
    for w in mount.warnings:
        print("warning:", w)


def test_stage1_site_and_time_round_trip(mount):
    mount.set_site(site_from_geodetic(40.0123, -105.2745, 1600.0))
    mount.set_time(Time.now())
    report = mount.read_report()
    assert abs(report.location[0] - 40.0123) < 2 / 3600
    assert abs(report.location[1] + 105.2745) < 2 / 3600
    assert abs((report.time_utc - Time.now()).to_value(u.s)) < 5


@pytest.mark.skipif(not MOTION, reason="set OKAPI_TEST_MOTION=1 to allow the telescope to move")
def test_stage2_axis_signs(mount):
    """A positive rate must raise azimuth / altitude. If not, use az_rate_sign=-1 or
    alt_rate_sign=-1 when constructing SynScanMount."""
    for axis, name in ((AZ_AXIS, "az"), (ALT_AXIS, "alt")):
        before = mount.get_altaz()
        mount._ack(encode_rate_slew(axis, 60.0))
        time.sleep(3)
        mount._ack(encode_rate_slew(axis, 0.0))
        time.sleep(0.5)
        after = mount.get_altaz()
        delta = (getattr(after, name) - getattr(before, name)).to_value(u.deg)
        delta = ((delta + 180) % 360) - 180
        print(f"\n{name}: moved {delta * 3600:+.0f} arcsec (expected about +180)")
        assert delta > 0.02, f"{name} did not move up for a positive rate: set {name}_rate_sign=-1"
        assert delta < 0.09


@pytest.mark.skipif(not MOTION, reason="set OKAPI_TEST_MOTION=1 to allow the telescope to move")
def test_stage2_small_goto_and_back(mount):
    start = mount.get_altaz()
    dest = AltAzPosition(start.az + 2 * u.deg, start.alt + 2 * u.deg)
    mount.slew_altaz(dest)
    mount.wait_until_settled(60 * u.s)
    assert mount.get_altaz().separation(dest).to_value(u.arcsec) < 60
    mount.slew_altaz(start)
    mount.wait_until_settled(60 * u.s)
    assert mount.get_altaz().separation(start).to_value(u.arcsec) < 60
