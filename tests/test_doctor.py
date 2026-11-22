import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from typer.testing import CliRunner

from okapi import doctor as d
from okapi.camera.base import CameraError
from okapi.cli import app
from okapi.clock import SimClock
from okapi.sensors.fake import FakeGps
from okapi.sim import make_synscan_rig

SITE = EarthLocation.from_geodetic(-105 * u.deg, 40 * u.deg, 1600 * u.m)
LISTING = "Model  Port\n" + "-" * 30 + "\nCanon EOS 600D  usb:001,004\n"
SHUTTER = "Label: Shutter Speed\nChoice: 0 30\nChoice: 1 25\nChoice: 2 bulb\n"


def runner(listing=LISTING, shutter=SHUTTER):
    def run(args):
        return listing if args == ["--auto-detect"] else shutter

    return run


def test_camera_passes_when_found_in_manual_mode():
    c = d.check_camera(runner())
    assert c.status == d.PASS and "EOS 600D" in c.detail


def test_camera_missing_fails_with_a_fix():
    c = d.check_camera(runner(listing="Model  Port\n------\n"))
    assert c.status == d.FAIL and "USB" in c.fix


def test_camera_not_in_manual_mode_fails():
    c = d.check_camera(runner(shutter="Label: Shutter Speed\nChoice: 0 Auto\n"))
    assert c.status == d.FAIL and "M" in c.fix


def test_camera_without_gphoto2_installed():
    def run(args):
        raise CameraError("gphoto2 is not installed")

    assert d.check_camera(run).status == d.FAIL


def test_gps_ok_and_clock_compared_to_gps_time():
    clock = SimClock(Time("2026-11-21T04:00:00"))
    ok = d.check_gps(FakeGps(clock, SITE), clock)
    assert [c.status for c in ok] == [d.PASS, d.PASS]
    loose = d.check_gps(FakeGps(clock, SITE, clock_error=1 * u.s), clock)
    assert loose[1].status == d.WARN
    bad = d.check_gps(FakeGps(clock, SITE, clock_error=5 * u.s), clock)
    assert bad[1].status == d.FAIL and "chrony" in bad[1].fix


def test_gps_without_fix_fails_and_skips_the_clock_check():
    clock = SimClock(Time("2026-11-21T04:00:00"))
    out = d.check_gps(FakeGps(clock, SITE, always_fail=True), clock)
    assert len(out) == 1 and out[0].status == d.FAIL


def test_gps_with_few_satellites_warns():
    clock = SimClock(Time("2026-11-21T04:00:00"))
    assert d.check_gps(FakeGps(clock, SITE, satellites=2), clock)[0].status == d.WARN


def test_mount_check_reports_firmware_and_failures():
    rig = make_synscan_rig()
    c = d.check_mount(rig.mount.read_report)
    assert c.status == d.PASS and "3.38" in c.detail

    def boom():
        raise OSError("timeout")

    assert d.check_mount(boom).status == d.FAIL


def test_architecture_32bit_fails():
    assert d.check_architecture("armv7l").status == d.FAIL
    assert d.check_architecture("aarch64").status == d.PASS


def test_astap_missing_database_and_present(tmp_path):
    exe = tmp_path / "astap"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert d.check_astap("/no/such/astap").status == d.FAIL
    assert d.check_astap(str(exe), db_dirs=[tmp_path]).status == d.WARN
    (tmp_path / "d50_star_database.290").write_text("x")
    assert d.check_astap(str(exe), db_dirs=[tmp_path]).status == d.PASS


def test_disk_threshold(tmp_path):
    assert d.check_disk(tmp_path, free_gb=2).status == d.FAIL
    assert d.check_disk(tmp_path, free_gb=100).status == d.PASS


def test_format_shows_fixes_only_for_problems():
    text = d.format_checks(
        [d.Check("a", d.PASS, "fine", "ignored"), d.Check("b", d.FAIL, "broken", "do this")]
    )
    assert "-> do this" in text and "ignored" not in text and "1 failed" in text


def test_doctor_command_fails_when_something_is_wrong(tmp_path):
    r = CliRunner().invoke(
        app,
        ["doctor", "--skip-camera", "--astap-path", "/no/astap", "--data-dir", str(tmp_path)],
    )
    assert r.exit_code == 1 and "[FAIL] astap" in r.stdout
