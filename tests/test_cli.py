from typer.testing import CliRunner

from okapi import __version__
from okapi.cli import app


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_calibrate_dry_run_reports_stars_and_accuracy():
    result = CliRunner().invoke(app, ["calibrate", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "aligned on:" in result.stdout
    assert "pointing model rms" in result.stdout
    assert "verification on" in result.stdout


def test_calibrate_without_dry_run_refuses_because_no_hardware_driver_exists():
    result = CliRunner().invoke(app, ["calibrate"])
    assert result.exit_code == 2
    assert "not implemented" in result.output


def test_mount_info_dry_run_prints_the_handset_report():
    result = CliRunner().invoke(app, ["mount-info", "--dry-run"])
    assert result.exit_code == 0, result.output
    for expected in ("firmware: 3.38.00", "DOB GOTO series", "aligned: no", "az/alt:", "UTC"):
        assert expected in result.stdout
    # a fresh simulated handset has never had its clock set: reported, not a crash
    assert "not set" in result.stdout


def test_mount_info_needs_a_port_or_dry_run():
    result = CliRunner().invoke(app, ["mount-info"])
    assert result.exit_code == 2
    assert "serial port" in result.output


def test_mount_info_reports_an_unopenable_port_cleanly():
    result = CliRunner().invoke(app, ["mount-info", "/dev/does-not-exist"])
    assert result.exit_code == 1
    assert "could not talk to the hand control" in result.output


def test_mount_info_sends_only_queries():
    from okapi.mount.synscan_sim import FakeHandset  # noqa: F401  (documents what is simulated)
    from okapi.sim import make_synscan_rig

    rig = make_synscan_rig()
    rig.mount.read_report()
    allowed = {b"K", b"V", b"m", b"J", b"z", b"t", b"w", b"h"}
    assert {c[:1] for c in rig.handset.commands} <= allowed
