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
