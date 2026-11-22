import astropy.units as u
from astropy.coordinates import SkyCoord
from typer.testing import CliRunner

from okapi.cli import app
from okapi.hardware import TargetRecorder


class _Inner:
    def __init__(self):
        self.sent = []

    def slew_radec(self, target):
        self.sent.append(target)

    def get_altaz(self):
        return "altaz"


def test_recorder_remembers_the_last_radec_target_and_forwards_it():
    inner = _Inner()
    rec = TargetRecorder(inner)
    assert rec.last_target is None
    t = SkyCoord(10 * u.deg, 20 * u.deg)
    rec.slew_radec(t)
    assert rec.last_target is t and inner.sent == [t]


def test_recorder_passes_other_calls_through():
    assert TargetRecorder(_Inner()).get_altaz() == "altaz"


def test_calibrate_needs_both_ports_or_dry_run():
    r = CliRunner().invoke(app, ["calibrate", "--mount-port", "/dev/x"])
    assert r.exit_code == 2 and "--gps-port" in r.output


def test_calibrate_fallback_site_needs_both_coordinates():
    r = CliRunner().invoke(app, ["calibrate", "--dry-run", "--fallback-lat", "40"])
    assert r.exit_code == 2 and "fallback-lon" in r.output


def test_calibrate_reports_an_unopenable_serial_port_cleanly():
    r = CliRunner().invoke(
        app, ["calibrate", "--mount-port", "/dev/nope1", "--gps-port", "/dev/nope2"]
    )
    assert r.exit_code == 1 and "could not open the hardware" in r.output


def test_dry_run_still_works():
    r = CliRunner().invoke(app, ["calibrate", "--dry-run"])
    assert r.exit_code == 0 and "aligned on:" in r.stdout
