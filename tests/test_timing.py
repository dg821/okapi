import astropy.units as u
import pytest
from astropy.time import Time
from typer.testing import CliRunner

from okapi.astrometry.observation import Observation, write_observations
from okapi.astrometry.timing import TimingError, estimate_time_offset
from okapi.cli import app
from okapi.coords import site_from_geodetic
from okapi.orbits.simulate import overhead_leo_state
from okapi.targets import StateTarget

SITE = site_from_geodetic(40.0, -105.3, 1600.0)
EPOCH = Time("2026-11-21T04:00:00")


def leo():
    r, v = overhead_leo_state(SITE, EPOCH + 60 * u.s)
    return StateTarget("sat", r, v, EPOCH)


def observed(target, recorded, delay_s, noise_arcsec=0.0):
    """The streak midpoint if the shutter really opened ``delay_s`` after the recorded time."""
    coord = target.predict(SITE, recorded + delay_s * u.s)
    if noise_arcsec:
        coord = coord.directional_offset_by(0 * u.deg, noise_arcsec * u.arcsec)
    return Observation(
        "point",
        recorded,
        coord,
        1 * u.arcsec,
        0.1 * u.s,
        1 * u.deg / u.s,
        0 * u.deg,
        1.0,
        1 * u.deg,
    )


@pytest.mark.parametrize("delay", [0.0, 0.35, -0.2, 1.1])
def test_known_delay_is_recovered(delay):
    t = leo()
    obs = [observed(t, EPOCH + s * u.s, delay) for s in (40, 55, 70, 85)]
    res = estimate_time_offset(obs, t, SITE)
    assert res.offset.to_value(u.s) == pytest.approx(delay, abs=0.01) and res.n == 4


def test_scatter_reflects_inconsistent_frames():
    t = leo()
    obs = [observed(t, EPOCH + s * u.s, d) for s, d in ((40, 0.2), (55, 0.4), (70, 0.3))]
    res = estimate_time_offset(obs, t, SITE)
    assert res.offset.to_value(u.s) == pytest.approx(0.3, abs=0.01)
    assert res.scatter.to_value(u.s) == pytest.approx(0.1, abs=0.02)


def test_a_different_satellite_is_rejected_not_averaged():
    t = leo()
    r, v = overhead_leo_state(SITE, EPOCH + 60 * u.s, altitude_km=900.0, heading_deg=20.0)
    other = StateTarget("other", r, v, EPOCH)
    obs = [observed(other, EPOCH + s * u.s, 0.0) for s in (40, 70)]
    with pytest.raises(TimingError):
        estimate_time_offset(obs, t, SITE)


def test_lines_are_ignored():
    t = leo()
    line = Observation(
        "line", EPOCH, t.predict(SITE, EPOCH), 1 * u.arcsec, 10 * u.s, None, 0 * u.deg, 1.0
    )
    with pytest.raises(TimingError):
        estimate_time_offset([line], t, SITE)


def test_cli_reports_the_offset(tmp_path):
    from tests.tle_helpers import make_tle

    t = leo()
    obs = [observed(t, EPOCH + 55 * u.s, 0.4)]
    f = tmp_path / "a.obs.json"
    write_observations(f, "x", obs)
    l1, l2 = make_tle(epoch_year=26, epoch_day=325.2)
    tle = tmp_path / "t.tle"
    tle.write_text(f"SAT\n{l1}\n{l2}\n")
    # the TLE is not the simulated orbit, so no estimate: the CLI says so cleanly
    r = CliRunner().invoke(
        app, ["timing", str(f), "--tle", str(tle), "--lat", "40", "--lon", "-105.3"]
    )
    assert r.exit_code == 1 and "no estimate" in r.output
    r = CliRunner().invoke(
        app, ["timing", str(f), "--tle", str(tle), "--name", "zzz", "--lat", "40", "--lon", "-105"]
    )
    assert r.exit_code == 2
