"""Whole chain on synthetic frames: render a satellite into star fields, run the frames through
decode + solve + streak detection, then fit an orbit to the observations that come out."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from okapi.astrometry.observation import read_observations, write_observations
from okapi.astrometry.pipeline import observations_from_array, process_frame
from okapi.astrometry.simulate import SimStar, SimStreak, render_frame, tangent_wcs
from okapi.camera.base import CapturedImage, ExposureSettings
from okapi.coords import site_from_geodetic
from okapi.orbits.fit import determine_orbit
from okapi.orbits.geometry import radec_of
from okapi.orbits.simulate import geo_state, topocentric
from okapi.orbits.twobody import propagate
from okapi.solve.base import SolveError, SolveResult

SITE = site_from_geodetic(40.0, -105.3, 1600.0)
EPOCH = Time("2026-11-21T04:00:00")
SHAPE = (700, 1000)
SCALE = 1.48  # arcsec/px: the T3i with 2x2 binning at 1200 mm
EXPOSURE = 20 * u.s


def sky_of(r0, v0, t):
    ra, dec = radec_of(topocentric(r0, v0, t, SITE, EPOCH))
    return np.degrees(ra), np.degrees(dec)


class StubSolver:
    def __init__(self, wcs):
        self.wcs = wcs

    def solve(self, image):
        centre = self.wcs.pixel_to_world((SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2).icrs
        return SolveResult(center=centre, wcs=self.wcs)


def make_frame(r0, v0, start_s, seed):
    """A frame aimed at the satellite's predicted position, with the satellite trailing across
    it and stars as points (sidereal tracking)."""
    ra_mid, dec_mid = sky_of(r0, v0, start_s + EXPOSURE.to_value(u.s) / 2)
    wcs = tangent_wcs(ra_mid, dec_mid, SHAPE, SCALE)
    ends = []
    for t in (start_s, start_s + EXPOSURE.to_value(u.s)):
        ra, dec = sky_of(r0, v0, t)
        x, y = wcs.world_to_pixel_values(ra, dec)
        ends.append((float(x), float(y)))
    rng = np.random.default_rng(seed)
    stars = [
        SimStar(rng.uniform(5, SHAPE[1] - 5), rng.uniform(5, SHAPE[0] - 5), rng.uniform(150, 3000))
        for _ in range(40)
    ]
    streak = SimStreak(*ends[0], *ends[1], peak=120)
    data = render_frame(SHAPE, stars=stars, streaks=[streak], seed=seed)
    image = CapturedImage(b"", ".cr2", EPOCH + start_s * u.s, ExposureSettings(EXPOSURE, 1600))
    return image, data, wcs


@pytest.fixture(scope="module")
def geo():
    return geo_state(-100.0, EPOCH)


def test_a_frame_yields_one_timed_point_matching_the_true_sky_position(geo):
    r0, v0 = geo
    image, data, wcs = make_frame(r0, v0, 600.0, seed=1)
    res = process_frame(image, solver=StubSolver(wcs), decode=lambda _i: data)
    assert res.n_streaks == 1 and res.warnings == ()
    obs = res.observations[0]
    assert obs.kind == "point"
    assert (obs.time - image.start_time).to_value(u.s) == pytest.approx(10.0)
    truth = SkyCoord(*(a * u.deg for a in sky_of(r0, v0, 610.0)))
    assert obs.coord.separation(truth).to_value(u.arcsec) < 2.0
    # the GEO object drifts against the stars at roughly the sidereal rate
    assert obs.rate.to_value(u.arcsec / u.s) == pytest.approx(15.0, rel=0.05)


def test_a_frame_with_no_satellite_gives_an_empty_result_not_an_error(geo):
    r0, v0 = geo
    _, _, wcs = make_frame(r0, v0, 600.0, seed=1)
    stars_only = render_frame(SHAPE, stars=[SimStar(300, 300, 2000), SimStar(700, 500, 900)])
    res = process_frame(
        CapturedImage(b"", ".cr2", EPOCH, ExposureSettings(EXPOSURE, 1600)),
        solver=StubSolver(wcs),
        decode=lambda _i: stars_only,
    )
    assert res.observations == [] and res.n_streaks == 0


def test_a_solver_without_a_wcs_is_an_error():
    class NoWcs:
        def solve(self, image):
            return SolveResult(center=SkyCoord(0 * u.deg, 0 * u.deg))

    with pytest.raises(SolveError, match="WCS"):
        process_frame(
            CapturedImage(b"", ".cr2", EPOCH, ExposureSettings(EXPOSURE, 1600)),
            solver=NoWcs(),
            decode=lambda _i: np.zeros(SHAPE),
        )


def test_a_failed_solve_propagates():
    class Failing:
        def solve(self, image):
            raise SolveError("cloud")

    with pytest.raises(SolveError, match="cloud"):
        process_frame(
            CapturedImage(b"", ".cr2", EPOCH, ExposureSettings(EXPOSURE, 1600)),
            solver=Failing(),
            decode=lambda _i: np.zeros(SHAPE),
        )


def test_observation_json_round_trips_and_an_empty_frame_is_a_valid_file(geo, tmp_path):
    r0, v0 = geo
    image, data, wcs = make_frame(r0, v0, 0.0, seed=2)
    obs = observations_from_array(data, wcs, start=image.start_time, exposure=EXPOSURE)
    path = tmp_path / "f.json"
    write_observations(path, "20261121T040000_0000", obs)
    back = read_observations(path)
    assert len(back) == 1
    assert back[0].kind == obs[0].kind
    assert back[0].coord.separation(obs[0].coord).to_value(u.arcsec) < 1e-6
    assert (back[0].time - obs[0].time).to_value(u.s) == pytest.approx(0, abs=1e-6)
    assert back[0].rate.to_value(u.arcsec / u.s) == pytest.approx(
        obs[0].rate.to_value(u.arcsec / u.s)
    )
    write_observations(path, "x", [])
    assert read_observations(path) == []
    assert not list(tmp_path.glob("*.tmp"))


def test_orbit_recovered_from_pixels_of_five_synthetic_frames(geo):
    r0, v0 = geo
    observations = []
    for k, start in enumerate([0.0, 1800.0, 3600.0, 5400.0, 7200.0]):
        image, data, wcs = make_frame(r0, v0, start, seed=10 + k)
        res = process_frame(image, solver=StubSolver(wcs), decode=lambda _i, d=data: d)
        assert res.n_streaks == 1, f"frame {k}"
        observations += res.observations
    sol = determine_orbit(observations, SITE)
    rt, vt = propagate(r0, v0, float((sol.epoch - EPOCH).to_value(u.s)))
    # real measurement noise now: pixel centroids, PSF, quantisation
    assert np.linalg.norm(sol.r - rt) < 15.0  # km
    assert np.linalg.norm(sol.v - vt) < 0.005  # km/s
    assert sol.elements.a == pytest.approx(42164, abs=60)
    assert 0.05 < sol.chi2_reduced < 5
