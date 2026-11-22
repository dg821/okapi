import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from okapi.astrometry.pipeline import process_frame
from okapi.camera.base import CameraError, ExposureSettings
from okapi.orbits.simulate import geo_state, overhead_leo_state
from okapi.orbits.twobody import propagate
from okapi.sim_night import _clip_segment, decode_npy
from okapi.sim_night import make_night_rig as _make_night_rig
from okapi.solve.base import SolveError
from okapi.targets import StateTarget


def test_clip_segment_inside_crossing_and_outside():
    assert _clip_segment((10, 10), (90, 50), 100, 100) == ((10, 10), (90, 50))
    crossing = _clip_segment((-50, 50), (150, 50), 100, 100)
    assert crossing is not None
    a, b = crossing
    assert a == pytest.approx((0, 50)) and b == pytest.approx((99, 50))
    assert _clip_segment((-50, -50), (-10, -10), 100, 100) is None
    assert _clip_segment((200, 0), (300, 90), 100, 100) is None
    assert _clip_segment((50, 50), (50, 50), 100, 100) is None  # a point


ZERO = (0 * u.deg,) * 3


def make_night_rig(*args, **kw):
    """These tests drive the physical mount directly, so give it no tilt."""
    kw.setdefault("frame_rotation", ZERO)
    return _make_night_rig(*args, **kw)


def geo_truth(rig, lon):
    r, v = geo_state(lon, rig.clock.now())
    return StateTarget("truth", r, v, rig.clock.now())


def point_at_satellite(rig, sat, exposure=20.0):
    """Cheat by driving the physical mount (not the handset): put the frame centre on where the
    satellite will be mid-exposure, then track. Repeats because a slew takes real time."""
    for _ in range(3):
        rig.physical.slew_radec(sat.predict(rig.site, rig.clock.now() + (3 + exposure / 2) * u.s))
        rig.physical.wait_until_settled(300 * u.s)
    rig.physical.start_tracking()


def test_the_camera_renders_a_satellite_streak_where_the_satellite_is():
    rig0 = make_night_rig()
    sat = geo_truth(rig0, -100.0)
    rig = make_night_rig([sat])
    point_at_satellite(rig, sat)
    rig.physical.start_tracking()
    image = rig.camera.capture(ExposureSettings(20 * u.s, 1600))
    res = process_frame(image, solver=rig.solver, decode=decode_npy)
    assert res.n_streaks == 1
    obs = res.observations[0]
    truth = sat.predict(rig.site, image.start_time + 10 * u.s)
    assert obs.coord.separation(truth).to_value(u.arcsec) < 3.0


def test_no_streak_when_the_satellite_is_outside_the_field():
    rig0 = make_night_rig()
    sat = geo_truth(rig0, -100.0)
    far = geo_truth(rig0, -90.0)
    rig = make_night_rig([far])
    point_at_satellite(rig, sat)
    rig.physical.start_tracking()
    image = rig.camera.capture(ExposureSettings(20 * u.s, 1600))
    assert process_frame(image, solver=rig.solver, decode=decode_npy).n_streaks == 0


def test_a_leo_pass_through_a_long_exposure_comes_out_as_a_line():
    rig0 = make_night_rig()
    t0 = rig0.clock.now() + 270 * u.s
    r, v = overhead_leo_state(rig0.site, t0, 600.0, 90.0)
    r30, v30 = propagate(r, v, 30.0)  # 30 s past the zenith: high, but under the 89 deg limit
    t_cross = t0 + 30 * u.s
    sat = StateTarget("leo", r30, v30, t_cross)
    rig = make_night_rig([sat])
    rig.physical.slew_radec(sat.predict(rig.site, t_cross))
    rig.physical.wait_until_settled(300 * u.s)
    rig.physical.start_tracking()
    wait = (t_cross - 5 * u.s - rig.clock.now()).to_value(u.s)
    assert wait > 0
    rig.clock.sleep(wait)
    image = rig.camera.capture(ExposureSettings(10 * u.s, 1600))
    res = process_frame(image, solver=rig.solver, decode=decode_npy)
    assert res.n_streaks == 1 and res.observations[0].kind == "line"


def test_camera_failure_injection_and_clock_advance():
    rig = make_night_rig(camera_fail_on=lambda n: n == 2)
    t0 = rig.clock.now()
    rig.camera.capture(ExposureSettings(5 * u.s, 1600))
    assert (rig.clock.now() - t0).to_value(u.s) == pytest.approx(5.0)
    with pytest.raises(CameraError):
        rig.camera.capture(ExposureSettings(5 * u.s, 1600))
    rig.camera.capture(ExposureSettings(5 * u.s, 1600))  # recovers


def test_solver_refuses_a_frame_the_camera_did_not_just_take():
    rig = make_night_rig()
    a = rig.camera.capture(ExposureSettings(2 * u.s, 1600))
    rig.camera.capture(ExposureSettings(2 * u.s, 1600))
    with pytest.raises(SolveError, match="did not just take"):
        rig.solver.solve(a)


def test_solver_failures_are_injectable_and_noise_moves_the_centre():
    rig = make_night_rig(solver_fail_on=lambda n: n == 1)
    img = rig.camera.capture(ExposureSettings(2 * u.s, 1600))
    with pytest.raises(SolveError, match="cloud"):
        rig.solver.solve(img)
    noisy = make_night_rig(solver_noise=20 * u.arcsec)
    img = noisy.camera.capture(ExposureSettings(2 * u.s, 1600))
    c = noisy.solver.solve(img).center
    true = noisy.physical.true_radec()
    assert 1.0 < c.separation(true).to_value(u.arcsec) < 120.0


def test_the_solved_wcs_maps_pixels_to_the_sky_the_frame_was_rendered_from():
    rig = make_night_rig()
    rig.physical.slew_radec(SkyCoord(200 * u.deg, 30 * u.deg))
    rig.physical.wait_until_settled(180 * u.s)
    rig.physical.start_tracking()  # a tracked stare holds the centre fixed on the sky
    img = rig.camera.capture(ExposureSettings(2 * u.s, 1600))
    sol = rig.solver.solve(img)
    h, w = rig.camera.shape
    centre = sol.wcs.pixel_to_world((w - 1) / 2, (h - 1) / 2)
    assert centre.separation(rig.physical.true_radec()).to_value(u.arcsec) < 5
    scale = sol.wcs.proj_plane_pixel_scales()[0].to_value(u.arcsec)
    assert scale == pytest.approx(1.48, abs=0.01)
    assert decode_npy(img).shape == rig.camera.shape
