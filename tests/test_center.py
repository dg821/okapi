import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from hypothesis import given, settings
from hypothesis import strategies as st

from okapi.align.center import CenteringError, align_on_star, center_on
from okapi.camera.base import ExposureSettings
from okapi.camera.fake import FakeCamera
from okapi.clock import SimClock
from okapi.coords import AltAzPosition, altaz_to_radec, site_from_geodetic
from okapi.mount.fake import FakeMount
from okapi.solve.fake import FakeSolver

SETTINGS = ExposureSettings(2 * u.s, 1600)


def sky_target(site, clock, az=150.0, alt=55.0) -> SkyCoord:
    return altaz_to_radec(AltAzPosition(az * u.deg, alt * u.deg), site, clock.now())


def rig(clock, site, *, error=(0.7 * u.deg, -0.4 * u.deg), **solver_kw):
    mount = FakeMount(clock, alignment_error=error)
    mount.set_site(site)
    camera = FakeCamera(clock)
    solver = FakeSolver(mount.true_radec, **solver_kw)
    return dict(mount=mount, camera=camera, solver=solver, site=site, settings=SETTINGS)


def true_error_arcsec(mount, target) -> float:
    return mount.true_radec().separation(target).to_value(u.arcsec)


def test_centering_fixes_a_large_alignment_error(clock, site):
    target = sky_target(site, clock)
    kw = rig(clock, site)
    result = center_on(target, **kw)
    assert result.error <= 15 * u.arcsec
    assert result.iterations <= 3
    assert true_error_arcsec(kw["mount"], target) < 15.0
    # The handset's own error is still there: we moved the true pointing, not the report.
    # An az error of 0.7 deg shrinks by cos(55 deg) on the sky: sqrt((0.7*cos55)^2 + 0.4^2) = 0.57.
    m = kw["mount"]
    assert 0.5 < m.get_altaz().separation(m.true_altaz()).to_value(u.deg) < 0.65


def test_already_centered_takes_one_iteration(clock, site):
    kw = rig(clock, site, error=(0 * u.deg, 0 * u.deg))
    result = center_on(sky_target(site, clock), **kw)
    assert result.iterations == 1
    assert len(kw["camera"].captures) == 1


def test_noisy_solver_still_converges_within_tolerance_plus_noise(clock, site):
    target = sky_target(site, clock)
    kw = rig(clock, site, noise=3 * u.arcsec, seed=5)
    center_on(target, **kw)
    assert true_error_arcsec(kw["mount"], target) < 15.0 + 3 * 4


def test_transient_solve_failures_are_retried_without_costing_iterations(clock, site):
    kw = rig(clock, site, fail_first=2)
    result = center_on(sky_target(site, clock), **kw)
    assert kw["solver"].calls >= 3
    assert result.iterations <= 3


def test_persistent_solve_failure_raises(clock, site):
    kw = rig(clock, site, always_fail=True)
    with pytest.raises(CenteringError, match="solve failures"):
        center_on(sky_target(site, clock), **kw)
    assert kw["solver"].calls == 3


def test_gives_up_when_the_mount_does_not_move(clock, site):
    class StuckMount(FakeMount):
        def nudge(self, d_az, d_alt):  # ignores the correction
            pass

    mount = StuckMount(clock, alignment_error=(0.7 * u.deg, 0 * u.deg))
    mount.set_site(site)
    with pytest.raises(CenteringError, match="after 4 iterations"):
        center_on(
            sky_target(site, clock),
            mount=mount,
            camera=FakeCamera(clock),
            solver=FakeSolver(mount.true_radec),
            site=site,
            settings=SETTINGS,
            max_iterations=4,
        )


def test_rejects_zero_iteration_budget(clock, site):
    with pytest.raises(ValueError):
        center_on(sky_target(site, clock), max_iterations=0, **rig(clock, site))


def test_align_on_star_syncs_so_the_next_slew_is_accurate(clock, site):
    target = sky_target(site, clock)
    kw = rig(clock, site)
    align_on_star(target, **kw)
    other = sky_target(site, clock, az=230.0, alt=40.0)
    kw["mount"].slew_radec(other)
    kw["mount"].wait_until_settled(120 * u.s)
    assert true_error_arcsec(kw["mount"], other) < 30.0  # without the sync this would be ~0.8 deg


def test_without_sync_a_second_slew_is_still_off(clock, site):
    kw = rig(clock, site)
    center_on(sky_target(site, clock), **kw)  # centered, but handset never told
    other = sky_target(site, clock, az=230.0, alt=40.0)
    kw["mount"].slew_radec(other)
    kw["mount"].wait_until_settled(120 * u.s)
    assert true_error_arcsec(kw["mount"], other) > 1000.0


@given(
    err_az=st.floats(-2, 2),
    err_alt=st.floats(-2, 2),
    az=st.floats(0, 359),
    alt=st.floats(25, 75),
)
@settings(max_examples=15, deadline=None)
def test_converges_for_any_alignment_error(err_az, err_alt, az, alt):
    from astropy.time import Time

    clock = SimClock(Time("2026-06-21T04:00:00", scale="utc"))
    site = site_from_geodetic(40.0, -105.0, 1600.0)
    target = sky_target(site, clock, az=az, alt=alt)
    kw = rig(clock, site, error=(err_az * u.deg, err_alt * u.deg))
    center_on(target, **kw)
    assert true_error_arcsec(kw["mount"], target) < 15.0
