import astropy.units as u
import pytest

from okapi.coords import site_from_geodetic
from okapi.session.calibrate import CalibrationError, calibrate
from okapi.sim import SimRig, make_sim_rig
from okapi.solve.fake import FakeSolver


def run(rig: SimRig, **kw):
    return calibrate(
        mount=rig.mount,
        camera=rig.camera,
        solver=rig.solver,
        gps=rig.gps,
        clock=rig.clock,
        settings=rig.settings,
        **kw,
    )


def test_happy_path_recovers_a_tilted_mount_and_verifies_on_a_third_star():
    rig = make_sim_rig()
    rep = run(rig)
    assert len(set(rep.stars)) == 2
    assert rep.verification_star not in rep.stars
    assert rep.model_rms.to_value(u.arcsec) < 15
    assert rep.verification_error.to_value(u.arcsec) < 30
    assert rep.warnings == ()
    assert not rep.used_fallback_site
    # The returned mount hides the tilt: slew to something new and land on it.
    from okapi.coords import AltAzPosition, altaz_to_radec

    other = altaz_to_radec(AltAzPosition(250 * u.deg, 45 * u.deg), rig.site, rig.clock.now())
    for _ in range(2):
        rep.mount.slew_radec(other)
        rep.mount.wait_until_settled(180 * u.s)
    assert rig.mount.true_radec().separation(other).to_value(u.arcsec) < 15


def test_site_and_gps_time_are_sent_to_the_mount():
    rig = make_sim_rig()
    rep = run(rig)
    assert rig.mount.mount_time == rep.gps_time
    assert rep.site.lat.to_value(u.deg) == pytest.approx(40.0)


def test_system_clock_disagreement_is_reported_and_gps_time_wins():
    rig = make_sim_rig(gps_clock_error=10 * u.s)
    rep = run(rig)
    assert rep.clock_error.to_value(u.s) == pytest.approx(10.0, abs=0.5)
    assert any("system clock" in w for w in rep.warnings)
    assert (rep.gps_time - rig.mount.mount_time).to_value(u.s) == pytest.approx(0.0)


def test_small_clock_error_is_not_a_warning():
    assert run(make_sim_rig(gps_clock_error=1 * u.s)).warnings == ()


def test_gps_failure_without_fallback_aborts():
    with pytest.raises(CalibrationError, match="no GPS fix"):
        run(make_sim_rig(gps_always_fail=True))


def test_gps_failure_with_fallback_site_warns_and_continues():
    rig = make_sim_rig(gps_always_fail=True)
    rep = run(rig, fallback_site=site_from_geodetic(40.0, -105.0, 1600.0))
    assert rep.used_fallback_site
    assert rep.gps_time is None
    assert any("no GPS fix" in w for w in rep.warnings)
    assert rep.verification_error.to_value(u.arcsec) < 30


def test_a_star_that_cannot_be_centered_is_skipped():
    rig = make_sim_rig(solver_fail_first=3)  # the first star gets three solve failures
    rep = run(rig)
    assert "Arcturus" not in rep.stars
    assert any("could not center Arcturus" in w for w in rep.warnings)
    assert rep.model_rms.to_value(u.arcsec) < 15


def test_gives_up_after_too_many_failed_stars():
    rig = make_sim_rig()
    rig.solver = FakeSolver(rig.mount.true_radec, always_fail=True)
    with pytest.raises(CalibrationError, match="after 4 attempts"):
        run(rig)


def test_verification_can_be_skipped():
    rep = run(make_sim_rig(), verify=False)
    assert rep.verification_star is None
    assert rep.verification_error is None


def test_poor_model_is_flagged_by_verification():
    # An altitude-encoder zero error is not a rotation, so the model cannot fully absorb it.
    rig = make_sim_rig(frame_rotation=(0 * u.deg,) * 3, alignment_error=(0 * u.deg, 4 * u.deg))
    rep = run(rig)
    assert rep.verification_error.to_value(u.arcsec) > 120
    assert any("pointing model is poor" in w for w in rep.warnings)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_noise_seeds_all_calibrate(seed):
    rep = run(make_sim_rig(seed=seed, solver_noise=4 * u.arcsec))
    assert rep.verification_error.to_value(u.arcsec) < 40
