import astropy.units as u
import pytest
from astropy.time import Time

from okapi.astrometry.identify import Gates, identify
from okapi.coords import site_from_geodetic
from okapi.orbits.simulate import geo_state, observe, overhead_leo_state
from okapi.orbits.twobody import propagate
from okapi.targets import FixedTarget, GeoSlotTarget, StateTarget

SITE = site_from_geodetic(40.0, -105.3, 1600.0)
EPOCH = Time("2026-11-21T04:00:00")


def geo_obs(lon, noise=1 * u.arcsec, seed=0, start=600.0):
    r, v = geo_state(lon, EPOCH)
    return observe(r, v, EPOCH, SITE, [start], 20 * u.s, kind="point", noise=noise, seed=seed)[0]


def geo_target(lon, name=None):
    r, v = geo_state(lon, EPOCH)
    return StateTarget(name or f"geo{lon}", r, v, EPOCH)


def test_a_geo_streak_is_matched_to_its_own_orbit_among_other_slots():
    obs = geo_obs(-100.0)
    cands = [geo_target(-100.0), geo_target(-95.0), geo_target(-110.0)]
    m = identify(obs, cands, SITE)
    assert m is not None and m.target == "geo-100.0"
    assert not m.ambiguous
    assert m.offset.to_value(u.arcsec) < 5
    assert m.rate_ratio == pytest.approx(1.0, abs=0.02)


def test_a_streak_with_no_matching_object_is_unmatched():
    assert identify(geo_obs(-100.0), [geo_target(-80.0), geo_target(-120.0)], SITE) is None


def test_two_objects_a_hair_apart_are_flagged_ambiguous_not_silently_picked():
    obs = geo_obs(-100.0)
    cands = [geo_target(-100.02, "a"), geo_target(-99.98, "b")]
    m = identify(obs, cands, SITE)
    assert m is not None and m.ambiguous


def test_a_large_timing_error_breaks_the_match_for_a_fast_object():
    r, v = overhead_leo_state(SITE, EPOCH + 30 * u.s, 600.0, 90.0)
    r0, v0 = propagate(r, v, -30.0)
    target = StateTarget("leo", r0, v0, EPOCH)
    good = observe(r0, v0, EPOCH, SITE, [25.0], 0.2 * u.s, kind="point")[0]
    assert identify(good, [target], SITE) is not None
    late = observe(r0, v0, EPOCH, SITE, [25.0], 0.2 * u.s, kind="point", timing_error=5 * u.s)[
        0
    ]  # recorded 5 s late: the LEO object has moved ~37 km (several degrees) since
    assert identify(late, [target], SITE) is None


def test_the_wrong_direction_of_motion_is_rejected():
    r, v = overhead_leo_state(SITE, EPOCH + 30 * u.s, 600.0, 90.0)
    r0, v0 = propagate(r, v, -30.0)
    obs = observe(r0, v0, EPOCH, SITE, [25.0], 0.2 * u.s, kind="point")[0]
    # a satellite at the same place moving at right angles to the observed one
    r2, v2 = overhead_leo_state(SITE, EPOCH + 30 * u.s, 600.0, 0.0)
    cross = StateTarget("cross", *propagate(r2, v2, -30.0), EPOCH)
    assert identify(obs, [cross], SITE) is None


def test_a_field_crossing_line_matches_on_track_and_direction_not_along_track_position():
    r, v = overhead_leo_state(SITE, EPOCH + 30 * u.s, 600.0, 90.0)
    r0, v0 = propagate(r, v, -30.0)
    target = StateTarget("leo", r0, v0, EPOCH)
    line = observe(r0, v0, EPOCH, SITE, [20.0], 10 * u.s, kind="line", noise=1 * u.arcsec)[0]
    m = identify(line, [target], SITE)
    assert m is not None and m.rate_ratio is None
    assert m.offset.to_value(u.deg) < 0.05
    other = StateTarget(
        "other", *propagate(*overhead_leo_state(SITE, EPOCH, 600.0, 20.0), 0.0), EPOCH
    )
    assert identify(line, [other], SITE) is None


def test_fixed_targets_and_non_satellites_are_never_candidates():
    obs = geo_obs(-100.0)
    star = FixedTarget("star", obs.coord)
    assert identify(obs, [star], SITE) is None


def test_gates_can_be_loosened_to_admit_a_poorer_match():
    obs = geo_obs(-100.0)
    near = geo_target(-99.0)  # about 1 degree off in longitude: ~0.7 deg on the sky
    assert identify(obs, [near], SITE) is None
    assert identify(obs, [near], SITE, Gates(position=3 * u.deg)) is not None


def test_slot_target_object_works_as_a_candidate_too():
    obs = geo_obs(-100.0)
    assert identify(obs, [GeoSlotTarget("slot", -100 * u.deg)], SITE) is not None
