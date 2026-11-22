import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from tests.tle_helpers import make_tle, revs_per_day_for

from okapi.coords import site_from_geodetic
from okapi.orbits.simulate import geo_state
from okapi.targets import (
    FixedTarget,
    GeoSlotTarget,
    StateTarget,
    TargetError,
    TleTarget,
    parse_tles,
)

SITE = site_from_geodetic(40.0, -105.3, 1600.0)
T0 = Time("2026-11-21T04:00:00")  # day of year 325


def test_fixed_target_does_not_move():
    c = SkyCoord(10 * u.deg, 20 * u.deg)
    t = FixedTarget("star", c)
    assert t.predict(SITE, T0) == t.predict(SITE, T0 + 1 * u.day) and not t.is_satellite


def test_geo_slot_stays_put_relative_to_the_ground_but_moves_against_the_stars():
    t = GeoSlotTarget("slot", -100 * u.deg)
    a, b = t.predict(SITE, T0), t.predict(SITE, T0 + 3600 * u.s)
    assert t.is_satellite
    assert a.separation(b).to_value(u.deg) == pytest.approx(15.04, abs=0.3)  # sidereal rate
    # the same slot a sidereal day later is back at the same RA/Dec
    c = t.predict(SITE, T0 + 86164.0905 * u.s)
    assert a.separation(c).to_value(u.arcsec) < 60


def test_state_target_matches_the_state_it_was_built_from():
    r, v = geo_state(-100.0, T0)
    st = StateTarget("sat", r, v, T0)
    slot = GeoSlotTarget("slot", -100 * u.deg)
    assert st.predict(SITE, T0).separation(slot.predict(SITE, T0)).to_value(u.arcsec) < 1.0
    # an hour later the two-body orbit and the slot still agree (both geostationary)
    t1 = T0 + 3600 * u.s
    assert st.predict(SITE, t1).separation(slot.predict(SITE, t1)).to_value(u.arcsec) < 60


@pytest.mark.parametrize("a_km", [6978.0, 7500.0, 26560.0])
def test_tle_target_lies_at_the_expected_distance_from_the_earth(a_km):
    l1, l2 = make_tle(revs_per_day=revs_per_day_for(a_km), eccentricity=0.0001)
    tle = TleTarget("x", l1, l2)
    # direction is all predict() gives; check it via the geocentric geometry instead:
    from astropy.coordinates import GCRS, TEME, CartesianRepresentation

    err, r, _ = tle._sat.sgp4(float(T0.utc.jd1), float(T0.utc.jd2))
    assert err == 0
    teme = TEME(CartesianRepresentation(*(x * u.km for x in r)), obstime=T0)
    xyz = np.asarray(teme.transform_to(GCRS(obstime=T0)).cartesian.xyz.to_value(u.km), float)
    # SGP4's Kozai mean motion differs from Kepler's by up to ~0.3% for LEO
    assert np.linalg.norm(xyz) == pytest.approx(a_km, rel=0.004)
    assert tle.predict(SITE, T0).separation(tle.predict(SITE, T0 + 1 * u.s)).to_value(u.arcsec) > 1


def test_tle_target_agrees_with_the_geo_slot_for_a_geostationary_element_set():
    # a geostationary set: one sidereal revolution per day, equatorial, circular
    rev = 86400.0 / 86164.0905
    l1, l2 = make_tle(
        inclination_deg=0.01, eccentricity=0.0001, revs_per_day=rev, raan_deg=0.0, argp_deg=0.0
    )
    # choose the mean anomaly that puts it near -100 deg longitude at the epoch
    r, _ = geo_state(-100.0, Time("2026-11-21T12:00:00"))
    lon_inertial = np.degrees(np.arctan2(r[1], r[0])) % 360
    l1, l2 = make_tle(
        inclination_deg=0.01,
        eccentricity=0.0001,
        revs_per_day=rev,
        raan_deg=0.0,
        argp_deg=0.0,
        mean_anomaly_deg=lon_inertial,
        epoch_day=325.5,
    )
    tle = TleTarget("geo", l1, l2)
    slot = GeoSlotTarget("slot", -100 * u.deg)
    t = Time("2026-11-21T12:00:00")  # the element epoch (day 325.5)
    assert tle.predict(SITE, t).separation(slot.predict(SITE, t)).to_value(u.deg) < 0.5


def test_bad_tle_lines_are_rejected():
    with pytest.raises(TargetError, match="start with"):
        TleTarget("x", "garbage", "more garbage")
    l1, l2 = make_tle()
    with pytest.raises(TargetError, match="69 characters"):
        TleTarget("x", l1[:30], l2)


def test_parse_tles_reads_two_and_three_line_sets():
    a1, a2 = make_tle(catalog=11111)
    b1, b2 = make_tle(catalog=22222, inclination_deg=98.0)
    got = parse_tles(f"{a1}\n{a2}\n\nNAMED SAT\n{b1}\n{b2}\n")
    assert [g.name for g in got] == ["NORAD 11111", "NAMED SAT"]
    with pytest.raises(TargetError, match="unexpected"):
        parse_tles("hello\nworld\n")


def test_a_decayed_satellite_raises_a_clear_error():
    l1, l2 = make_tle(revs_per_day=16.4, eccentricity=0.0001, bstar=" 99999-1")
    tle = TleTarget("x", l1, l2)
    with pytest.raises(TargetError, match="SGP4 error"):
        tle.predict(SITE, T0 + 200 * u.day)


def test_a_tle_with_a_corrupted_digit_fails_its_checksum():
    l1, l2 = make_tle()
    bad = l2[:20] + ("1" if l2[20] != "1" else "2") + l2[21:]
    with pytest.raises(TargetError, match="checksum"):
        TleTarget("x", l1, bad)
