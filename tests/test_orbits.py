import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time
from scipy.optimize import brentq

from okapi.coords import site_from_geodetic
from okapi.orbits import MU_EARTH, R_EARTH
from okapi.orbits.fit import OrbitFitError, _ground_penalty, determine_orbit
from okapi.orbits.geometry import observer_position, radec_of, tangent_offset, unit_vector
from okapi.orbits.iod import IodError, gauss_iod
from okapi.orbits.simulate import geo_state, observe, overhead_leo_state
from okapi.orbits.twobody import Elements, elements_from_state, propagate, state_from_elements

SITE = site_from_geodetic(40.0, -105.3, 1600.0)
EPOCH = Time("2026-11-21T04:00:00")


def energy(r, v):
    return 0.5 * np.dot(v, v) - MU_EARTH / np.linalg.norm(r)


# -- two-body propagation -----------------------------------------------------------------------
def test_a_circular_orbit_closes_after_one_period_and_keeps_its_radius():
    r0 = np.array([7000.0, 0, 0])
    v0 = np.array([0, np.sqrt(MU_EARTH / 7000.0), 0])
    period = 2 * np.pi * np.sqrt(7000.0**3 / MU_EARTH)
    r, v = propagate(r0, v0, period)
    assert np.allclose(r, r0, atol=1e-6) and np.allclose(v, v0, atol=1e-9)
    r, _ = propagate(r0, v0, period / 4)
    assert np.allclose(r, [0, 7000, 0], atol=1e-6)


@pytest.mark.parametrize("e", [0.0, 0.3, 0.7, 0.95])
def test_energy_and_angular_momentum_are_conserved(e):
    el = Elements(20000.0, e, np.radians(50), np.radians(20), np.radians(70), np.radians(40))
    r0, v0 = state_from_elements(el)
    for dt in (100.0, 5000.0, 40000.0, -9000.0):
        r, v = propagate(r0, v0, dt)
        assert energy(r, v) == pytest.approx(energy(r0, v0), rel=1e-9)
        assert np.allclose(np.cross(r, v), np.cross(r0, v0), rtol=1e-8)


def test_propagation_agrees_with_keplers_equation_solved_independently():
    el = Elements(26000.0, 0.7, 0.0, 0.0, 0.0, 0.0)  # starts at perigee in the x-y plane
    r0, v0 = state_from_elements(el)
    dt = 9000.0
    n = np.sqrt(MU_EARTH / el.a**3)
    mean = n * dt
    ecc_anom = brentq(lambda x: x - el.e * np.sin(x) - mean, 0, 2 * np.pi)
    nu = 2 * np.arctan2(
        np.sqrt(1 + el.e) * np.sin(ecc_anom / 2), np.sqrt(1 - el.e) * np.cos(ecc_anom / 2)
    )
    expected = state_from_elements(Elements(el.a, el.e, 0.0, 0.0, 0.0, nu))[0]
    assert np.allclose(propagate(r0, v0, dt)[0], expected, atol=1e-4)


def test_a_hyperbolic_flyby_conserves_energy_and_forward_then_back_returns():
    r0 = np.array([8000.0, 0, 0])
    v0 = np.array([0, 12.0, 0])  # above escape speed
    assert energy(r0, v0) > 0
    r, v = propagate(r0, v0, 3000.0)
    assert energy(r, v) == pytest.approx(energy(r0, v0), rel=1e-9)
    rb, vb = propagate(r, v, -3000.0)
    assert np.allclose(rb, r0, atol=1e-6) and np.allclose(vb, v0, atol=1e-9)


def test_zero_time_is_identity():
    r0, v0 = np.array([7000.0, 100, 50]), np.array([0.1, 7.5, 0.2])
    r, v = propagate(r0, v0, 0.0)
    assert np.array_equal(r, r0) and np.array_equal(v, v0)


@pytest.mark.parametrize(
    "a,e,i,raan,argp,nu",
    [
        (7000, 0.01, 51.6, 100, 30, 200),
        (26000, 0.7, 63.4, 10, 270, 5),
        (42164, 0.001, 0.05, 80, 120, 300),
    ],
)
def test_elements_round_trip(a, e, i, raan, argp, nu):
    el = Elements(a, e, *np.radians([i, raan, argp, nu]))
    back = elements_from_state(*state_from_elements(el))
    assert back.a == pytest.approx(el.a, rel=1e-9)
    assert back.e == pytest.approx(el.e, abs=1e-9)
    assert back.i == pytest.approx(el.i, abs=1e-9)
    assert back.raan == pytest.approx(el.raan, abs=1e-8)
    assert back.argp == pytest.approx(el.argp, abs=1e-6)
    assert back.nu == pytest.approx(el.nu, abs=1e-6)


def test_period_and_perigee_of_a_known_orbit():
    el = Elements(42164.0, 0.0, 0, 0, 0, 0)
    assert el.period == pytest.approx(86164, rel=1e-3)  # a sidereal day
    assert Elements(10000.0, 0.3, 0, 0, 0, 0).perigee == pytest.approx(7000.0)


# -- geometry -------------------------------------------------------------------------------------
def test_the_observer_sits_on_the_earths_surface():
    radius = np.linalg.norm(observer_position(SITE, EPOCH))
    assert R_EARTH - 25 < radius < R_EARTH  # a 40 deg latitude site at 1.6 km altitude


def test_tangent_offsets_have_the_right_signs_and_scale():
    ra0, dec0 = 1.0, 0.5
    d = unit_vector(ra0, dec0)
    assert tangent_offset(d, ra0, dec0) == pytest.approx((0.0, 0.0), abs=1e-12)
    east, north = tangent_offset(unit_vector(ra0 + 0.001, dec0), ra0, dec0)
    assert east > 0 and east == pytest.approx(0.001 * np.cos(dec0), rel=1e-3) and abs(north) < 1e-6
    east, north = tangent_offset(unit_vector(ra0, dec0 + 0.001), ra0, dec0)
    assert north == pytest.approx(0.001, rel=1e-3) and abs(east) < 1e-9


def test_radec_unit_vector_round_trip():
    ra, dec = radec_of(unit_vector(4.0, -0.6))
    assert (ra, dec) == pytest.approx((4.0, -0.6))


# -- simulator -------------------------------------------------------------------------------------
def test_the_simulated_geo_satellite_is_geostationary_over_its_longitude():
    r, v = geo_state(-100.0, EPOCH)
    assert np.linalg.norm(r) == pytest.approx(42164, abs=5)
    assert energy(r, v) == pytest.approx(-MU_EARTH / (2 * np.linalg.norm(r)), rel=1e-9)
    # Geostationary: the range from a ground site stays constant, and the line of sight turns in
    # the inertial frame at the Earth's rotation rate (15.04 deg/h).
    d0 = r - observer_position(SITE, EPOCH)
    r2, _ = propagate(r, v, 3600.0)
    d1 = r2 - observer_position(SITE, EPOCH + 3600 * u.s)
    assert abs(np.linalg.norm(d1) - np.linalg.norm(d0)) < 10.0  # km
    sep = np.degrees(np.arccos(np.dot(d0, d1) / np.linalg.norm(d0) / np.linalg.norm(d1)))
    assert sep == pytest.approx(15.04, abs=0.3)


def test_the_overhead_leo_pass_goes_through_the_zenith():
    r, _v = overhead_leo_state(SITE, EPOCH, 600.0, 90.0)
    d = r - observer_position(SITE, EPOCH)
    assert np.linalg.norm(d) == pytest.approx(600.0, rel=0.02)
    up = observer_position(SITE, EPOCH)
    assert np.degrees(np.arccos(np.dot(d, up) / np.linalg.norm(d) / np.linalg.norm(up))) < 1.0


# -- Gauss IOD -------------------------------------------------------------------------------------
def _gauss_inputs(r0, v0, times):
    rho, obs = [], []
    for t in times:
        r, _ = propagate(r0, v0, t)
        o = observer_position(SITE, EPOCH + t * u.s)
        d = r - o
        rho.append(d / np.linalg.norm(d))
        obs.append(o)
    return rho, obs


def test_gauss_recovers_a_geo_orbit_from_three_well_separated_noise_free_observations():
    r0, v0 = geo_state(-100.0, EPOCH)
    times = [0.0, 3600.0, 7200.0]
    rho, obs = _gauss_inputs(r0, v0, times)
    r2, v2, t2 = gauss_iod(rho, times, obs)
    rt, vt = propagate(r0, v0, t2)
    assert np.linalg.norm(r2 - rt) / np.linalg.norm(rt) < 0.02
    assert np.linalg.norm(v2 - vt) / np.linalg.norm(vt) < 0.05


def test_gauss_rejects_unordered_times():
    r0, v0 = geo_state(-100.0, EPOCH)
    rho, obs = _gauss_inputs(r0, v0, [0.0, 3600.0, 7200.0])
    with pytest.raises(IodError, match="increase"):
        gauss_iod(rho, [0.0, 7200.0, 3600.0], obs)


# -- the fit: what it recovers, and what it admits it cannot -------------------------------------
def solution_error(sol, r0, v0):
    """Error against the truth propagated to the solution's own epoch."""
    rt, vt = propagate(r0, v0, float((sol.epoch - EPOCH).to_value(u.s)))
    return sol.r - rt, sol.v - vt


@pytest.fixture(scope="module")
def geo():
    return geo_state(-100.0, EPOCH)


def test_geo_orbit_recovered_from_five_20s_stares_over_two_hours(geo):
    r0, v0 = geo
    obs = observe(r0, v0, EPOCH, SITE, [0, 1800, 3600, 5400, 7200], 20 * u.s, kind="point")
    sol = determine_orbit(obs, SITE)
    dr, dv = solution_error(sol, r0, v0)
    assert np.linalg.norm(dr) < 5.0  # km, of 42164
    assert np.linalg.norm(dv) < 0.002  # km/s
    assert np.all(np.abs(dr) < 4 * sol.r_sigma)  # the stated uncertainty is honest
    assert 0.1 < sol.chi2_reduced < 3
    assert sol.elements.a == pytest.approx(42164, abs=30)
    assert sol.warnings == ()


def test_geo_timing_bias_shifts_the_orbit_along_track_by_speed_times_bias(geo):
    r0, v0 = geo
    errs = {}
    for bias in (0.0, 1.0):
        obs = observe(
            r0,
            v0,
            EPOCH,
            SITE,
            [0, 1800, 3600, 5400, 7200],
            20 * u.s,
            kind="point",
            timing_error=bias * u.s,
        )
        sol = determine_orbit(obs, SITE)
        # compare against where the object really was at the time the camera RECORDED
        rt, _ = propagate(r0, v0, float((sol.epoch - EPOCH).to_value(u.s)))
        errs[bias] = np.linalg.norm(sol.r - rt)
    assert errs[1.0] > errs[0.0] + 1.0
    assert errs[1.0] == pytest.approx(3.07, abs=1.5)  # GEO moves 3.07 km/s


def test_fewer_observations_than_constraints_is_refused_not_guessed(geo):
    r0, v0 = geo
    one = observe(r0, v0, EPOCH, SITE, [0], 20 * u.s, kind="point")
    with pytest.raises(OrbitFitError, match="at least two"):
        determine_orbit(one, SITE)


@pytest.fixture(scope="module")
def leo():
    r, v = overhead_leo_state(SITE, EPOCH + 30 * u.s, 600.0, 90.0)
    return propagate(r, v, -30.0)


def test_leo_from_a_thirty_second_arc_is_reported_as_poorly_constrained_but_consistent(leo):
    r0, v0 = leo
    obs = observe(r0, v0, EPOCH, SITE, [15 + 3 * k for k in range(10)], 0.2 * u.s, kind="point")
    sol = determine_orbit(obs, SITE)
    dr, _ = solution_error(sol, r0, v0)
    assert np.all(np.abs(dr) < 4 * sol.r_sigma)  # the truth is inside the stated uncertainty
    assert sol.r_sigma.max() > 50.0  # ... which is large: range is the unknown
    assert any("poorly conditioned" in w for w in sol.warnings)
    assert sol.elements.perigee > R_EARTH  # a physical orbit, not one inside the Earth


def test_leo_field_crossing_lines_are_weaker_than_timed_points(leo):
    r0, v0 = leo
    pts = determine_orbit(
        observe(r0, v0, EPOCH, SITE, [15 + 3 * k for k in range(10)], 0.2 * u.s, kind="point"), SITE
    )
    lines = determine_orbit(
        observe(r0, v0, EPOCH, SITE, [0, 15, 30, 45, 60], 10 * u.s, kind="line"), SITE
    )
    assert lines.condition > pts.condition
    assert any("poorly conditioned" in w for w in lines.warnings)


def test_ground_penalty_is_zero_for_a_safe_orbit_and_positive_for_a_diving_one():
    r = np.array([7000.0, 0, 0])
    assert _ground_penalty(r, np.array([0, np.sqrt(MU_EARTH / 7000), 0])) == 0.0
    assert _ground_penalty(r, np.array([0, 3.0, 0])) > 0.0  # perigee far below the surface
