import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from okapi.astrometry.observation import streak_to_observation
from okapi.astrometry.simulate import SimStar, SimStreak, render_frame, tangent_wcs
from okapi.astrometry.streaks import background, detect_streaks

SHAPE = (1000, 1500)


def stars(n=60, seed=1):
    rng = np.random.default_rng(seed)
    return [
        SimStar(rng.uniform(5, 1495), rng.uniform(5, 995), rng.uniform(150, 4000)) for _ in range(n)
    ]


def test_background_ignores_bright_objects():
    img = render_frame(SHAPE, stars=stars(), sky=500, noise=8)
    level, noise = background(img)
    assert level == pytest.approx(500, abs=1)
    assert noise == pytest.approx(8, rel=0.1)


def test_star_field_alone_has_no_streaks():
    assert detect_streaks(render_frame(SHAPE, stars=stars())) == []


def test_a_contained_streak_is_found_with_endpoints_within_a_pixel_or_two():
    k = SimStreak(300, 400, 700, 600, peak=120)
    found = detect_streaks(render_frame(SHAPE, stars=stars(), streaks=[k]))
    assert len(found) == 1
    s = found[0]
    ends = sorted([(s.x0, s.y0), (s.x1, s.y1)])
    for got, want in zip(ends, [(300, 400), (700, 600)], strict=True):
        assert np.hypot(got[0] - want[0], got[1] - want[1]) < 4
    assert not s.touches_edge
    assert s.length == pytest.approx(np.hypot(400, 200), abs=2.5)


def test_a_streak_crossing_the_frame_is_marked_as_touching_the_edge():
    k = SimStreak(0, 300, 1499, 700, peak=120)
    found = detect_streaks(render_frame(SHAPE, streaks=[k]))
    assert len(found) == 1 and found[0].touches_edge


def test_a_streak_is_found_through_stars_it_crosses():
    k = SimStreak(200, 500, 900, 500, peak=120)
    crossing = [SimStar(x, 500, 3000) for x in (350, 600)]
    found = detect_streaks(render_frame(SHAPE, stars=stars() + crossing, streaks=[k]))
    assert len(found) == 1
    assert found[0].length == pytest.approx(700, abs=15)


def test_two_streaks_are_both_found_brightest_first():
    ks = [SimStreak(100, 100, 400, 200, 80), SimStreak(800, 800, 1300, 600, 160)]
    found = detect_streaks(render_frame(SHAPE, streaks=ks))
    assert len(found) == 2 and found[0].flux > found[1].flux


def test_short_blobs_and_hot_pixels_are_not_streaks():
    img = render_frame(SHAPE, stars=stars())
    img[100:103, 100] += 5000  # a hot column fragment
    img[800, 800] += 20000
    assert detect_streaks(img) == []


def test_a_very_faint_streak_is_not_invented():
    k = SimStreak(300, 400, 700, 600, peak=0.5)
    assert detect_streaks(render_frame(SHAPE, streaks=[k], noise=8)) == []


def test_contained_streak_becomes_a_timed_point_with_rate_and_position_angle():
    wcs = tangent_wcs(150.0, 20.0, SHAPE)
    k = SimStreak(300, 500, 700, 500, peak=120)  # 400 px due west... along x
    s = detect_streaks(render_frame(SHAPE, streaks=[k]))[0]
    start = Time("2026-11-21T04:00:00")
    obs = streak_to_observation(s, wcs, start=start, exposure=20 * u.s)
    assert obs.kind == "point"
    assert (obs.time - start).to_value(u.s) == pytest.approx(10)
    # 400 px * 0.74 arcsec over 20 s = 14.8 arcsec/s, about the sidereal rate
    assert obs.rate.to_value(u.arcsec / u.s) == pytest.approx(14.8, rel=0.03)
    # x runs along RA here, so the streak lies east-west: position angle near 90 deg
    assert obs.position_angle.to_value(u.deg) == pytest.approx(90, abs=1.5)
    truth = wcs.pixel_to_world_values(500, 500)
    assert (
        obs.coord.separation(
            __import__("astropy.coordinates", fromlist=["SkyCoord"]).SkyCoord(
                truth[0] * u.deg, truth[1] * u.deg
            )
        ).to_value(u.arcsec)
        < 3
    )


def test_edge_streak_becomes_a_line_with_no_rate_and_a_wide_time_uncertainty():
    wcs = tangent_wcs(150.0, 20.0, SHAPE)
    s = detect_streaks(render_frame(SHAPE, streaks=[SimStreak(0, 300, 1499, 700, 120)]))[0]
    obs = streak_to_observation(s, wcs, start=Time("2026-11-21T04:00:00"), exposure=10 * u.s)
    assert obs.kind == "line" and obs.rate is None
    assert obs.time_sigma == 5 * u.s


# -- stars touching the streak must not bias its endpoints --------------------------------------
def test_a_star_in_the_middle_of_a_streak_does_not_bias_it_or_flag_it():
    k = SimStreak(300, 400, 600, 400, peak=120)
    found = detect_streaks(
        render_frame(SHAPE, stars=[SimStar(450, 403, 3500)], streaks=[k], seed=4)
    )
    assert len(found) == 1 and not found[0].contaminated
    s = found[0]
    ends = sorted([(s.x0, s.y0), (s.x1, s.y1)])
    assert abs(ends[0][0] - 300) < 1.5 and abs(ends[1][0] - 600) < 1.5
    assert s.length == pytest.approx(300, abs=2.0)


@pytest.mark.parametrize("star_at,true_end", [((296, 402), 300), ((604, 398), 600)])
def test_a_star_on_an_end_is_flagged_and_the_observation_admits_the_uncertainty(star_at, true_end):
    k = SimStreak(300, 400, 600, 400, peak=120)
    found = detect_streaks(
        render_frame(SHAPE, stars=[SimStar(*star_at, peak=3500)], streaks=[k], seed=4)
    )
    assert len(found) == 1
    s = found[0]
    assert s.contaminated
    ends = sorted([(s.x0, s.y0), (s.x1, s.y1)])
    got = ends[0][0] if true_end == 300 else ends[1][0]
    wcs = tangent_wcs(150.0, 20.0, SHAPE)
    obs = streak_to_observation(s, wcs, start=Time("2026-11-21T04:00:00"), exposure=20 * u.s)
    sigma_px = obs.sigma.to_value(u.arcsec) / 0.74
    assert sigma_px >= 4.0  # inflated from the usual 0.5 px
    assert abs(got - true_end) < 2 * sigma_px  # and the stated uncertainty covers the real error


def test_an_uncontaminated_streak_keeps_the_tight_uncertainty():
    s = detect_streaks(render_frame(SHAPE, streaks=[SimStreak(300, 400, 600, 400, 120)]))[0]
    assert not s.contaminated
    obs = streak_to_observation(
        s, tangent_wcs(150.0, 20.0, SHAPE), start=Time("2026-11-21T04:00:00"), exposure=20 * u.s
    )
    assert obs.sigma.to_value(u.arcsec) / 0.74 == pytest.approx(0.5)


def test_streak_endpoints_are_accurate_to_a_fraction_of_a_pixel_on_a_clean_frame():
    k = SimStreak(210.3, 355.7, 640.9, 520.2, peak=150)
    s = detect_streaks(render_frame(SHAPE, streaks=[k], seed=9))[0]
    ends = sorted([(s.x0, s.y0), (s.x1, s.y1)])
    for got, want in zip(ends, [(210.3, 355.7), (640.9, 520.2)], strict=True):
        assert np.hypot(got[0] - want[0], got[1] - want[1]) < 1.0
