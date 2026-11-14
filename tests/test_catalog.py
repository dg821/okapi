import astropy.units as u
import pytest
from astropy.time import Time

from okapi.catalog import (
    STARS,
    get_star,
    partner_candidates,
    pick_alignment_pair,
    visible_stars,
)
from okapi.coords import radec_to_altaz

T = Time("2026-06-21T04:30:00", scale="utc")

# Well-known angular separations (degrees), independent of how the catalog was typed in.
KNOWN_SEPARATIONS = [
    ("Vega", "Altair", 34.1),
    ("Deneb", "Vega", 23.8),
    ("Altair", "Deneb", 38.0),
    ("Dubhe", "Polaris", 28.5),
    ("Sirius", "Procyon", 25.6),
    ("Betelgeuse", "Rigel", 18.5),
]


@pytest.mark.parametrize(("a", "b", "expected"), KNOWN_SEPARATIONS)
def test_known_separations(a, b, expected):
    sep = get_star(a).coord().separation(get_star(b).coord()).to_value(u.deg)
    assert sep == pytest.approx(expected, abs=0.4)


def test_catalog_names_are_unique_and_magnitudes_plausible():
    names = [s.name for s in STARS]
    assert len(names) == len(set(names))
    assert all(-2 < s.vmag < 3 for s in STARS)


def test_unknown_star():
    with pytest.raises(KeyError, match="Nopeus"):
        get_star("Nopeus")


def test_proper_motion_moves_arcturus_about_sixty_arcsec_since_j2000():
    s = get_star("Arcturus")
    shift = s.coord().separation(s.coord(T)).to_value(u.arcsec)
    assert 55 < shift < 66  # 2.28 arcsec/yr * 26.5 yr


def test_no_epoch_means_j2000():
    s = get_star("Vega")
    assert s.coord().separation(s.coord(Time("J2000"))).to_value(u.arcsec) < 1e-6


def test_visible_stars_respect_altitude_window_and_order(site):
    stars = visible_stars(site, T, min_alt=30 * u.deg, max_alt=70 * u.deg)
    assert stars
    for s in stars:
        alt = radec_to_altaz(s.coord(T), site, T).alt.to_value(u.deg)
        assert 30 <= alt <= 70
    assert [s.vmag for s in stars] == sorted(s.vmag for s in stars)


def test_exclude_removes_stars(site):
    first = visible_stars(site, T)[0]
    assert first.name not in [s.name for s in visible_stars(site, T, exclude=(first.name,))]


def test_pair_is_bright_and_well_separated(site):
    a, b = pick_alignment_pair(site, T)
    sep = a.coord(T).separation(b.coord(T)).to_value(u.deg)
    assert 60 <= sep <= 150
    assert a is visible_stars(site, T)[0]


def test_partner_candidates_never_include_first_or_excluded(site):
    first = visible_stars(site, T)[0]
    cands = partner_candidates(first, site, T, exclude=("Vega",))
    assert first.name not in [s.name for s in cands]
    assert "Vega" not in [s.name for s in cands]


def test_no_pair_when_everything_excluded(site):
    with pytest.raises(LookupError):
        pick_alignment_pair(site, T, exclude=tuple(s.name for s in STARS))
