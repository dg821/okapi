import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time
from hypothesis import given, settings
from hypothesis import strategies as st

from okapi.coords import AltAzPosition, altaz_to_radec, radec_to_altaz, site_from_geodetic

T = Time("2026-06-21T04:00:00", scale="utc")


def test_polaris_altitude_is_about_latitude(site):
    polaris = SkyCoord(ra="02h31m49s", dec="+89d15m51s")
    pos = radec_to_altaz(polaris, site, T)
    assert abs(pos.alt.to_value(u.deg) - 40.0) < 1.2  # Polaris is ~0.7 deg off the pole
    assert min(pos.az.to_value(u.deg), 360 - pos.az.to_value(u.deg)) < 2.0  # due north


def test_object_on_meridian_at_dec_equal_lat_is_at_zenith(site):
    # Sidereal time at the site gives the RA currently on the meridian.
    lst = T.sidereal_time("apparent", longitude=site.lon)
    star = SkyCoord(ra=lst, dec=40 * u.deg)
    pos = radec_to_altaz(star, site, T)
    assert pos.alt.to_value(u.deg) > 89.5


@given(az=st.floats(0, 359.9), alt=st.floats(10, 80))
@settings(max_examples=40, deadline=None)
def test_altaz_radec_roundtrip(az, alt):
    site = site_from_geodetic(40.0, -105.0, 1600.0)
    pos = AltAzPosition(az * u.deg, alt * u.deg)
    back = radec_to_altaz(altaz_to_radec(pos, site, T), site, T)
    assert pos.separation(back).to_value(u.arcsec) < 0.01


def test_azimuth_wraps_into_0_360():
    assert AltAzPosition(-10 * u.deg, 20 * u.deg).az.to_value(u.deg) == pytest.approx(350)
    assert AltAzPosition(370 * u.deg, 20 * u.deg).az.to_value(u.deg) == pytest.approx(10)


def test_separation_is_zero_for_same_point_and_wraps():
    a = AltAzPosition(359.9 * u.deg, 0.0 * u.deg)
    b = AltAzPosition(0.1 * u.deg, 0.0 * u.deg)
    assert a.separation(a).to_value(u.deg) == pytest.approx(0, abs=1e-9)
    assert a.separation(b).to_value(u.deg) == pytest.approx(0.2, abs=1e-6)


@pytest.mark.parametrize("lat", [-90.1, 90.1])
def test_site_rejects_bad_latitude(lat):
    with pytest.raises(ValueError):
        site_from_geodetic(lat, 0.0)
