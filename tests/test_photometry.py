import math

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from okapi.astrometry.photometry import (
    PhotometryError,
    aperture_flux,
    fit_zero_point,
    limiting_magnitude,
    magnitude,
    streak_limiting_magnitude,
)
from okapi.astrometry.simulate import SimStar, render_frame, tangent_wcs

SHAPE = (400, 600)
SIGMA = 1.5
TRUE_ZP = 20.0


def counts(mag):
    return 10 ** ((TRUE_ZP - mag) / 2.5)


def peak_for(mag):
    return counts(mag) / (2 * math.pi * SIGMA**2)


def scene(mags, noise=5.0, seed=1):
    wcs = tangent_wcs(150.0, 20.0, SHAPE, 1.5)
    positions = [(80 + 90 * i, 100 + 40 * (i % 3)) for i in range(len(mags))]
    img = render_frame(
        SHAPE,
        stars=[SimStar(x, y, peak_for(m)) for (x, y), m in zip(positions, mags, strict=True)],
        noise=noise,
        sky=300.0,
        seed=seed,
    )
    refs = [(wcs.pixel_to_world(x, y).icrs, m) for (x, y), m in zip(positions, mags, strict=True)]
    return img, wcs, refs


def test_aperture_flux_recovers_the_injected_total():
    img, _, _ = scene([10.0])
    flux, noise, _ = aperture_flux(img, 80, 100)
    assert flux == pytest.approx(counts(10.0), rel=0.03)
    assert noise == pytest.approx(5.0, rel=0.25)


def test_zero_point_recovered_from_several_stars():
    img, wcs, refs = scene([9.0, 10.0, 11.0, 10.5])
    zp = fit_zero_point(img, wcs, refs)
    assert zp.value == pytest.approx(TRUE_ZP, abs=0.05)
    assert zp.n_stars == 4 and zp.scatter < 0.08


def test_saturated_stars_are_dropped_and_do_not_bias_the_result():
    img, wcs, refs = scene([2.0, 10.0, 11.0])
    zp = fit_zero_point(img, wcs, refs, saturation=60000)
    assert zp.n_stars == 2 and zp.value == pytest.approx(TRUE_ZP, abs=0.05)


def test_no_usable_stars_is_an_error():
    img, wcs, refs = scene([1.0, 2.0])
    with pytest.raises(PhotometryError, match="no usable"):
        fit_zero_point(img, wcs, refs, saturation=100)
    off = [(SkyCoord(10 * u.deg, -60 * u.deg), 9.0)]
    with pytest.raises(PhotometryError):
        fit_zero_point(img, wcs, off)


def test_magnitude_round_trip_and_limits_are_ordered():
    img, wcs, refs = scene([9.0, 10.0, 11.0])
    zp = fit_zero_point(img, wcs, refs)
    assert magnitude(counts(12.5), zp) == pytest.approx(12.5, abs=0.06)
    point = limiting_magnitude(zp)
    assert 12 < point < 17
    # a 300 px streak must be brighter (smaller magnitude) than a point to be detected
    assert streak_limiting_magnitude(zp, 300) < point
    assert limiting_magnitude(zp, snr=10) < point
    with pytest.raises(PhotometryError):
        magnitude(0.0, zp)
